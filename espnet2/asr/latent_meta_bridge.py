"""Latent-space Meta-BRIDGE for test-time ASR representation alignment."""

from collections import OrderedDict
import copy
from contextlib import nullcontext
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentResidualAdapter(nn.Module):
    """Identity-initialized bottleneck adapter for ASR latent sequences."""

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int = 32,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("feature and bottleneck dimensions must be positive")
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")

        self.feature_dim = feature_dim
        self.residual_scale = residual_scale
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        features: torch.Tensor,
        parameters: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if parameters is None:
            residual = self.up(F.gelu(self.down(self.norm(features))))
        else:
            normalized = F.layer_norm(
                features,
                (self.feature_dim,),
                parameters["norm.weight"],
                parameters["norm.bias"],
                self.norm.eps,
            )
            hidden = F.linear(
                normalized,
                parameters["down.weight"],
                parameters["down.bias"],
            )
            residual = F.linear(
                F.gelu(hidden),
                parameters["up.weight"],
                parameters["up.bias"],
            )
        return features + self.residual_scale * residual


class LatentMetaBridge(nn.Module):
    """Adapt a small latent aligner with masked EMA-target prediction.

    The support objective is entirely self-supervised: the online branch sees
    a masked, adapter-aligned ASR latent sequence and predicts stop-gradient
    latents from an unmasked EMA target encoder.  Only fast copies of the
    adapter parameters are changed in the inner loop.  The resulting unmasked
    aligned sequence is returned to the frozen remainder of the ASR encoder.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 112,
        adapter_dim: int = 32,
        adapter_scale: float = 1.0,
        predictor_layers: int = 1,
        predictor_heads: int = 4,
        predictor_ff_dim: int = 224,
        dropout_rate: float = 0.0,
        mask_ratio: float = 0.15,
        mask_span: int = 3,
        inner_lr: float = 0.1,
        inner_steps: int = 1,
        first_order: bool = False,
        gradient_clip: float = 1.0,
        adapt_during_training: bool = True,
        adapt_at_inference: bool = True,
        train_per_sample: bool = True,
        ema_decay: float = 0.999,
        inference_seed: int = 0,
        loss_type: str = "cosine",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        if predictor_layers <= 0:
            raise ValueError("predictor_layers must be positive")
        if latent_dim % predictor_heads != 0:
            raise ValueError("latent_dim must be divisible by predictor_heads")
        if not 0.0 < mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in (0, 1]")
        if mask_span <= 0:
            raise ValueError("mask_span must be positive")
        if inner_lr <= 0.0 or inner_steps < 0:
            raise ValueError("inner_lr must be positive and inner_steps non-negative")
        if gradient_clip < 0.0:
            raise ValueError("gradient_clip must be non-negative")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if loss_type not in ("cosine", "smooth_l1"):
            raise ValueError("loss_type must be 'cosine' or 'smooth_l1'")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio
        self.mask_span = mask_span
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.gradient_clip = gradient_clip
        self.adapt_during_training = adapt_during_training
        self.adapt_at_inference = adapt_at_inference
        self.train_per_sample = train_per_sample
        self.ema_decay = ema_decay
        self.inference_seed = inference_seed
        self.loss_type = loss_type

        self.adapter = LatentResidualAdapter(
            input_dim,
            bottleneck_dim=adapter_dim,
            residual_scale=adapter_scale,
        )
        self.online_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.target_encoder.eval()

        predictor_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=predictor_heads,
            dim_feedforward=predictor_ff_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            predictor_layer,
            num_layers=predictor_layers,
            norm=nn.LayerNorm(latent_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self._last_stats: Dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        online_parameters = dict(self.online_encoder.named_parameters())
        for name, target_parameter in self.target_encoder.named_parameters():
            target_parameter.mul_(self.ema_decay).add_(
                online_parameters[name], alpha=1.0 - self.ema_decay
            )
        online_buffers = dict(self.online_encoder.named_buffers())
        for name, target_buffer in self.target_encoder.named_buffers():
            if name in online_buffers and torch.is_floating_point(target_buffer):
                target_buffer.mul_(self.ema_decay).add_(
                    online_buffers[name], alpha=1.0 - self.ema_decay
                )

    def get_stats(self) -> Dict[str, torch.Tensor]:
        return self._last_stats.copy()

    def forward(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if base_features.dim() != 3 or base_features.size(-1) != self.input_dim:
            raise ValueError(
                "Expected latent features shaped (B, T, input_dim), got "
                f"{tuple(base_features.shape)}"
            )
        feature_lengths = feature_lengths.to(
            device=base_features.device, dtype=torch.long
        ).clamp(min=1, max=base_features.size(1))
        should_adapt = (
            self.adapt_during_training if self.training else self.adapt_at_inference
        )
        adaptation_enabled = should_adapt and self.inner_steps > 0
        self._last_stats = {}

        if not adaptation_enabled:
            aligned = self.adapter(base_features)
            delta_rms, delta_relative = self._feature_delta_stats(
                base_features.detach(), aligned.detach(), feature_lengths
            )
            zero = base_features.new_zeros(())
            self._last_stats = {
                "latent_meta_support_pre": zero,
                "latent_meta_support_post": zero,
                "latent_meta_support_reduction": zero,
                "latent_meta_adapter_delta_norm": zero,
                "latent_meta_feature_delta_rms": delta_rms.detach(),
                "latent_meta_feature_delta_relative": delta_relative.detach(),
            }
            return aligned, feature_lengths

        # The SE frontend and ASR subsampler are frozen.  Detaching their
        # output avoids retaining an unnecessary graph, while the functional
        # fast weights below still retain the exact-MAML graph back to the
        # persistent adapter, predictor, and online target projector.
        original_dtype = base_features.dtype
        detached_base = base_features.detach().float()
        # Validation and Speech2Text are wrapped in torch.no_grad().  Re-enable
        # gradients locally for the support update, and keep higher-order
        # differentiation in fp32 rather than relying on AMP kernels.
        autocast_context = (
            torch.amp.autocast("cuda", enabled=False)
            if detached_base.is_cuda
            else nullcontext()
        )
        with torch.enable_grad(), autocast_context:
            if self.train_per_sample:
                aligned_batches = []
                episode_stats = []
                for batch_index in range(detached_base.size(0)):
                    aligned, stats = self._adapt_episode(
                        detached_base[batch_index : batch_index + 1],
                        feature_lengths[batch_index : batch_index + 1],
                        sample_offset=batch_index,
                    )
                    aligned_batches.append(aligned)
                    episode_stats.append(stats)
                aligned_features = torch.cat(aligned_batches, dim=0)
                self._last_stats = self._mean_stats(episode_stats)
            else:
                aligned_features, self._last_stats = self._adapt_episode(
                    detached_base,
                    feature_lengths,
                    sample_offset=0,
                )

        aligned_features = aligned_features.to(dtype=original_dtype)

        if not self.training:
            aligned_features = aligned_features.detach()
        return aligned_features, feature_lengths

    def _adapt_episode(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        sample_offset: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        support_mask = self._make_mask(
            base_features,
            feature_lengths,
            step=0,
            sample_offset=sample_offset,
        )
        fast_parameters = OrderedDict(self.adapter.named_parameters())
        support_pre = self._diagnostic_loss(
            base_features, feature_lengths, fast_parameters, support_mask
        )
        support_loss = base_features.new_zeros(())

        for step in range(self.inner_steps):
            if step > 0:
                support_mask = self._make_mask(
                    base_features,
                    feature_lengths,
                    step=step,
                    sample_offset=sample_offset,
                )
            support_loss = self._latent_prediction_loss(
                base_features,
                feature_lengths,
                fast_parameters,
                support_mask,
            )
            gradients = torch.autograd.grad(
                support_loss,
                tuple(fast_parameters.values()),
                create_graph=self.training and not self.first_order,
                retain_graph=self.training and not self.first_order,
            )
            gradients = self._clip_gradients(gradients)
            fast_parameters = OrderedDict(
                (
                    name,
                    parameter
                    - self.inner_lr
                    * (
                        gradient.detach()
                        if self.first_order or not self.training
                        else gradient
                    ),
                )
                for (name, parameter), gradient in zip(
                    fast_parameters.items(), gradients
                )
            )

        support_post = self._diagnostic_loss(
            base_features, feature_lengths, fast_parameters, support_mask
        )
        aligned_features = self.adapter(base_features, parameters=fast_parameters)
        delta_rms, delta_relative = self._feature_delta_stats(
            base_features, aligned_features, feature_lengths
        )
        valid_count = feature_lengths.sum().clamp_min(1).float()
        stats = {
            "latent_meta_support": support_loss.detach(),
            "latent_meta_support_pre": support_pre.detach(),
            "latent_meta_support_post": support_post.detach(),
            "latent_meta_support_reduction": (support_pre - support_post).detach(),
            "latent_meta_adapter_delta_norm": self._parameter_delta_norm(
                fast_parameters
            ).detach(),
            "latent_meta_feature_delta_rms": delta_rms.detach(),
            "latent_meta_feature_delta_relative": delta_relative.detach(),
            "latent_meta_mask_ratio": (
                support_mask.sum().float() / valid_count
            ).detach(),
        }
        return aligned_features, stats

    def _latent_prediction_loss(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        adapter_parameters: Mapping[str, torch.Tensor],
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self.adapter(base_features, parameters=adapter_parameters)
        masked_aligned = torch.where(
            support_mask.unsqueeze(-1),
            self.mask_token.expand_as(aligned),
            aligned,
        )
        online_latents = self.online_encoder(masked_aligned)
        frame_ids = torch.arange(
            base_features.size(1), device=base_features.device
        ).unsqueeze(0)
        valid_mask = frame_ids < feature_lengths.unsqueeze(1)
        # Exact MAML differentiates through the inner gradient.  CUDA flash
        # and memory-efficient SDPA kernels in PyTorch 2.4 do not implement
        # that double backward, so explicitly select the math kernel here.
        attention_context = (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
            if online_latents.is_cuda
            else nullcontext()
        )
        with attention_context:
            predicted_latents = self.predictor(
                online_latents,
                src_key_padding_mask=~valid_mask,
            )
        with torch.no_grad():
            target_latents = self.target_encoder(base_features)

        selected = support_mask & valid_mask
        if not selected.any():
            raise RuntimeError("Latent Meta-BRIDGE support mask is empty")
        predicted = predicted_latents[selected].float()
        target = target_latents[selected].float()
        if self.loss_type == "cosine":
            return (1.0 - F.cosine_similarity(predicted, target, dim=-1)).mean()
        return F.smooth_l1_loss(predicted, target)

    def _diagnostic_loss(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        adapter_parameters: Mapping[str, torch.Tensor],
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        training_states = (self.online_encoder.training, self.predictor.training)
        try:
            self.online_encoder.eval()
            self.predictor.eval()
            with torch.no_grad():
                return self._latent_prediction_loss(
                    base_features,
                    feature_lengths,
                    adapter_parameters,
                    support_mask,
                )
        finally:
            self.online_encoder.train(training_states[0])
            self.predictor.train(training_states[1])

    def _make_mask(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        step: int,
        sample_offset: int,
    ) -> torch.Tensor:
        mask = torch.zeros(
            features.shape[:2], dtype=torch.bool, device=features.device
        )
        for batch_index in range(features.size(0)):
            valid_length = int(feature_lengths[batch_index].item())
            target_count = max(1, int(valid_length * self.mask_ratio))
            generator = None
            if not self.training:
                generator = torch.Generator(device=features.device)
                generator.manual_seed(
                    self.inference_seed + step + sample_offset + batch_index
                )
            starts = torch.randperm(
                valid_length, device=features.device, generator=generator
            )
            selected = 0
            for start_tensor in starts:
                start = int(start_tensor.item())
                end = min(start + self.mask_span, valid_length)
                before = int(mask[batch_index, start:end].sum().item())
                mask[batch_index, start:end] = True
                selected += end - start - before
                if selected >= target_count:
                    break
            if selected > target_count:
                selected_indices = torch.nonzero(
                    mask[batch_index], as_tuple=False
                ).squeeze(1)
                mask[batch_index].zero_()
                mask[batch_index, selected_indices[:target_count]] = True
        return mask

    def _clip_gradients(
        self, gradients: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        if self.gradient_clip == 0.0:
            return gradients
        norm = torch.sqrt(
            sum(gradient.float().square().sum() for gradient in gradients)
            + 1.0e-12
        )
        scale = torch.clamp(self.gradient_clip / norm, max=1.0)
        return tuple(gradient * scale.to(gradient.dtype) for gradient in gradients)

    def _parameter_delta_norm(
        self, fast_parameters: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        base_parameters = dict(self.adapter.named_parameters())
        return torch.sqrt(
            sum(
                (fast_parameters[name] - base_parameters[name])
                .float()
                .square()
                .sum()
                for name in fast_parameters
            )
            + 1.0e-12
        )

    @staticmethod
    def _feature_delta_stats(
        base_features: torch.Tensor,
        aligned_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_ids = torch.arange(
            aligned_features.size(1), device=aligned_features.device
        ).unsqueeze(0)
        valid = (frame_ids < feature_lengths.unsqueeze(1)).unsqueeze(-1)
        valid = valid.to(aligned_features.dtype)
        count = valid.sum().clamp_min(1).float() * aligned_features.size(-1)
        delta = (aligned_features - base_features) * valid
        reference = base_features * valid
        delta_rms = torch.sqrt(delta.float().square().sum() / count)
        reference_rms = torch.sqrt(reference.float().square().sum() / count)
        return delta_rms, delta_rms / reference_rms.clamp_min(1.0e-12)

    @staticmethod
    def _mean_stats(
        episode_stats: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if not episode_stats:
            return {}
        return {
            key: torch.stack([stats[key] for stats in episode_stats]).mean()
            for key in episode_stats[0]
        }
