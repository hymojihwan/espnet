"""Meta-learned test-time adaptation frontend for BRIDGE ASR."""

from collections import OrderedDict
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.asr.frontend.jepa_masked import JEPA_MaskedPatchFrontend


class ResidualMetaAdapter(nn.Module):
    """Identity-initialized residual adapter for log-Mel features."""

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int = 16,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
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


class MetaBridgeFrontend(JEPA_MaskedPatchFrontend):
    """Run a self-supervised inner update before the unmasked ASR query."""

    def __init__(
        self,
        meta_adapter_dim: int = 16,
        meta_adapter_scale: float = 0.1,
        meta_inner_lr: float = 1.0e-3,
        meta_inner_steps: int = 1,
        meta_first_order: bool = True,
        meta_gradient_clip: float = 1.0,
        meta_adapt_during_training: bool = True,
        meta_adapt_at_inference: bool = True,
        meta_inference_seed: int = 0,
        **kwargs,
    ) -> None:
        required_options = {
            "asr_input_mode": "base",
            "enable_jepa_loss": True,
            "latent_loss_type": "mel_reconstruction",
            "mel_reconstruction_target": "base",
        }
        for option_name, expected_value in required_options.items():
            configured_value = kwargs.pop(option_name, expected_value)
            if configured_value != expected_value:
                raise ValueError(
                    f"MetaBridgeFrontend requires {option_name}="
                    f"{expected_value!r}, got {configured_value!r}"
                )
            kwargs[option_name] = expected_value

        super().__init__(**kwargs)
        if self.output_dim != self.n_mels:
            raise ValueError(
                "MetaBridgeFrontend requires output_dim == n_mels, got "
                f"{self.output_dim} and {self.n_mels}"
            )
        if meta_inner_lr <= 0.0:
            raise ValueError("meta_inner_lr must be positive")
        if meta_inner_steps < 0:
            raise ValueError("meta_inner_steps must be non-negative")
        if meta_gradient_clip < 0.0:
            raise ValueError("meta_gradient_clip must be non-negative")

        self.meta_adapter = ResidualMetaAdapter(
            feature_dim=self.n_mels,
            bottleneck_dim=meta_adapter_dim,
            residual_scale=meta_adapter_scale,
        )
        self.meta_inner_lr = meta_inner_lr
        self.meta_inner_steps = meta_inner_steps
        self.meta_first_order = meta_first_order
        self.meta_gradient_clip = meta_gradient_clip
        self.meta_adapt_during_training = meta_adapt_during_training
        self.meta_adapt_at_inference = meta_adapt_at_inference
        self.meta_inference_seed = meta_inference_seed
        self._last_meta_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        clean_input: Optional[torch.Tensor] = None,
        clean_input_lengths: Optional[torch.Tensor] = None,
        asr_input: Optional[torch.Tensor] = None,
        asr_input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del clean_input, clean_input_lengths
        base_features, feature_lengths = self._extract_logmel(input, input_lengths)
        if asr_input is None:
            asr_features = base_features
            asr_feature_lengths = feature_lengths
        else:
            selected_lengths = (
                asr_input_lengths
                if asr_input_lengths is not None
                else input_lengths
            )
            asr_features, asr_feature_lengths = self._extract_logmel(
                asr_input,
                selected_lengths,
            )
            if asr_features.shape != base_features.shape or not torch.equal(
                asr_feature_lengths,
                feature_lengths,
            ):
                raise ValueError(
                    "Meta support and ASR query features must have matching "
                    "shapes and lengths"
                )

        should_adapt = (
            self.meta_adapt_during_training
            if self.training
            else self.meta_adapt_at_inference
        )
        self._last_meta_loss = None
        self._last_jepa_loss_stats = {}
        if not should_adapt or self.meta_inner_steps == 0:
            return self.meta_adapter(asr_features), asr_feature_lengths

        detached_base = base_features.detach()
        with torch.enable_grad():
            fast_parameters, support_mask, support_loss = self._inner_adapt(
                detached_base,
                feature_lengths,
            )
            adapted_features = self.meta_adapter(
                asr_features,
                parameters=fast_parameters,
            )
            if self.training:
                self._last_meta_loss = self._masked_reconstruction_loss(
                    detached_base,
                    feature_lengths,
                    OrderedDict(self.meta_adapter.named_parameters()),
                    support_mask,
                )
                self._last_jepa_loss_stats = {
                    "loss_meta_support": support_loss.detach(),
                    "loss_jepa_mel_reconstruction": (
                        self._last_meta_loss.detach()
                    ),
                    "meta_adapter_delta_norm": self._parameter_delta_norm(
                        fast_parameters
                    ).detach(),
                }
            else:
                adapted_features = adapted_features.detach()

        return adapted_features, asr_feature_lengths

    def _inner_adapt(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> Tuple[
        OrderedDict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        fast_parameters = OrderedDict(self.meta_adapter.named_parameters())
        support_mask = self._make_support_mask(
            base_features,
            feature_lengths,
            step=0,
        )
        support_loss = base_features.new_zeros(())
        for inner_step in range(self.meta_inner_steps):
            if inner_step > 0:
                support_mask = self._make_support_mask(
                    base_features,
                    feature_lengths,
                    step=inner_step,
                )
            support_loss = self._masked_reconstruction_loss(
                base_features,
                feature_lengths,
                fast_parameters,
                support_mask,
            )
            gradients = torch.autograd.grad(
                support_loss,
                tuple(fast_parameters.values()),
                create_graph=not self.meta_first_order,
                retain_graph=not self.meta_first_order,
            )
            gradients = self._clip_inner_gradients(gradients)
            fast_parameters = OrderedDict(
                (
                    parameter_name,
                    parameter
                    - self.meta_inner_lr
                    * (
                        gradient.detach()
                        if self.meta_first_order
                        else gradient
                    ),
                )
                for (parameter_name, parameter), gradient in zip(
                    fast_parameters.items(),
                    gradients,
                )
            )
        return fast_parameters, support_mask, support_loss

    def _masked_reconstruction_loss(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        adapter_parameters: Mapping[str, torch.Tensor],
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        adapted_features = self.meta_adapter(
            base_features,
            parameters=adapter_parameters,
        )
        adapted_patches, patch_mask, _ = self._create_patches(
            adapted_features,
            feature_lengths,
        )
        target_patches, _, _ = self._create_patches(
            base_features.detach(),
            feature_lengths,
        )
        valid_support_mask = support_mask & patch_mask
        if not valid_support_mask.any():
            raise RuntimeError("Meta-BRIDGE support mask has no valid patches")

        context_features = self._encode_context_patches(
            adapted_patches,
            patch_mask,
            valid_support_mask,
        )
        predicted_latents = self._predict_latents(
            context_features,
            patch_mask,
        )
        predicted_masked = predicted_latents[valid_support_mask]
        reconstructed_patches = self.decoder(predicted_masked)
        target_masked = target_patches[valid_support_mask].detach()
        return F.mse_loss(
            reconstructed_patches.float(),
            target_masked.float(),
            reduction="mean",
        )

    def _make_support_mask(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        patches, patch_mask, _ = self._create_patches(
            base_features,
            feature_lengths,
        )
        support_mask = torch.zeros_like(patch_mask)
        generator = None
        if not self.training:
            generator = torch.Generator(device=base_features.device)
            generator.manual_seed(self.meta_inference_seed + step)

        for batch_index in range(patches.size(0)):
            valid_indices = torch.nonzero(
                patch_mask[batch_index],
                as_tuple=False,
            ).squeeze(1)
            if valid_indices.numel() == 0:
                continue
            num_masked = max(
                1,
                int(valid_indices.numel() * self.mask_ratio),
            )
            permutation = torch.randperm(
                valid_indices.numel(),
                device=base_features.device,
                generator=generator,
            )
            selected_indices = valid_indices[permutation[:num_masked]]
            support_mask[batch_index, selected_indices] = True
        return support_mask

    def _clip_inner_gradients(
        self,
        gradients: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        if self.meta_gradient_clip == 0.0:
            return gradients
        squared_norm = sum(
            gradient.float().square().sum() for gradient in gradients
        )
        total_norm = torch.sqrt(squared_norm + 1.0e-12)
        scale = torch.clamp(
            self.meta_gradient_clip / total_norm,
            max=1.0,
        )
        return tuple(gradient * scale.to(gradient.dtype) for gradient in gradients)

    def _parameter_delta_norm(
        self,
        fast_parameters: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        base_parameters = dict(self.meta_adapter.named_parameters())
        squared_norm = sum(
            (
                fast_parameters[parameter_name]
                - base_parameters[parameter_name]
            )
            .float()
            .square()
            .sum()
            for parameter_name in fast_parameters
        )
        return torch.sqrt(squared_norm + 1.0e-12)

    def compute_jepa_loss(self) -> Optional[torch.Tensor]:
        if self._last_meta_loss is None:
            return None
        return self.embedding_loss_weight * self._last_meta_loss

    def get_jepa_loss_stats(self) -> Dict[str, torch.Tensor]:
        return self._last_jepa_loss_stats.copy()
