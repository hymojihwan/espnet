"""Meta-learned test-time adaptation frontend for BRIDGE ASR."""

from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, Mapping, Optional, Sequence, Tuple

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
    """Adapt on a support mask before constructing the ASR query features.

    ``meta_query_input_mode="adapter"`` preserves the original Meta-BRIDGE
    behaviour.  ``"bridge_hybrid"`` uses a disjoint query mask, reconstructs
    those patches with the post-update fast weights, and sends the resulting
    hybrid log-Mel representation to ASR in train, validation, and inference.
    """

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
        meta_train_per_sample: bool = False,
        meta_outer_support_loss: bool = True,
        meta_inner_eval_mode: bool = False,
        meta_freeze_backbone_eval: bool = False,
        meta_inference_seed: int = 0,
        meta_query_input_mode: str = "adapter",
        meta_query_mask_ratio: Optional[float] = None,
        meta_query_residual_weight: float = 1.0,
        meta_query_mask_seed: int = 1000,
        meta_disjoint_support_query: bool = True,
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
        if meta_query_input_mode not in ("adapter", "bridge_hybrid"):
            raise ValueError(
                "meta_query_input_mode must be 'adapter' or "
                f"'bridge_hybrid', got {meta_query_input_mode!r}"
            )
        if meta_query_mask_ratio is None:
            meta_query_mask_ratio = self.mask_ratio
        if not 0.0 <= meta_query_mask_ratio <= 1.0:
            raise ValueError("meta_query_mask_ratio must be in [0, 1]")
        if not 0.0 <= meta_query_residual_weight <= 1.0:
            raise ValueError(
                "meta_query_residual_weight must be in [0, 1]"
            )
        if (
            meta_disjoint_support_query
            and self.mask_ratio + meta_query_mask_ratio > 1.0
        ):
            raise ValueError(
                "Disjoint support/query masks require mask_ratio + "
                "meta_query_mask_ratio <= 1"
            )

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
        self.meta_train_per_sample = meta_train_per_sample
        self.meta_outer_support_loss = meta_outer_support_loss
        self.meta_inner_eval_mode = meta_inner_eval_mode
        self.meta_freeze_backbone_eval = meta_freeze_backbone_eval
        self.meta_inference_seed = meta_inference_seed
        self.meta_query_input_mode = meta_query_input_mode
        self.meta_query_mask_ratio = meta_query_mask_ratio
        self.meta_query_residual_weight = meta_query_residual_weight
        self.meta_query_mask_seed = meta_query_mask_seed
        self.meta_disjoint_support_query = meta_disjoint_support_query
        self._last_meta_loss: Optional[torch.Tensor] = None
        self._last_meta_support_mask: Optional[torch.Tensor] = None
        self._last_meta_query_mask: Optional[torch.Tensor] = None

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
        self._last_meta_support_mask = None
        self._last_meta_query_mask = None
        adaptation_enabled = should_adapt and self.meta_inner_steps > 0
        if (
            not adaptation_enabled
            and self.meta_query_input_mode == "adapter"
        ):
            return self.meta_adapter(asr_features), asr_feature_lengths

        detached_base = base_features.detach()
        with torch.enable_grad():
            if self.meta_train_per_sample:
                adapted_batches = []
                auxiliary_losses = []
                episode_stats = []
                for batch_index in range(detached_base.size(0)):
                    episode = (
                        self._adapt_episode
                        if adaptation_enabled
                        else self._query_episode_without_adaptation
                    )
                    adapted_batch, auxiliary_loss, stats = episode(
                        detached_base[batch_index : batch_index + 1],
                        feature_lengths[batch_index : batch_index + 1],
                        asr_features[batch_index : batch_index + 1],
                        asr_feature_lengths[
                            batch_index : batch_index + 1
                        ],
                    )
                    adapted_batches.append(adapted_batch)
                    episode_stats.append(stats)
                    if auxiliary_loss is not None:
                        auxiliary_losses.append(auxiliary_loss)
                adapted_features = torch.cat(adapted_batches, dim=0)
                if auxiliary_losses:
                    self._last_meta_loss = torch.stack(
                        auxiliary_losses
                    ).mean()
                self._last_jepa_loss_stats = self._mean_episode_stats(
                    episode_stats
                )
            else:
                episode = (
                    self._adapt_episode
                    if adaptation_enabled
                    else self._query_episode_without_adaptation
                )
                (
                    adapted_features,
                    self._last_meta_loss,
                    self._last_jepa_loss_stats,
                ) = episode(
                    detached_base,
                    feature_lengths,
                    asr_features,
                    asr_feature_lengths,
                )

            if not self.training:
                adapted_features = adapted_features.detach()

        return adapted_features, asr_feature_lengths

    def _query_episode_without_adaptation(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        asr_features: torch.Tensor,
        asr_feature_lengths: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """Build the same BRIDGE query as K1 while keeping K=0 weights."""
        base_parameters = OrderedDict(self.meta_adapter.named_parameters())
        support_mask = self._make_support_mask(
            base_features,
            feature_lengths,
            step=0,
        )
        query_mask = self._make_query_mask(
            base_features,
            feature_lengths,
            support_mask,
            step=0,
        )
        with self._bridge_query_context():
            query_features = self._build_bridge_query(
                asr_features,
                asr_feature_lengths,
                base_parameters,
                query_mask,
            )
        self._last_meta_support_mask = support_mask.detach()
        self._last_meta_query_mask = query_mask.detach()
        feature_delta_rms, feature_delta_relative = self._feature_delta_stats(
            asr_features.detach(),
            query_features.detach(),
            asr_feature_lengths,
        )
        stats = self._query_stats(
            base_features,
            feature_lengths,
            support_mask,
            query_mask,
        )
        stats.update(
            {
                "meta_adapter_delta_norm": base_features.new_zeros(()),
                "meta_feature_delta_rms": feature_delta_rms.detach(),
                "meta_feature_delta_relative": (
                    feature_delta_relative.detach()
                ),
            }
        )
        return query_features, None, stats

    def _adapt_episode(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        asr_features: torch.Tensor,
        asr_feature_lengths: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        support_mask = None
        query_mask = None
        if self.meta_query_input_mode == "bridge_hybrid":
            support_mask = self._make_support_mask(
                base_features,
                feature_lengths,
                step=0,
            )
            query_mask = self._make_query_mask(
                base_features,
                feature_lengths,
                support_mask,
                step=0,
            )
        inner_kwargs = {}
        if self.meta_query_input_mode == "bridge_hybrid":
            inner_kwargs["support_mask"] = support_mask
        (
            fast_parameters,
            support_mask,
            support_loss,
            support_loss_pre,
            support_loss_post,
        ) = self._inner_adapt(
            base_features,
            feature_lengths,
            **inner_kwargs,
        )
        self._last_meta_support_mask = support_mask.detach()
        if query_mask is None:
            if self.training:
                adapted_features = self.meta_adapter(
                    asr_features,
                    parameters=fast_parameters,
                )
            else:
                with torch.no_grad():
                    adapted_features = self.meta_adapter(
                        asr_features,
                        parameters=fast_parameters,
                    )
            with torch.no_grad():
                comparison_features = self.meta_adapter(
                    asr_features.detach()
                )
        else:
            with self._bridge_query_context():
                adapted_features = self._build_bridge_query(
                    asr_features,
                    asr_feature_lengths,
                    fast_parameters,
                    query_mask,
                )
            comparison_features = asr_features.detach()
            self._last_meta_support_mask = support_mask.detach()
            self._last_meta_query_mask = query_mask.detach()

        with torch.no_grad():
            feature_delta_rms, feature_delta_relative = (
                self._feature_delta_stats(
                    comparison_features,
                    adapted_features.detach(),
                    asr_feature_lengths,
                )
            )

        auxiliary_loss = None
        if self.training and self.meta_outer_support_loss:
            auxiliary_loss = self._masked_reconstruction_loss(
                base_features,
                feature_lengths,
                OrderedDict(self.meta_adapter.named_parameters()),
                support_mask,
            )

        stats = {
            # Keep the original key for log/config compatibility.
            "loss_meta_support": support_loss.detach(),
            "loss_meta_support_pre": support_loss_pre.detach(),
            "loss_meta_support_post": support_loss_post.detach(),
            "meta_support_loss_reduction": (
                support_loss_pre - support_loss_post
            ).detach(),
            "meta_adapter_delta_norm": self._parameter_delta_norm(
                fast_parameters
            ).detach(),
            "meta_feature_delta_rms": feature_delta_rms.detach(),
            "meta_feature_delta_relative": (
                feature_delta_relative.detach()
            ),
        }
        if query_mask is not None:
            stats.update(
                self._query_stats(
                    base_features,
                    feature_lengths,
                    support_mask,
                    query_mask,
                )
            )
        if auxiliary_loss is not None:
            stats["loss_jepa_mel_reconstruction"] = (
                auxiliary_loss.detach()
            )
        return adapted_features, auxiliary_loss, stats

    @staticmethod
    def _mean_episode_stats(
        episode_stats: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if not episode_stats:
            return {}
        return {
            stat_name: torch.stack(
                [stats[stat_name] for stats in episode_stats]
            ).mean()
            for stat_name in episode_stats[0]
        }

    def _inner_adapt(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        support_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        OrderedDict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        fast_parameters = OrderedDict(self.meta_adapter.named_parameters())
        fixed_support_mask = support_mask is not None
        if support_mask is None:
            support_mask = self._make_support_mask(
                base_features,
                feature_lengths,
                step=0,
            )
        diagnostic_mask = support_mask
        support_loss_pre = self._diagnostic_support_loss(
            base_features,
            feature_lengths,
            fast_parameters,
            diagnostic_mask,
        )
        support_loss = base_features.new_zeros(())
        with self._temporary_inner_eval(self.meta_inner_eval_mode):
            for inner_step in range(self.meta_inner_steps):
                if inner_step > 0 and not fixed_support_mask:
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
                    create_graph=(
                        self.training and not self.meta_first_order
                    ),
                    retain_graph=(
                        self.training and not self.meta_first_order
                    ),
                )
                gradients = self._clip_inner_gradients(gradients)
                fast_parameters = OrderedDict(
                    (
                        parameter_name,
                        parameter
                        - self.meta_inner_lr
                        * (
                            gradient.detach()
                            if self.meta_first_order or not self.training
                            else gradient
                        ),
                    )
                    for (parameter_name, parameter), gradient in zip(
                        fast_parameters.items(),
                        gradients,
                    )
                )
        support_loss_post = self._diagnostic_support_loss(
            base_features,
            feature_lengths,
            fast_parameters,
            diagnostic_mask,
        )
        return (
            fast_parameters,
            support_mask,
            support_loss,
            support_loss_pre,
            support_loss_post,
        )

    def _diagnostic_support_loss(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        adapter_parameters: Mapping[str, torch.Tensor],
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Eval mode makes pre/post values comparable and avoids consuming
        # dropout RNG, so enabling diagnostics does not perturb training.
        with self._temporary_inner_eval(True), torch.no_grad():
            return self._masked_reconstruction_loss(
                base_features,
                feature_lengths,
                adapter_parameters,
                support_mask,
            )

    @contextmanager
    def _temporary_inner_eval(self, enabled: bool):
        if not enabled:
            yield
            return

        modules = [
            self.context_patch_embed,
            self.context_pos_enc,
            self.context_encoder,
            self.predictor_pos_enc,
            self.predictor,
            self.decoder,
        ]
        modules = [module for module in modules if module is not None]
        training_states = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            yield
        finally:
            for module, was_training in zip(modules, training_states):
                module.train(was_training)

    def enforce_meta_loss_network_eval(self) -> None:
        """Keep the pretrained reconstruction network deterministic."""
        modules = [
            self.context_patch_embed,
            self.context_pos_enc,
            self.context_encoder,
            self.predictor_pos_enc,
            self.predictor,
            self.decoder,
        ]
        for module in modules:
            if module is not None:
                module.eval()

    @contextmanager
    def _bridge_query_context(self):
        """Match support/query module modes and avoid eval-only graphs."""
        with self._temporary_inner_eval(self.meta_inner_eval_mode):
            if self.training:
                yield
            else:
                with torch.no_grad():
                    yield

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

    def _build_bridge_query(
        self,
        asr_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        adapter_parameters: Mapping[str, torch.Tensor],
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct query patches and merge them into the ASR features.

        The adapter is used only to condition the BRIDGE prediction path.
        Unmasked patches always come from the original ASR-base features, so
        a global static-adapter shift cannot be mistaken for meta adaptation.
        """
        adapted_features = self.meta_adapter(
            asr_features,
            parameters=adapter_parameters,
        )
        adapted_patches, patch_mask, patch_grid = self._create_patches(
            adapted_features,
            feature_lengths,
        )
        base_patches, base_patch_mask, base_patch_grid = self._create_patches(
            asr_features,
            feature_lengths,
        )
        if (
            query_mask.shape != patch_mask.shape
            or not torch.equal(base_patch_mask, patch_mask)
            or base_patch_grid != patch_grid
        ):
            raise ValueError("Meta-BRIDGE query and ASR patch layouts differ")

        valid_query_mask = query_mask & patch_mask
        if not valid_query_mask.any():
            return asr_features.clone()

        context_features = self._encode_context_patches(
            adapted_patches,
            patch_mask,
            valid_query_mask,
        )
        predicted_latents = self._predict_latents(
            context_features,
            patch_mask,
        )
        reconstructed_patches = self.decoder(
            predicted_latents[valid_query_mask]
        ).to(base_patches.dtype)

        hybrid_patches = base_patches.clone()
        base_query_patches = hybrid_patches[valid_query_mask]
        hybrid_patches[valid_query_mask] = base_query_patches + (
            self.meta_query_residual_weight
            * (reconstructed_patches - base_query_patches)
        )
        return self._unpatch(
            hybrid_patches,
            (asr_features.size(1), asr_features.size(2)),
            patch_grid,
        )

    def _make_query_mask(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        support_mask: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Create a query mask independent of the inner update itself."""
        patches, patch_mask, _ = self._create_patches(
            base_features,
            feature_lengths,
        )
        if support_mask.shape != patch_mask.shape:
            raise ValueError("Support and query patch layouts differ")

        query_mask = torch.zeros_like(patch_mask)
        if self.meta_query_mask_ratio == 0.0:
            return query_mask

        generator = None
        if not self.training:
            generator = torch.Generator(device=base_features.device)
            generator.manual_seed(self.meta_query_mask_seed + step)

        for batch_index in range(patches.size(0)):
            valid_indices = torch.nonzero(
                patch_mask[batch_index],
                as_tuple=False,
            ).squeeze(1)
            if valid_indices.numel() == 0:
                continue
            candidate_indices = valid_indices
            if self.meta_disjoint_support_query:
                candidate_indices = valid_indices[
                    ~support_mask[batch_index, valid_indices]
                ]
            if candidate_indices.numel() == 0:
                continue

            num_masked = max(
                1,
                int(valid_indices.numel() * self.meta_query_mask_ratio),
            )
            num_masked = min(num_masked, candidate_indices.numel())
            permutation = torch.randperm(
                candidate_indices.numel(),
                device=base_features.device,
                generator=generator,
            )
            selected_indices = candidate_indices[permutation[:num_masked]]
            query_mask[batch_index, selected_indices] = True
        return query_mask

    def _query_stats(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        support_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        _, patch_mask, _ = self._create_patches(
            base_features,
            feature_lengths,
        )
        valid_count = patch_mask.sum().clamp_min(1).float()
        support_mask = support_mask & patch_mask
        query_mask = query_mask & patch_mask
        overlap = support_mask & query_mask
        return {
            "meta_support_mask_ratio": (
                support_mask.sum().float() / valid_count
            ).detach(),
            "meta_query_mask_ratio": (
                query_mask.sum().float() / valid_count
            ).detach(),
            "meta_support_query_overlap": (
                overlap.sum().float() / valid_count
            ).detach(),
        }

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

    @staticmethod
    def _feature_delta_stats(
        unadapted_features: torch.Tensor,
        adapted_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_indices = torch.arange(
            adapted_features.size(1),
            device=adapted_features.device,
        )
        valid_frames = frame_indices.unsqueeze(0) < feature_lengths.to(
            adapted_features.device
        ).unsqueeze(1)
        valid_elements = (
            valid_frames.sum().clamp_min(1).float()
            * adapted_features.size(-1)
        )
        valid_frames = valid_frames.unsqueeze(-1).to(adapted_features.dtype)
        delta = (adapted_features - unadapted_features) * valid_frames
        reference = unadapted_features * valid_frames
        delta_rms = torch.sqrt(delta.float().square().sum() / valid_elements)
        reference_rms = torch.sqrt(
            reference.float().square().sum() / valid_elements
        )
        relative_delta = delta_rms / reference_rms.clamp_min(1.0e-12)
        return delta_rms, relative_delta

    def compute_jepa_loss(self) -> Optional[torch.Tensor]:
        if self._last_meta_loss is None:
            return None
        return self.embedding_loss_weight * self._last_meta_loss

    def get_jepa_loss_stats(self) -> Dict[str, torch.Tensor]:
        return self._last_jepa_loss_stats.copy()
