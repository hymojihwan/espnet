"""Self-supervised BRIDGE operating on subsampled ASR latent sequences."""

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualTransformerPath(nn.Module):
    """Contextual transformer with an exactly identity-initialized output."""

    def __init__(
        self,
        feature_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")

        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Disabling nested tensors keeps train/eval behavior consistent across
        # CPU and CUDA and avoids implicit packing of variable-length batches.
        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(feature_dim),
            enable_nested_tensor=False,
        )
        self.input_norm = nn.LayerNorm(feature_dim)
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.feature_dim = feature_dim
        self.residual_scale = residual_scale

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return contextual features while preserving padded input tokens."""
        position = self._sinusoidal_positions(
            features.size(1),
            device=features.device,
            dtype=features.dtype,
        )
        hidden = self.input_norm(features) + position.unsqueeze(0)
        hidden = self.blocks(hidden, src_key_padding_mask=padding_mask)
        delta = self.output_projection(hidden)
        output = features + self.residual_scale * delta
        return torch.where(padding_mask.unsqueeze(-1), features, output)

    def _sinusoidal_positions(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build deterministic sinusoidal positions in the requested dtype."""
        positions = torch.arange(
            length,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        frequencies = torch.arange(
            0,
            self.feature_dim,
            2,
            device=device,
            dtype=torch.float32,
        )
        frequencies = torch.exp(
            frequencies * (-math.log(10000.0) / self.feature_dim)
        )
        encoding = torch.zeros(
            length,
            self.feature_dim,
            device=device,
            dtype=torch.float32,
        )
        angles = positions * frequencies.unsqueeze(0)
        encoding[:, 0::2] = torch.sin(angles)
        odd_width = encoding[:, 1::2].size(1)
        encoding[:, 1::2] = torch.cos(angles[:, :odd_width])
        return encoding.to(dtype=dtype)


class LatentBridge(nn.Module):
    """Predict masked enhanced-speech latents with an EMA teacher.

    During training, valid time steps are masked before an online contextual
    encoder.  A predictor estimates the unmasked representation produced by
    an exponential-moving-average copy of that encoder.  The sequence passed
    to ASR is a hybrid of predicted masked positions and online-context
    features at unmasked positions.  Evaluation uses no masking and returns
    the complete online context path.

    The module consumes and returns the same latent dimension (256 by
    default), uses no clean-speech target, and contains no waveform or Mel
    reconstruction decoder.
    """

    def __init__(
        self,
        input_dim: int = 256,
        context_layers: int = 1,
        context_heads: int = 4,
        context_ff_dim: int = 512,
        predictor_layers: int = 1,
        predictor_heads: int = 4,
        predictor_ff_dim: int = 512,
        dropout_rate: float = 0.1,
        context_residual_scale: float = 1.0,
        predictor_residual_scale: float = 1.0,
        mask_ratio: float = 0.15,
        mask_span: int = 3,
        ema_decay: float = 0.999,
        loss_type: str = "cosine",
        loss_weight: float = 0.08,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if context_layers <= 0 or predictor_layers <= 0:
            raise ValueError("context_layers and predictor_layers must be positive")
        if context_heads <= 0 or predictor_heads <= 0:
            raise ValueError("context_heads and predictor_heads must be positive")
        if context_ff_dim <= 0 or predictor_ff_dim <= 0:
            raise ValueError("transformer feed-forward dimensions must be positive")
        if not 0.0 < mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in (0, 1]")
        if mask_span <= 0:
            raise ValueError("mask_span must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if context_residual_scale < 0.0 or predictor_residual_scale < 0.0:
            raise ValueError("residual scales must be non-negative")
        if loss_type not in ("cosine", "smooth_l1"):
            raise ValueError("loss_type must be 'cosine' or 'smooth_l1'")
        if loss_weight < 0.0:
            raise ValueError("loss_weight must be non-negative")

        self.input_dim = input_dim
        self.mask_ratio = mask_ratio
        self.mask_span = mask_span
        self.ema_decay = ema_decay
        self.loss_type = loss_type
        self.loss_weight = loss_weight

        self.online_encoder = _ResidualTransformerPath(
            feature_dim=input_dim,
            num_layers=context_layers,
            num_heads=context_heads,
            ff_dim=context_ff_dim,
            dropout_rate=dropout_rate,
            residual_scale=context_residual_scale,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.target_encoder.eval()

        self.predictor = _ResidualTransformerPath(
            feature_dim=input_dim,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            ff_dim=predictor_ff_dim,
            dropout_rate=dropout_rate,
            residual_scale=predictor_residual_scale,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        self._last_loss: Optional[torch.Tensor] = None
        self._last_mask: Optional[torch.Tensor] = None
        self._last_stats: Dict[str, torch.Tensor] = {}

    def output_size(self) -> int:
        """Return the unchanged ASR latent dimension."""
        return self.input_dim

    def train(self, mode: bool = True):
        """Set module mode while keeping the EMA target deterministic."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform enhanced ASR latents and retain their time layout."""
        self._validate_inputs(base_features, feature_lengths)
        feature_lengths = feature_lengths.to(
            device=base_features.device,
            dtype=torch.long,
        )
        valid_mask = self._valid_mask(base_features, feature_lengths)
        padding_mask = ~valid_mask
        self._last_loss = None
        self._last_mask = None
        self._last_stats = {}

        if not self.training:
            context = self.online_encoder(base_features, padding_mask)
            context = torch.where(
                valid_mask.unsqueeze(-1), context, base_features
            )
            delta_rms, delta_relative = self._feature_delta_stats(
                base_features,
                context,
                valid_mask,
            )
            zero = base_features.new_zeros(())
            self._last_stats = {
                "loss_latent_bridge": zero,
                "latent_bridge_mask_ratio": zero,
                "latent_bridge_feature_delta_rms": delta_rms.detach(),
                "latent_bridge_feature_delta_relative": (
                    delta_relative.detach()
                ),
            }
            return context, feature_lengths

        mask = self._make_training_mask(valid_mask)
        mask_token = self.mask_token.to(dtype=base_features.dtype)
        masked_features = torch.where(
            mask.unsqueeze(-1),
            mask_token.expand_as(base_features),
            base_features,
        )
        context = self.online_encoder(masked_features, padding_mask)
        predicted = self.predictor(context, padding_mask)
        with torch.no_grad():
            target = self.target_encoder(base_features.detach(), padding_mask)

        selected_prediction = predicted[mask].float()
        selected_target = target[mask].float()
        if self.loss_type == "cosine":
            latent_loss = (
                1.0
                - F.cosine_similarity(
                    selected_prediction,
                    selected_target,
                    dim=-1,
                    eps=1.0e-8,
                )
            ).mean()
        else:
            latent_loss = F.smooth_l1_loss(
                selected_prediction,
                selected_target,
            )

        hybrid = torch.where(mask.unsqueeze(-1), predicted, context)
        hybrid = torch.where(valid_mask.unsqueeze(-1), hybrid, base_features)
        delta_rms, delta_relative = self._feature_delta_stats(
            base_features,
            hybrid,
            valid_mask,
        )
        valid_count = valid_mask.sum().clamp_min(1).float()
        self._last_loss = latent_loss
        self._last_mask = mask.detach()
        self._last_stats = {
            "loss_latent_bridge": latent_loss.detach(),
            "latent_bridge_mask_ratio": (
                mask.sum().float() / valid_count
            ).detach(),
            "latent_bridge_feature_delta_rms": delta_rms.detach(),
            "latent_bridge_feature_delta_relative": delta_relative.detach(),
        }
        return hybrid, feature_lengths

    def compute_loss(self) -> Optional[torch.Tensor]:
        """Return the current differentiable masked latent loss, if any."""
        return self._last_loss

    def get_stats(self) -> Dict[str, torch.Tensor]:
        """Return detached diagnostics from the most recent forward pass."""
        return self._last_stats.copy()

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Update the frozen target encoder from the online encoder by EMA."""
        online_parameters = dict(self.online_encoder.named_parameters())
        for name, target_parameter in self.target_encoder.named_parameters():
            target_parameter.mul_(self.ema_decay).add_(
                online_parameters[name],
                alpha=1.0 - self.ema_decay,
            )

        online_buffers = dict(self.online_encoder.named_buffers())
        for name, target_buffer in self.target_encoder.named_buffers():
            if name not in online_buffers:
                continue
            online_buffer = online_buffers[name]
            if torch.is_floating_point(target_buffer):
                target_buffer.mul_(self.ema_decay).add_(
                    online_buffer,
                    alpha=1.0 - self.ema_decay,
                )
            else:
                target_buffer.copy_(online_buffer)
        self.target_encoder.eval()

    def _validate_inputs(
        self,
        base_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> None:
        if (
            base_features.dim() != 3
            or base_features.size(-1) != self.input_dim
        ):
            raise ValueError(
                "Expected base_features shaped (B, T, input_dim), got "
                f"{tuple(base_features.shape)}"
            )
        if base_features.size(1) == 0:
            raise ValueError("base_features must contain at least one time step")
        if (
            feature_lengths.dim() != 1
            or feature_lengths.size(0) != base_features.size(0)
        ):
            raise ValueError("feature_lengths must have shape (batch,)")
        lengths = feature_lengths.to(dtype=torch.long)
        if (lengths <= 0).any() or (lengths > base_features.size(1)).any():
            raise ValueError("feature_lengths must be in [1, time_steps]")

    @staticmethod
    def _valid_mask(
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> torch.Tensor:
        frame_ids = torch.arange(
            features.size(1),
            device=features.device,
        ).unsqueeze(0)
        return frame_ids < feature_lengths.unsqueeze(1)

    def _make_training_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """Mask an exact fraction of valid tokens using short time spans."""
        mask = torch.zeros_like(valid_mask)
        for batch_index in range(valid_mask.size(0)):
            valid_length = int(valid_mask[batch_index].sum().item())
            target_count = max(1, int(valid_length * self.mask_ratio))
            starts = torch.randperm(valid_length, device=valid_mask.device)
            selected = 0
            for start_tensor in starts:
                start = int(start_tensor.item())
                end = min(start + self.mask_span, valid_length)
                previous = int(mask[batch_index, start:end].sum().item())
                mask[batch_index, start:end] = True
                selected += end - start - previous
                if selected >= target_count:
                    break

            if selected > target_count:
                selected_indices = torch.nonzero(
                    mask[batch_index],
                    as_tuple=False,
                ).squeeze(1)
                mask[batch_index].zero_()
                mask[batch_index, selected_indices[:target_count]] = True
        return mask & valid_mask

    @staticmethod
    def _feature_delta_stats(
        base_features: torch.Tensor,
        output_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = valid_mask.unsqueeze(-1).to(output_features.dtype)
        count = valid.sum().clamp_min(1).float() * output_features.size(-1)
        delta = (output_features - base_features) * valid
        reference = base_features * valid
        delta_rms = torch.sqrt(delta.float().square().sum() / count)
        reference_rms = torch.sqrt(
            reference.float().square().sum() / count
        )
        return delta_rms, delta_rms / reference_rms.clamp_min(1.0e-12)
