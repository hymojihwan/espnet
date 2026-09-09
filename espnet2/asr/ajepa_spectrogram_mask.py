"""A-JEPA-inspired structured masking for log-Mel spectrogram views."""

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn


class AJEPASpectrogramMask(nn.Module):
    """Build a structured masked view without consuming model dropout RNG.

    The mask is sampled on a non-overlapping time-frequency patch grid.  A
    square-root curriculum changes the batch-level sampling mode from random
    rectangular blocks to time-frequency-aware crosses.  The selected patch
    count is constrained to ``mask_ratio`` so experiments can compare the two
    geometries at the same corruption budget.

    This module adapts A-JEPA masking to a length-preserving ASR frontend.  It
    zeros selected, utterance-normalized Mel patches rather than gathering a
    sparse visible-token sequence; consequently it should be described as
    A-JEPA-inspired rather than as an exact reproduction of A-JEPA.
    """

    def __init__(
        self,
        input_dim: int = 80,
        patch_size: Sequence[int] = (16, 16),
        mask_ratio: float = 0.75,
        strategy: str = "curriculum",
        curriculum_steps: int = 11500,
        curriculum_c0: float = 0.01,
        random_target_blocks: int = 4,
        random_target_scale: Sequence[float] = (0.15, 0.20),
        random_target_aspect: Sequence[float] = (0.75, 1.50),
        tf_target_blocks: int = 3,
        tf_target_scale: Sequence[float] = (0.05, 0.075),
        mask_value: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        patch_size = self._pair(patch_size, "patch_size", integer=True)
        random_target_scale = self._pair(
            random_target_scale, "random_target_scale"
        )
        random_target_aspect = self._pair(
            random_target_aspect, "random_target_aspect"
        )
        tf_target_scale = self._pair(tf_target_scale, "tf_target_scale")

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if patch_size[0] <= 0 or patch_size[1] <= 0:
            raise ValueError("patch dimensions must be positive")
        if input_dim % patch_size[1] != 0:
            raise ValueError(
                "The Mel dimension must be divisible by the frequency patch "
                f"size: {input_dim} % {patch_size[1]} != 0"
            )
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if strategy not in ("curriculum", "random", "time_frequency"):
            raise ValueError(
                "strategy must be 'curriculum', 'random', or "
                "'time_frequency'"
            )
        if curriculum_steps <= 0:
            raise ValueError("curriculum_steps must be positive")
        if not 0.0 < curriculum_c0 <= 1.0:
            raise ValueError("curriculum_c0 must be in (0, 1]")
        if random_target_blocks <= 0 or tf_target_blocks <= 0:
            raise ValueError("target block counts must be positive")
        self._validate_range(random_target_scale, "random_target_scale")
        self._validate_range(tf_target_scale, "tf_target_scale")
        self._validate_range(
            random_target_aspect,
            "random_target_aspect",
            upper_bound=None,
        )

        self.input_dim = input_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.strategy = strategy
        self.curriculum_steps = curriculum_steps
        self.curriculum_c0 = curriculum_c0
        self.random_target_blocks = random_target_blocks
        self.random_target_scale = random_target_scale
        self.random_target_aspect = random_target_aspect
        self.tf_target_blocks = tf_target_blocks
        self.tf_target_scale = tf_target_scale
        self.mask_value = mask_value
        self.seed = seed

        self.register_buffer(
            "curriculum_step", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "mask_forward_count", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "tf_batch_count", torch.zeros((), dtype=torch.long)
        )
        self._last_stats: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _pair(
        value: Sequence[float],
        name: str,
        integer: bool = False,
    ) -> Tuple:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        if integer:
            return int(value[0]), int(value[1])
        return float(value[0]), float(value[1])

    @staticmethod
    def _validate_range(
        value: Tuple[float, float],
        name: str,
        upper_bound: float = 1.0,
    ) -> None:
        lower, upper = value
        if lower <= 0.0 or lower > upper:
            raise ValueError(f"Invalid {name}: {value}")
        if upper_bound is not None and upper > upper_bound:
            raise ValueError(f"Invalid {name}: {value}")

    def curriculum_probability(self) -> float:
        """Return the paper's square-root easy-to-hard probability."""
        progress = min(
            1.0,
            float(self.curriculum_step.item()) / self.curriculum_steps,
        )
        c0_squared = self.curriculum_c0 ** 2
        return min(
            1.0,
            math.sqrt(progress * (1.0 - c0_squared) + c0_squared),
        )

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a masked Mel view and its target mask on the patch grid."""
        self._validate_inputs(features, feature_lengths)
        patch_time, patch_frequency = self.patch_size
        num_time_patches = features.size(1) // patch_time
        num_frequency_patches = features.size(2) // patch_frequency
        if num_time_patches == 0:
            raise ValueError(
                "Every batch must contain at least one complete time patch"
            )

        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        forward_count = int(self.mask_forward_count.item())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.seed + 1000003 * forward_count + 9176 * rank
        )

        probability = self.curriculum_probability()
        if self.strategy == "random":
            use_time_frequency = False
        elif self.strategy == "time_frequency":
            use_time_frequency = True
        else:
            draw = float(torch.rand((), generator=generator).item())
            use_time_frequency = draw < probability

        lengths_cpu = feature_lengths.detach().to(
            device="cpu", dtype=torch.long
        )
        patch_mask_cpu = torch.zeros(
            features.size(0),
            num_time_patches,
            num_frequency_patches,
            dtype=torch.bool,
        )
        valid_patch_total = 0
        target_patch_total = 0
        skipped_short = 0
        for batch_index, length in enumerate(lengths_cpu.tolist()):
            valid_time_patches = min(
                int(length) // patch_time,
                num_time_patches,
            )
            if valid_time_patches == 0:
                skipped_short += 1
                continue
            num_valid = valid_time_patches * num_frequency_patches
            target_count = int(round(num_valid * self.mask_ratio))
            if num_valid > 1:
                target_count = min(max(1, target_count), num_valid - 1)
            else:
                target_count = 1
            patch_mask_cpu[batch_index, :valid_time_patches] = (
                self._sample_patch_mask(
                    valid_time_patches,
                    num_frequency_patches,
                    target_count,
                    use_time_frequency,
                    generator,
                )
            )
            valid_patch_total += num_valid
            target_patch_total += target_count

        if target_patch_total == 0:
            raise RuntimeError("No complete valid patch was available for SSL")

        patch_mask = patch_mask_cpu.to(device=features.device)
        complete_cell_mask = patch_mask.repeat_interleave(
            patch_time, dim=1
        ).repeat_interleave(patch_frequency, dim=2)
        complete_cell_mask = complete_cell_mask[
            :, : features.size(1), : features.size(2)
        ]
        cell_mask = torch.zeros_like(features, dtype=torch.bool)
        cell_mask[:, : complete_cell_mask.size(1)] = complete_cell_mask
        masked_features = features.masked_fill(cell_mask, self.mask_value)

        self.mask_forward_count.add_(1)
        if use_time_frequency:
            self.tf_batch_count.add_(1)
        scalar = features.new_tensor
        realized_ratio = target_patch_total / max(valid_patch_total, 1)
        self._last_stats = {
            "ajepa_target_patch_ratio": scalar(realized_ratio).detach(),
            "ajepa_curriculum_probability": scalar(probability).detach(),
            "ajepa_time_frequency_batch": scalar(
                float(use_time_frequency)
            ).detach(),
            "ajepa_time_frequency_fraction": scalar(
                float(self.tf_batch_count.item())
                / max(float(self.mask_forward_count.item()), 1.0)
            ).detach(),
            "ajepa_curriculum_step": scalar(
                float(self.curriculum_step.item())
            ).detach(),
            "ajepa_valid_patch_count": scalar(
                float(valid_patch_total)
            ).detach(),
            "ajepa_short_utterance_fraction": scalar(
                skipped_short / max(features.size(0), 1)
            ).detach(),
        }
        return masked_features, patch_mask

    def _sample_patch_mask(
        self,
        height: int,
        width: int,
        target_count: int,
        use_time_frequency: bool,
        generator: torch.Generator,
    ) -> torch.Tensor:
        # Tiny independent noise gives deterministic tie breaking and lets us
        # enforce exactly the requested budget after drawing structured blocks.
        priority = torch.rand((height, width), generator=generator) * 1.0e-4
        if use_time_frequency:
            blocks = self.tf_target_blocks
            scale_range = self.tf_target_scale
            aspect_range = (1.0, 1.0)
        else:
            blocks = self.random_target_blocks
            scale_range = self.random_target_scale
            aspect_range = self.random_target_aspect

        for _ in range(blocks):
            scale = self._uniform(scale_range, generator)
            aspect = self._uniform(aspect_range, generator)
            block_height = max(
                1,
                min(height, int(round(height * math.sqrt(scale * aspect)))),
            )
            block_width = max(
                1,
                min(width, int(round(width * math.sqrt(scale / aspect)))),
            )
            top = self._randint(height - block_height + 1, generator)
            left = self._randint(width - block_width + 1, generator)
            if use_time_frequency:
                priority[top : top + block_height, :] += 1.0
                priority[:, left : left + block_width] += 1.0
            else:
                priority[
                    top : top + block_height,
                    left : left + block_width,
                ] += 1.0

        selected = torch.topk(
            priority.flatten(),
            k=target_count,
            largest=True,
            sorted=False,
        ).indices
        mask = torch.zeros(height * width, dtype=torch.bool)
        mask[selected] = True
        return mask.view(height, width)

    @staticmethod
    def _uniform(
        value_range: Tuple[float, float],
        generator: torch.Generator,
    ) -> float:
        lower, upper = value_range
        if lower == upper:
            return lower
        draw = float(torch.rand((), generator=generator).item())
        return lower + draw * (upper - lower)

    @staticmethod
    def _randint(upper: int, generator: torch.Generator) -> int:
        if upper <= 1:
            return 0
        return int(torch.randint(upper, (), generator=generator).item())

    def project_to_latent(
        self,
        patch_mask: torch.Tensor,
        feature_time_steps: int,
        latent_time_steps: int,
        latent_lengths: torch.Tensor,
        subsampling_factor: int = 4,
        receptive_field_center: int = 3,
    ) -> torch.Tensor:
        """Map 2-D patch coverage to post-Conv2d time-token weights.

        ESPnet's two kernel-3/stride-2 Conv2d layers place latent token ``k``
        at input-frame center ``4*k + 3``.  Frequency coverage becomes a
        fractional loss weight because frequency has already been flattened
        into the 256-D latent at this interface.
        """
        if patch_mask.dim() != 3:
            raise ValueError("patch_mask must have shape (B, Nt, Nf)")
        if latent_time_steps <= 0:
            raise ValueError("latent_time_steps must be positive")
        patch_time = self.patch_size[0]
        time_weights = patch_mask.float().mean(dim=2)
        centers = (
            torch.arange(latent_time_steps, device=patch_mask.device)
            * subsampling_factor
            + receptive_field_center
        )
        patch_indices = torch.div(centers, patch_time, rounding_mode="floor")
        in_complete_grid = patch_indices < time_weights.size(1)
        safe_indices = patch_indices.clamp(max=time_weights.size(1) - 1)
        weights = time_weights[:, safe_indices]
        weights = weights * in_complete_grid.unsqueeze(0).to(weights.dtype)
        weights = weights * (centers < feature_time_steps).unsqueeze(0).to(
            weights.dtype
        )
        valid_latent = torch.arange(
            latent_time_steps, device=patch_mask.device
        ).unsqueeze(0) < latent_lengths.to(patch_mask.device).unsqueeze(1)
        return weights * valid_latent.to(weights.dtype)

    @torch.no_grad()
    def advance_step(self) -> None:
        """Advance the curriculum once per optimizer/EMA update."""
        self.curriculum_step.add_(1)

    def get_stats(self) -> Dict[str, torch.Tensor]:
        return self._last_stats.copy()

    def _validate_inputs(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> None:
        if features.dim() != 3 or features.size(2) != self.input_dim:
            raise ValueError(
                "Expected features shaped (B, T, input_dim), got "
                f"{tuple(features.shape)}"
            )
        if feature_lengths.dim() != 1 or feature_lengths.size(0) != features.size(0):
            raise ValueError("feature_lengths must have shape (batch,)")
        lengths = feature_lengths.to(dtype=torch.long)
        if (lengths <= 0).any() or (lengths > features.size(1)).any():
            raise ValueError("feature_lengths must be in [1, time_steps]")
