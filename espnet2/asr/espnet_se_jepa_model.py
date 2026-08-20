"""SE-JEPA ASR Model: Conv-TasNet + JEPA + ASR distillation."""

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from espnet2.asr.espnet_jepa_model import ESPnetJEPAASRModel
from espnet2.asr.frontend.se_jepa import SE_JEPAFrontend
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetSEJEPAASRModel(ESPnetJEPAASRModel):
    """SE-JEPA ASR: SE(wav) → log-mel → JEPA predictive → Conformer → CTC.

    The optional ASR distillation term matches online encoder states to clean
    states produced by a train-only EMA copy of the ASR encoder.
    """

    def __init__(
        self,
        *args,
        se_loss_weight: float = 0.5,
        encoder_distill_weight: float = 0.1,
        encoder_distill_ema_decay: float = 0.999,
        encoder_distill_start_steps: int = 0,
        encoder_distill_ramp_steps: int = 0,
        **kwargs,
    ):
        if "se_loss_weight" in kwargs:
            se_loss_weight = kwargs.pop("se_loss_weight")
        if "encoder_distill_weight" in kwargs:
            encoder_distill_weight = kwargs.pop("encoder_distill_weight")
        super().__init__(*args, **kwargs)
        self.se_loss_weight = se_loss_weight
        self.encoder_distill_weight = encoder_distill_weight
        self.encoder_distill_ema_decay = encoder_distill_ema_decay
        self.encoder_distill_start_steps = encoder_distill_start_steps
        self.encoder_distill_ramp_steps = encoder_distill_ramp_steps
        self.register_buffer(
            "_encoder_distill_step",
            torch.tensor(0, dtype=torch.long),
        )

        if not 0.0 <= self.encoder_distill_ema_decay < 1.0:
            raise ValueError(
                "Expected 0.0 <= encoder_distill_ema_decay < 1.0, "
                f"got {self.encoder_distill_ema_decay}"
            )
        if self.encoder_distill_start_steps < 0:
            raise ValueError("encoder_distill_start_steps must be non-negative")
        if self.encoder_distill_ramp_steps < 0:
            raise ValueError("encoder_distill_ramp_steps must be non-negative")

        if self.encoder_distill_weight > 0.0:
            self.asr_teacher_encoder = copy.deepcopy(self.encoder)
            self.asr_teacher_encoder.eval()
            for parameter in self.asr_teacher_encoder.parameters():
                parameter.requires_grad = False

            output_size = self.encoder.output_size()
            self.encoder_distill_projection = nn.Sequential(
                nn.LayerNorm(output_size),
                nn.Linear(output_size, output_size),
                nn.GELU(),
                nn.Linear(output_size, output_size),
            )
        else:
            self.asr_teacher_encoder = None
            self.encoder_distill_projection = None

        self._last_online_encoder_out = None
        self._last_online_encoder_out_lens = None
        self._last_clean_teacher_out = None
        self._last_clean_teacher_lens = None

    def train(self, mode: bool = True):
        """Keep the EMA ASR teacher deterministic during student training."""
        super().train(mode)
        if self.asr_teacher_encoder is not None:
            self.asr_teacher_encoder.eval()
        return self

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        clean_speech: Optional[torch.Tensor] = None,
        clean_speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Forward with SE + JEPA + CTC loss."""
        effective_distill_weight = self._get_encoder_distill_weight()
        if (
            self.training
            and clean_speech is not None
            and self.encoder_distill_weight > 0.0
        ):
            if clean_speech_lengths is None:
                clean_speech_lengths = speech_lengths
            (
                self._last_clean_teacher_out,
                self._last_clean_teacher_lens,
            ) = self._encode_clean_teacher(
                clean_speech,
                clean_speech_lengths,
            )
        else:
            self._last_clean_teacher_out = None
            self._last_clean_teacher_lens = None

        loss, stats, weight = super().forward(
            speech, speech_lengths, text, text_lengths,
            clean_speech=clean_speech, clean_speech_lengths=clean_speech_lengths,
            **kwargs,
        )

        # Add SE (SI-SNR) loss when using SE_JEPAFrontend
        if (
            isinstance(self.frontend, SE_JEPAFrontend)
            and clean_speech is not None
            and self.training
        ):
            # 1. Waveform-domain SI-SNR loss (always present to prevent SE "hacks")
            if self.se_loss_weight > 0.0:
                loss_se = self.frontend.compute_se_loss()
                if loss_se is not None:
                    loss = loss + self.se_loss_weight * loss_se
                    stats["loss_se"] = loss_se.detach()
                    stats["se_loss_weight"] = self.se_loss_weight

            # 2. ASR encoder feature distillation loss (clean encoder as target)
            if self.encoder_distill_weight > 0.0:
                loss_distill = self._compute_encoder_distill_loss()
                if loss_distill is not None:
                    loss = loss + effective_distill_weight * loss_distill
                    stats["loss_encoder_distill"] = loss_distill.detach()
                    stats["encoder_distill_weight"] = effective_distill_weight

        self._last_online_encoder_out = None
        self._last_online_encoder_out_lens = None
        self._last_clean_teacher_out = None
        self._last_clean_teacher_lens = None

        if "loss" in stats:
            stats["loss"] = loss.detach()
        loss, stats, weight = force_gatherable((loss, stats, text.shape[0]), loss.device)
        return loss, stats, weight

    def _compute_encoder_distill_loss(
        self,
    ) -> Optional[torch.Tensor]:
        """Match online ASR states to clean EMA-teacher ASR states."""
        online_out = self._last_online_encoder_out
        online_lens = self._last_online_encoder_out_lens
        teacher_out = self._last_clean_teacher_out
        teacher_lens = self._last_clean_teacher_lens
        if (
            online_out is None
            or online_lens is None
            or teacher_out is None
            or teacher_lens is None
            or self.encoder_distill_projection is None
        ):
            return None

        max_length = min(online_out.size(1), teacher_out.size(1))
        if max_length == 0:
            return None
        valid_lens = torch.minimum(online_lens, teacher_lens).clamp(
            max=max_length
        )
        valid_mask = (
            torch.arange(max_length, device=online_out.device).unsqueeze(0)
            < valid_lens.unsqueeze(1)
        )
        if not valid_mask.any():
            return None

        projected_online = self.encoder_distill_projection(
            online_out[:, :max_length]
        )
        cosine = F.cosine_similarity(
            projected_online.float(),
            teacher_out[:, :max_length].float().detach(),
            dim=-1,
            eps=1.0e-8,
        )
        return (1.0 - cosine[valid_mask]).mean()

    @torch.no_grad()
    def _encode_clean_teacher(
        self,
        clean_speech: torch.Tensor,
        clean_speech_lengths: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode clean speech with the train-only EMA ASR teacher."""
        if (
            self.asr_teacher_encoder is None
            or not hasattr(self.frontend, "jepa_frontend")
        ):
            return None, None

        clean_speech = clean_speech[:, : clean_speech_lengths.max()]
        clean_feats, clean_feats_lengths = (
            self.frontend.jepa_frontend._extract_logmel(
                clean_speech,
                clean_speech_lengths,
            )
        )
        if self.preencoder is not None:
            clean_feats, clean_feats_lengths = self.preencoder(
                clean_feats,
                clean_feats_lengths,
            )
        if self.normalize is not None:
            clean_feats, clean_feats_lengths = self.normalize(
                clean_feats,
                clean_feats_lengths,
            )
        clean_feats_lengths = clean_feats_lengths.clamp(
            min=1,
            max=clean_feats.size(1),
        )
        if self.asr_teacher_encoder.interctc_use_conditioning or getattr(
            self.asr_teacher_encoder,
            "ctc_trim",
            False,
        ):
            teacher_out, teacher_lens, _ = self.asr_teacher_encoder(
                clean_feats,
                clean_feats_lengths,
                ctc=self.ctc,
            )
        else:
            teacher_out, teacher_lens, _ = self.asr_teacher_encoder(
                clean_feats,
                clean_feats_lengths,
            )
        if isinstance(teacher_out, tuple):
            teacher_out = teacher_out[0]
        return teacher_out, teacher_lens

    def _get_encoder_distill_weight(self) -> float:
        """Return the warmup-scaled clean distillation weight."""
        step = int(self._encoder_distill_step.item())
        if step < self.encoder_distill_start_steps:
            return 0.0
        if self.encoder_distill_ramp_steps == 0:
            return self.encoder_distill_weight
        progress = min(
            1.0,
            (step - self.encoder_distill_start_steps)
            / self.encoder_distill_ramp_steps,
        )
        return self.encoder_distill_weight * progress

    @torch.no_grad()
    def update_asr_teacher_encoder(self) -> None:
        """EMA-update the clean ASR teacher after each optimizer step."""
        if self.asr_teacher_encoder is None:
            return
        for teacher_parameter, student_parameter in zip(
            self.asr_teacher_encoder.parameters(),
            self.encoder.parameters(),
        ):
            teacher_parameter.data.mul_(self.encoder_distill_ema_decay).add_(
                student_parameter.data,
                alpha=1.0 - self.encoder_distill_ema_decay,
            )
        for teacher_buffer, student_buffer in zip(
            self.asr_teacher_encoder.buffers(),
            self.encoder.buffers(),
        ):
            if teacher_buffer.is_floating_point():
                teacher_buffer.mul_(self.encoder_distill_ema_decay).add_(
                    student_buffer,
                    alpha=1.0 - self.encoder_distill_ema_decay,
                )
            else:
                teacher_buffer.copy_(student_buffer)
        self._encoder_distill_step.add_(1)
