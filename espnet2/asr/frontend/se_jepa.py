"""SE-JEPA Frontend: Conv-TasNet (SI-SNR) -> log-mel -> JEPA predictive learning.

Flow: noisy_wav → SE(ConvTasNet) → enhanced_wav → log-mel → JEPA predictive (enhanced vs clean).
"""

from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.jepa_masked import JEPA_MaskedPatchFrontend
from espnet2.enh.separator.asteroid_models import AsteroidModel_Converter


class SE_JEPAFrontend(AbsFrontend):
    """SE + JEPA Frontend: Conv-TasNet for waveform enhancement, then JEPA predictive on log-mel.

    Architecture:
        Noisy wav → Conv-TasNet (SE) → Enhanced wav  [SI-SNR loss vs clean wav]
        Enhanced wav → log-mel
        Enhanced log-mel + Clean log-mel → JEPA predictive (same as jepa_masked)
        Output: denoised mel → ASR encoder
    """

    def __init__(
        self,
        jepa_frontend: Optional[JEPA_MaskedPatchFrontend] = None,
        jepa_frontend_conf: Optional[Dict[str, Any]] = None,
        se_model: Optional[nn.Module] = None,
        use_asteroid_convtasnet: bool = True,
        convtasnet_n_src: int = 1,
        convtasnet_n_filters: int = 512,
        convtasnet_kernel_size: int = 16,
        convtasnet_stride: int = 8,
        convtasnet_n_blocks: int = 8,
        convtasnet_n_repeats: int = 3,
        convtasnet_bn_chan: int = 128,
        convtasnet_hid_chan: int = 512,
        convtasnet_skip_chan: int = 128,
        convtasnet_conv_kernel_size: int = 3,
        se_train_config: Optional[str] = None,
        se_model_file: Optional[str] = None,
        se_no_grad: bool = True,
        observation_addition: bool = False,
        enhanced_weight: float = 0.8,
        noisy_weight: float = 0.2,
        rms_alignment_eps: float = 1.0e-8,
        **kwargs,
    ):
        super().__init__()
        if enhanced_weight < 0.0 or noisy_weight < 0.0:
            raise ValueError("Observation-addition weights must be non-negative")
        if observation_addition and enhanced_weight + noisy_weight <= 0.0:
            raise ValueError("At least one observation-addition weight must be positive")

        if jepa_frontend is not None:
            self.jepa_frontend = jepa_frontend
        elif jepa_frontend_conf is not None:
            self.jepa_frontend = JEPA_MaskedPatchFrontend(**jepa_frontend_conf)
        else:
            raise ValueError("Either jepa_frontend or jepa_frontend_conf must be provided")
        self.output_dim = self.jepa_frontend.output_dim

        self.se_no_grad = se_no_grad
        self.se_mode = "asteroid"
        self.observation_addition = observation_addition
        weight_sum = enhanced_weight + noisy_weight
        self.enhanced_weight = enhanced_weight / weight_sum
        self.noisy_weight = noisy_weight / weight_sum
        self.rms_alignment_eps = rms_alignment_eps

        if se_train_config is not None and se_model_file is not None:
            from espnet2.tasks.enh import EnhancementTask

            enh_model, _ = EnhancementTask.build_model_from_file(
                se_train_config, se_model_file, device="cpu"
            )
            self.se_model = (
                enh_model.enh_model if hasattr(enh_model, "enh_model") else enh_model
            )
            self.se_mode = "enh_core"
        elif use_asteroid_convtasnet and se_model is None:
            # Use AsteroidModel_Converter from enh/separator instead of direct asteroid import
            self.se_model = AsteroidModel_Converter(
                encoder_output_dim=1,
                model_name="ConvTasNet",
                num_spk=convtasnet_n_src,
                n_src=convtasnet_n_src,
                out_chan=None,
                n_blocks=convtasnet_n_blocks,
                n_repeats=convtasnet_n_repeats,
                bn_chan=convtasnet_bn_chan,
                hid_chan=convtasnet_hid_chan,
                skip_chan=convtasnet_skip_chan,
                conv_kernel_size=convtasnet_conv_kernel_size,
                norm_type="gLN",
                mask_act="sigmoid",
                in_chan=None,
                fb_name="free",
                kernel_size=convtasnet_kernel_size,
                n_filters=convtasnet_n_filters,
                stride=convtasnet_stride,
                encoder_activation=None,
            )
        elif se_model is not None:
            self.se_model = se_model
        else:
            raise ValueError("Either use_asteroid_convtasnet=True or se_model must be provided")

        if self.se_no_grad:
            self.se_model.eval()
            for p in self.se_model.parameters():
                p.requires_grad = False

        self._last_enhanced_wav = None
        self._last_clean_wav = None
        self._sisnr_loss = None  # Lazy init to avoid import at module load

    @property
    def embedding_loss_weight(self) -> float:
        return getattr(self.jepa_frontend, "embedding_loss_weight", 1.0)

    def output_size(self) -> int:
        return self.output_dim

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        clean_input: Optional[torch.Tensor] = None,
        clean_input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: SE(noisy) → enhanced_wav → JEPA(enhanced, clean) → mel feats.

        Args:
            input: (B, samples) - Noisy waveform
            input_lengths: (B,)
            clean_input: (B, samples) - Clean waveform for JEPA target & SI-SNR
            clean_input_lengths: (B,)

        Returns:
            feats: (B, T, D) - Mel features for ASR
            feats_lengths: (B,)
        """
        B = input.shape[0]
        device = input.device

        # Ensure (B, T) for SE
        if input.dim() == 1:
            input = input.unsqueeze(0)
        if input.dim() == 3:
            input = input.squeeze(1)

        # 1. SE: noisy_wav -> enhanced_wav
        ctx = torch.no_grad() if self.se_no_grad else nullcontext()
        with ctx:
            if self.se_mode == "enh_core":
                feature_mix, flens = self.se_model.encoder(input, input_lengths)
                feature_pre, _, _ = self.se_model.separator(feature_mix, flens, {})
                if not isinstance(feature_pre, (list, tuple)):
                    feature_pre = [feature_pre]
                enhanced_wav = self.se_model.decoder(feature_pre[0], input_lengths)[0]
            else:
                # AsteroidModel_Converter.forward returns (est_source_list, ilens, masks)
                est_source_list, _, _ = self.se_model.forward(input, input_lengths)
                enhanced_wav = est_source_list[0]

        if enhanced_wav.dim() == 3 and enhanced_wav.size(1) == 1:
            enhanced_wav = enhanced_wav.squeeze(1)
        elif enhanced_wav.dim() == 3:
            enhanced_wav = enhanced_wav[..., 0]

        # Match length to input for loss
        min_len = min(enhanced_wav.shape[-1], input.shape[-1])
        if clean_input is not None and clean_input.dim() >= 2:
            min_len = min(min_len, clean_input.shape[-1])
        enhanced_wav = enhanced_wav[..., :min_len]
        input_trim = input[..., :min_len]
        speech_lengths = input_lengths.to(device=enhanced_wav.device, dtype=torch.long)
        speech_lengths = torch.clamp(speech_lengths, max=min_len)

        if self.observation_addition:
            sample_ids = torch.arange(min_len, device=enhanced_wav.device)
            valid_mask = sample_ids.unsqueeze(0) < speech_lengths.unsqueeze(1)
            valid_mask = valid_mask.to(enhanced_wav.dtype)
            num_samples = speech_lengths.clamp_min(1).to(enhanced_wav.dtype).unsqueeze(1)
            enhanced_rms = torch.sqrt(
                (enhanced_wav.square() * valid_mask).sum(dim=1, keepdim=True)
                / num_samples
                + self.rms_alignment_eps
            )
            noisy_rms = torch.sqrt(
                (input_trim.square() * valid_mask).sum(dim=1, keepdim=True)
                / num_samples
                + self.rms_alignment_eps
            )
            enhanced_wav = enhanced_wav * (noisy_rms / enhanced_rms)
            enhanced_wav = (
                self.enhanced_weight * enhanced_wav + self.noisy_weight * input_trim
            ) * valid_mask

        if clean_input is not None and clean_input.dim() >= 2:
            clean_for_jepa = clean_input[..., :min_len]
            clean_lengths_for_jepa = speech_lengths
        else:
            clean_for_jepa = clean_input
            clean_lengths_for_jepa = clean_input_lengths

        if self.training:
            self._last_enhanced_wav = enhanced_wav
            self._last_clean_wav = clean_for_jepa

        # 2. JEPA: enhanced_wav as "noisy" input, clean_wav as target
        feats, feats_lengths = self.jepa_frontend(
            enhanced_wav,
            speech_lengths,
            clean_input=clean_for_jepa,
            clean_input_lengths=clean_lengths_for_jepa,
        )

        return feats, feats_lengths

    def compute_jepa_loss(self) -> Optional[torch.Tensor]:
        """Return the unweighted JEPA loss from the inner frontend."""
        loss = self.jepa_frontend.compute_jepa_loss()
        if loss is None:
            return None

        weight = self.embedding_loss_weight
        if weight == 0.0:
            return loss.detach() * 0.0
        return loss / weight

    def compute_se_loss(self) -> Optional[torch.Tensor]:
        """SI-SNR loss: enhanced_wav vs clean_wav (ref=clean, est=enhanced)."""
        if self._last_enhanced_wav is None or self._last_clean_wav is None:
            return None

        if self._sisnr_loss is None:
            from espnet2.enh.loss.criterions.time_domain import SISNRLoss

            self._sisnr_loss = SISNRLoss().to(self._last_enhanced_wav.device)

        ref = self._last_clean_wav
        est = self._last_enhanced_wav
        min_len = min(ref.shape[-1], est.shape[-1])
        ref, est = ref[..., :min_len], est[..., :min_len]
        loss = self._sisnr_loss(ref, est)
        return loss.mean()

    def update_target_encoder(self):
        """EMA update for JEPA target encoder."""
        if hasattr(self.jepa_frontend, "update_target_encoder"):
            self.jepa_frontend.update_target_encoder()
