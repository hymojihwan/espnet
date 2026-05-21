"""SE-JEPA ASR Model: Conv-TasNet (SI-SNR) + JEPA predictive + Conformer CTC."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from espnet2.asr.espnet_jepa_model import ESPnetJEPAASRModel
from espnet2.asr.frontend.se_jepa import SE_JEPAFrontend
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetSEJEPAASRModel(ESPnetJEPAASRModel):
    """SE-JEPA ASR: SE(wav) → log-mel → JEPA predictive → Conformer → CTC.

    Loss = se_loss_weight * SI-SNR + jepa_weight * JEPA + ctc_weight * CTC 
           + encoder_distill_weight * MSE(enhanced_encoder_features, clean_encoder_features)
    
    The encoder distillation loss prevents SE from learning "hacks" that minimize
    log-mel losses but degrade intelligibility.
    """

    def __init__(self, *args, se_loss_weight: float = 0.5, encoder_distill_weight: float = 0.1, **kwargs):
        if "se_loss_weight" in kwargs:
            se_loss_weight = kwargs.pop("se_loss_weight")
        if "encoder_distill_weight" in kwargs:
            encoder_distill_weight = kwargs.pop("encoder_distill_weight")
        super().__init__(*args, **kwargs)
        self.se_loss_weight = se_loss_weight
        self.encoder_distill_weight = encoder_distill_weight

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
            loss_se = self.frontend.compute_se_loss()
            if loss_se is not None:
                loss = loss + self.se_loss_weight * loss_se
                stats["loss_se"] = loss_se.detach()
                stats["se_loss_weight"] = self.se_loss_weight
            
            # 2. ASR encoder feature distillation loss (clean encoder as target)
            # This prevents SE from learning tricks that minimize log-mel loss but degrade intelligibility
            if self.encoder_distill_weight > 0.0:
                loss_distill = self._compute_encoder_distill_loss(
                    speech, speech_lengths, clean_speech, clean_speech_lengths
                )
                if loss_distill is not None:
                    loss = loss + self.encoder_distill_weight * loss_distill
                    stats["loss_encoder_distill"] = loss_distill.detach()
                    stats["encoder_distill_weight"] = self.encoder_distill_weight

        if "loss" in stats:
            stats["loss"] = loss.detach()
        loss, stats, weight = force_gatherable((loss, stats, text.shape[0]), loss.device)
        return loss, stats, weight
    
    def _compute_encoder_distill_loss(
        self,
        noisy_speech: torch.Tensor,
        noisy_speech_lengths: torch.Tensor,
        clean_speech: torch.Tensor,
        clean_speech_lengths: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute encoder feature distillation loss.
        
        Enhanced encoder features should match clean encoder features to prevent
        SE from learning "hacks" that minimize log-mel loss but degrade intelligibility.
        
        Args:
            noisy_speech: (B, T) - Noisy speech input
            noisy_speech_lengths: (B,)
            clean_speech: (B, T) - Clean speech target
            clean_speech_lengths: (B,)
            
        Returns:
            loss_distill: MSE loss between enhanced and clean encoder features
        """
        # Enhanced path: noisy → SE → JEPA frontend → encoder
        # (already computed in forward, but we need to recompute for distillation)
        enhanced_feats, enhanced_feats_lengths = self._extract_feats_with_jepa(
            noisy_speech, noisy_speech_lengths, clean_speech, clean_speech_lengths
        )
        
        # Clean path: clean → JEPA frontend (bypass SE) → encoder
        # Use jepa_frontend directly to bypass SE module
        if not hasattr(self.frontend, 'jepa_frontend'):
            return None
        
        # Extract clean features using JEPA frontend directly (bypassing SE)
        clean_feats, clean_feats_lengths = self.frontend.jepa_frontend(
            clean_speech, clean_speech_lengths,
            clean_input=clean_speech,
            clean_input_lengths=clean_speech_lengths
        )
        
        # Normalize features
        if self.normalize is not None:
            enhanced_feats, enhanced_feats_lengths = self.normalize(enhanced_feats, enhanced_feats_lengths)
            clean_feats, clean_feats_lengths = self.normalize(clean_feats, clean_feats_lengths)
        
        # Pre-encoder
        if self.preencoder is not None:
            enhanced_feats, enhanced_feats_lengths = self.preencoder(enhanced_feats, enhanced_feats_lengths)
            clean_feats, clean_feats_lengths = self.preencoder(clean_feats, clean_feats_lengths)
        
        # Get encoder outputs
        # Clean encoder output (without gradients - it's the target)
        with torch.no_grad():
            if self.encoder.interctc_use_conditioning or getattr(self.encoder, "ctc_trim", False):
                clean_encoder_out, clean_encoder_out_lens, _ = self.encoder(
                    clean_feats, clean_feats_lengths, ctc=self.ctc
                )
            else:
                clean_encoder_out, clean_encoder_out_lens, _ = self.encoder(clean_feats, clean_feats_lengths)
            
            if isinstance(clean_encoder_out, tuple):
                clean_encoder_out = clean_encoder_out[0]
        
        # Enhanced encoder output (with gradients - this is what we optimize)
        if self.encoder.interctc_use_conditioning or getattr(self.encoder, "ctc_trim", False):
            enhanced_encoder_out, enhanced_encoder_out_lens, _ = self.encoder(
                enhanced_feats, enhanced_feats_lengths, ctc=self.ctc
            )
        else:
            enhanced_encoder_out, enhanced_encoder_out_lens, _ = self.encoder(enhanced_feats, enhanced_feats_lengths)
        
        if isinstance(enhanced_encoder_out, tuple):
            enhanced_encoder_out = enhanced_encoder_out[0]
        
        # Align lengths and compute MSE loss with stability checks
        batch_size = enhanced_encoder_out.size(0)
        loss_distill = 0.0
        valid_samples = 0
        
        for b in range(batch_size):
            enhanced_len = enhanced_encoder_out_lens[b].item()
            clean_len = clean_encoder_out_lens[b].item()
            min_len = min(enhanced_len, clean_len)
            
            if min_len > 0:
                enhanced_feat = enhanced_encoder_out[b, :min_len, :]  # (T, D)
                clean_feat = clean_encoder_out[b, :min_len, :]  # (T, D)
                
                # Check for NaN/Inf before computing loss
                if torch.isnan(enhanced_feat).any() or torch.isnan(clean_feat).any():
                    continue
                if torch.isinf(enhanced_feat).any() or torch.isinf(clean_feat).any():
                    continue
                
                # MSE loss: enhanced features should match clean features
                # Clamp values to prevent overflow
                enhanced_feat = torch.clamp(enhanced_feat, min=-10.0, max=10.0)
                clean_feat = torch.clamp(clean_feat, min=-10.0, max=10.0)
                
                mse_loss = F.mse_loss(enhanced_feat, clean_feat, reduction='mean')
                
                # Check if loss is valid
                if torch.isnan(mse_loss) or torch.isinf(mse_loss):
                    continue
                
                loss_distill += mse_loss
                valid_samples += 1
        
        if valid_samples > 0:
            final_loss = loss_distill / valid_samples
            # Final safety check
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                return None
            return final_loss
        else:
            return None
