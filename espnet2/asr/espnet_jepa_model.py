"""ESPnet JEPA ASR Model for training with JEPA frontend.

This model extends ESPnetASRModel to support JEPA frontend training
with clean speech input for computing target embeddings.
"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.jepa import JEPAFrontend
from espnet2.asr.frontend.jepa_residual import JEPAResidualFrontend
from espnet2.asr.frontend.jepa_masked import JEPA_MaskedPatchFrontend
from espnet2.asr.frontend.jepa_Vit import JEPA_MaskedPatchFrontend as JEPA_ViTFrontend
from espnet2.asr.frontend.jepa_hybrid import JEPA_HybridFrontend
from espnet2.asr.frontend.jepa_audio import JEPA_MaskedPatchLatentFrontend
from espnet2.asr.frontend.jepa_balanced import JEPA_BalancedFrontend
from espnet2.asr.frontend.jepa_mel_latent import JEPAMelLatentFrontend
from espnet2.asr.frontend.se_jepa import SE_JEPAFrontend
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.torch_utils.device_funcs import force_gatherable

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetJEPAASRModel(ESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model with JEPA frontend support.

    This model extends ESPnetASRModel to support JEPA frontend training
    by accepting clean_speech input for computing target embeddings.
    """

    def __init__(self, *args, **kwargs):
        """Initialize ESPnetJEPAASRModel.

        Args are the same as ESPnetASRModel, but frontend should be JEPAFrontend.
        """
        # Extract CTC weight schedule parameters from kwargs before passing to parent
        # These are passed via **args.model_conf in asr_jepa.py
        # Use get() and del to safely remove them
        ctc_weight_start = kwargs.get("ctc_weight_start", 0.01)
        ctc_weight_end = kwargs.get("ctc_weight_end", 1.0)
        ctc_weight_warmup_steps = kwargs.get("ctc_weight_warmup_steps", 10000)
        
        # Remove these keys from kwargs to prevent passing to parent
        if "ctc_weight_start" in kwargs:
            del kwargs["ctc_weight_start"]
        if "ctc_weight_end" in kwargs:
            del kwargs["ctc_weight_end"]
        if "ctc_weight_warmup_steps" in kwargs:
            del kwargs["ctc_weight_warmup_steps"]
        
        super().__init__(*args, **kwargs)
        
        # Verify that frontend is JEPA-style frontend
        if self.frontend is not None and not isinstance(self.frontend, (JEPAFrontend, JEPAResidualFrontend, JEPA_MaskedPatchFrontend, JEPA_ViTFrontend, JEPA_HybridFrontend, JEPA_MaskedPatchLatentFrontend, JEPA_BalancedFrontend, JEPAMelLatentFrontend, SE_JEPAFrontend)):
            logging.warning(
                f"Frontend is {type(self.frontend)}, not JEPAFrontend, JEPAResidualFrontend, JEPA_MaskedPatchFrontend, JEPA_ViTFrontend, JEPA_HybridFrontend, JEPA_MaskedPatchLatentFrontend, JEPA_BalancedFrontend, or JEPAMelLatentFrontend. "
                "JEPA-specific features may not work correctly."
            )
        
        # CTC weight schedule for progressive training
        # Start with small weight, gradually increase to allow CTC learning
        self.ctc_weight_start = ctc_weight_start
        self.ctc_weight_end = ctc_weight_end
        self.ctc_weight_warmup_steps = ctc_weight_warmup_steps
        # Step counter for CTC weight scheduling (will be updated during training)
        self.register_buffer("_ctc_weight_step", torch.tensor(0, dtype=torch.long))

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
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...) - Noisy speech input
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            clean_speech: Optional (Batch, Length, ...) - Clean speech for JEPA training
            clean_speech_lengths: Optional (Batch, ) - Clean speech lengths
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder with JEPA frontend support
        encoder_out, encoder_out_lens = self._encode_with_jepa(
            speech, speech_lengths, clean_speech, clean_speech_lengths
        )
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]
        self._last_online_encoder_out = encoder_out if self.training else None
        self._last_online_encoder_out_lens = (
            encoder_out_lens if self.training else None
        )

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc, wer_ctc = None, None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        loss_classif, acc_classif = None, None
        loss_jepa = None
        stats = dict()

        # 1. CTC branch
        # Always compute CTC loss to ensure CTC parameters are used in forward pass
        # (required for DDP), even if we don't use it in the final loss for JEPA-only training
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc, wer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc
            stats["wer_ctc"] = wer_ctc
        else:
            loss_ctc, cer_ctc, wer_ctc = None, None, None
            stats["loss_ctc"] = None
            stats["cer_ctc"] = None
            stats["wer_ctc"] = None

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out

                # use auxillary ctc data if specified
                loss_ic = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            loss_ic, cer_ic = self._calc_ctc_loss(
                                intermediate_out,
                                encoder_out_lens,
                                aux_data_tensor,
                                aux_data_lengths,
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        intermediate_out, encoder_out_lens, text, text_lengths
                    )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        elif self.use_linear_decoder:
            # 2b. Linear decoder branch for classification tasks
            loss, acc = self._calc_classif_loss(encoder_out, encoder_out_lens, text)
            stats["loss"] = loss
            stats["acc"] = acc
        else:
            # 2c. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # 4. JEPA loss (the configured target may not require clean speech)
        if (
            isinstance(self.frontend, (JEPAFrontend, JEPAResidualFrontend, JEPA_MaskedPatchFrontend, JEPA_ViTFrontend, JEPA_HybridFrontend, JEPA_MaskedPatchLatentFrontend, JEPA_BalancedFrontend, JEPAMelLatentFrontend, SE_JEPAFrontend))
            and self.training
        ):
            loss_jepa = self.frontend.compute_jepa_loss()
            if loss_jepa is not None and loss_jepa.item() > 0:
                # Get JEPA loss weight from frontend config
                jepa_weight = getattr(
                    self.frontend, "embedding_loss_weight", 0.5
                )
                # Store jepa_weight for monitoring
                stats["jepa_weight"] = jepa_weight
                # Add JEPA loss on top of the existing ASR objective.
                # This keeps RNNT/attention/CTC supervision intact.
                if loss is not None:
                    loss = loss + jepa_weight * loss_jepa
                elif loss_ctc is not None:
                    # Fallback (should rarely happen): JEPA + scheduled CTC
                    if hasattr(self, "_ctc_weight_step"):
                        step = self._ctc_weight_step.item()
                        if step < self.ctc_weight_warmup_steps:
                            progress = step / self.ctc_weight_warmup_steps
                            effective_ctc_weight = (
                                self.ctc_weight_start * (1 - progress)
                                + self.ctc_weight_end * progress
                            )
                        else:
                            effective_ctc_weight = self.ctc_weight_end
                        self._ctc_weight_step += 1
                    else:
                        effective_ctc_weight = self.ctc_weight_start
                    loss = jepa_weight * loss_jepa + effective_ctc_weight * loss_ctc
                    stats["effective_ctc_weight"] = effective_ctc_weight
                else:
                    loss = jepa_weight * loss_jepa
                stats["loss_jepa"] = loss_jepa.detach()
                if hasattr(self.frontend, "get_jepa_loss_stats"):
                    stats.update(self.frontend.get_jepa_loss_stats())
            else:
                # If JEPA loss is not available, fall back to CTC/attention loss
                if loss is None:
                    raise ValueError("Neither JEPA loss nor CTC/attention loss is available")
        elif loss is None:
            raise ValueError("No loss computed: JEPA loss not available and CTC/attention loss not computed")

        if self.training and hasattr(
            self.frontend, "compute_base_reconstruction_loss"
        ):
            loss_base_reconstruction = (
                self.frontend.compute_base_reconstruction_loss()
            )
            base_reconstruction_weight = getattr(
                self.frontend,
                "base_reconstruction_loss_weight",
                0.0,
            )
            if (
                loss_base_reconstruction is not None
                and base_reconstruction_weight > 0.0
            ):
                loss = loss + (
                    base_reconstruction_weight * loss_base_reconstruction
                )
                stats["base_reconstruction_weight"] = (
                    base_reconstruction_weight
                )
                stats["loss_base_reconstruction"] = (
                    loss_base_reconstruction.detach()
                )

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss with dtype-safe handling for AMP.

        warprnnt kernel may print "unsupported data type" when fed fp16/bf16
        activations. We keep decoder/joint in current precision, then cast only
        the joint output to fp32 right before RNNT loss.
        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)
        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        # RNNT loss backend expects float logits. Keep integer tensors explicit.
        loss_transducer = self.criterion_transducer(
            joint_out.float(),
            target.int(),
            t_len.int(),
            u_len.int(),
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

    def _encode_with_jepa(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        clean_speech: Optional[torch.Tensor] = None,
        clean_speech_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder with JEPA frontend support.

        Args:
            speech: (Batch, Length, ...) - Noisy speech
            speech_lengths: (Batch, )
            clean_speech: Optional (Batch, Length, ...) - Clean speech for JEPA
            clean_speech_lengths: Optional (Batch, ) - Clean speech lengths

        Returns:
            encoder_out: (Batch, Length2, Dim2)
            encoder_out_lens: (Batch,)
        """
        with autocast(False):
            # 1. Extract feats with JEPA frontend support
            feats, feats_lengths = self._extract_feats_with_jepa(
                speech, speech_lengths, clean_speech, clean_speech_lengths
            )

            # 2. Pre-encoder: transform JEPA frontend output to encoder input dimension
            #    Applied before normalization to match pretrained encoder's expected input
            if self.preencoder is not None:
                feats, feats_lengths = self.preencoder(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            # Skip normalization if frontend outputs embeddings (not mel features)
            # Embedding frontends (like jepa_hybrid) output embeddings directly
            if self.normalize is not None:
                # Check if frontend outputs embeddings (dim != 80 for mel)
                # jepa_hybrid outputs embedding_dim (256), not mel (80)
                if hasattr(self.frontend, 'output_dim') and self.frontend.output_dim != 80:
                    # Skip normalization for embedding outputs
                    pass
                else:
                    feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Defensive alignment: some frontends can return lengths that are
        # slightly inconsistent with the actual feature time axis when paired
        # noisy/clean streams are temporally mismatched (e.g., speed-perturbed
        # noisy with non-perturbed clean targets). Clamp lengths to prevent
        # downstream attention-mask shape mismatches.
        max_t = feats.size(1)
        feats_lengths = feats_lengths.clamp(min=1, max=max_t)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning or getattr(
            self.encoder, "ctc_trim", False
        ):
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _extract_feats_with_jepa(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        clean_speech: Optional[torch.Tensor] = None,
        clean_speech_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features with JEPA frontend support.

        Args:
            speech: (Batch, Length, ...) - Noisy speech
            speech_lengths: (Batch, )
            clean_speech: Optional (Batch, Length, ...) - Clean speech for JEPA
            clean_speech_lengths: Optional (Batch, ) - Clean speech lengths

        Returns:
            feats: (Batch, NFrames, Dim)
            feats_lengths: (Batch,)
        """
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            # For JEPA frontend, pass clean_speech if available
            if isinstance(self.frontend, (JEPAFrontend, JEPAResidualFrontend, JEPA_MaskedPatchFrontend, JEPA_ViTFrontend, JEPA_HybridFrontend, JEPA_MaskedPatchLatentFrontend, JEPA_BalancedFrontend, JEPAMelLatentFrontend, SE_JEPAFrontend)) and clean_speech is not None:
                feats, feats_lengths = self.frontend(
                    speech,
                    speech_lengths,
                    clean_input=clean_speech,
                    clean_input_lengths=clean_speech_lengths,
                )
            else:
                feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        # For inference, no clean speech is needed
        return self._encode_with_jepa(speech, speech_lengths, None, None)
