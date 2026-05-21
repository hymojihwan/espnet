"""Dual-transducer ASR model with text-guided lower acoustic refinement."""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.encoder.encoder import Encoder
from espnet2.asr_transducer.espnet_transducer_model import ESPnetASRTransducerModel
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:

    @contextmanager
    def autocast(enabled=True):
        yield


class IdentityUpperEncoder(torch.nn.Module):
    """Identity upper encoder."""

    def __init__(self, input_size: int):
        super().__init__()
        self.output_size = input_size

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, x_lengths

    def forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearUpperEncoder(torch.nn.Module):
    """Per-frame upper encoder projection."""

    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        output_size = output_size or input_size
        hidden_size = hidden_size or output_size

        layers = []
        in_size = input_size
        for layer_idx in range(num_layers):
            out_size = output_size if layer_idx == num_layers - 1 else hidden_size
            layers.append(torch.nn.Linear(in_size, out_size))
            if layer_idx != num_layers - 1:
                layers.append(torch.nn.SiLU())
                if dropout_rate > 0.0:
                    layers.append(torch.nn.Dropout(dropout_rate))
            in_size = out_size

        self.network = torch.nn.Sequential(*layers)
        self.norm = torch.nn.LayerNorm(output_size) if use_layer_norm else None
        self.dropout = (
            torch.nn.Dropout(dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )
        self.output_size = output_size

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x)
        if self.norm is not None:
            y = self.norm(y)
        return self.dropout(y)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._forward_impl(x), x_lengths

    def forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class TransformerLowerPredictor(torch.nn.Module):
    """TTS-style transformer predictor for text-conditioned mel priors."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        linear_size: int = 1024,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        max_len: int = 1024,
        embed_pad: int = 0,
    ):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=embed_pad)
        self.pos_embed = torch.nn.Embedding(max_len, hidden_size)
        self.embed_dropout = torch.nn.Dropout(positional_dropout_rate)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=linear_size,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_size = hidden_size
        self.device = next(self.parameters()).device

    def set_device(self, device: torch.device) -> None:
        self.device = device

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = labels.shape
        positions = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(
            batch_size, seq_len
        )
        x = self.embed(labels) + self.pos_embed(positions)
        x = self.embed_dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=labels.device, dtype=torch.bool),
            diagonal=1,
        )
        lengths = labels.ne(0).sum(dim=1) + 1
        lengths = lengths.clamp_max(seq_len)
        key_padding_mask = (
            torch.arange(seq_len, device=labels.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
            >= lengths.unsqueeze(1)
        )
        return self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)


class AcousticOnlyEncoderWrapper(torch.nn.Module):
    """Inference-time encoder path without text conditioning."""

    def __init__(
        self,
        lower_transducer: torch.nn.Module,
        upper_encoder: torch.nn.Module,
    ):
        super().__init__()
        self.lower_transducer = lower_transducer
        self.upper_encoder = upper_encoder
        self.output_size = upper_encoder.output_size
        self.embed = getattr(lower_transducer.acoustic_encoder, "embed", None)

    @property
    def dynamic_chunk_training(self):
        return getattr(self.lower_transducer.acoustic_encoder, "dynamic_chunk_training", False)

    @dynamic_chunk_training.setter
    def dynamic_chunk_training(self, value):
        if hasattr(self.lower_transducer.acoustic_encoder, "dynamic_chunk_training"):
            self.lower_transducer.acoustic_encoder.dynamic_chunk_training = value

    def reset_cache(self, *args, **kwargs):
        if hasattr(self.lower_transducer.acoustic_encoder, "reset_cache"):
            return self.lower_transducer.acoustic_encoder.reset_cache(*args, **kwargs)
        return None

    def chunk_forward(self, *args, **kwargs):
        if self.lower_transducer.acoustic_mode != "encoder":
            raise NotImplementedError(
                "Streaming is not implemented for direct-mel lower acoustic mode."
            )
        if not hasattr(self.lower_transducer.acoustic_encoder, "chunk_forward"):
            raise NotImplementedError("Streaming is not supported by the wrapped encoder")
        if hasattr(self.upper_encoder, "chunk_forward"):
            raise NotImplementedError(
                "Streaming is not implemented yet for the dual-transducer "
                "configuration with an upper transducer encoder."
            )
        lower_out = self.lower_transducer.acoustic_encoder.chunk_forward(*args, **kwargs)
        denoised_mel = self.lower_transducer.acoustic_mel_head(lower_out)
        return self.upper_encoder.forward_chunk(denoised_mel)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        denoised_mel, lower_out_lens = self.lower_transducer.acoustic_only_mel(
            x, x_lengths
        )
        return self.upper_encoder(denoised_mel, lower_out_lens)


class LowerDualTransducer(torch.nn.Module):
    """Lower dual-transducer with text-guided acoustic refinement."""

    def __init__(
        self,
        acoustic_encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        prior_joint: JointNetwork,
        mel_dim: int,
        aggregator_type: str = "softmax",
        prior_dropout_rate: float = 0.1,
        gate_hidden_size: int = 256,
        gate_dropout_rate: float = 0.0,
        gate_bias_init: float = 0.0,
        text_decoder_num_layers: int = 2,
        text_decoder_num_heads: int = 4,
        text_decoder_linear_size: int = 1024,
        text_decoder_dropout_rate: float = 0.1,
        text_decoder_max_len: int = 4096,
        fusion_hidden_size: int = 256,
        fusion_dropout_rate: float = 0.1,
        acoustic_mode: str = "encoder",
        guidance_scale: float = 1.0,
        zero_init_guidance: bool = False,
    ):
        super().__init__()
        self.acoustic_encoder = acoustic_encoder
        self.predictor = predictor
        self.prior_joint = prior_joint
        self.acoustic_mode = acoustic_mode
        if acoustic_mode == "encoder":
            acoustic_hidden_size = acoustic_encoder.output_size
            self.acoustic_input_proj = None
        elif acoustic_mode == "direct_mel":
            acoustic_hidden_size = acoustic_encoder.output_size
            self.acoustic_input_proj = torch.nn.Sequential(
                torch.nn.Linear(mel_dim, acoustic_hidden_size),
                torch.nn.LayerNorm(acoustic_hidden_size),
                torch.nn.SiLU(),
            )
            for parameter in self.acoustic_encoder.parameters():
                parameter.requires_grad_(False)
        else:
            raise ValueError(f"Unsupported acoustic_mode: {acoustic_mode}")

        self.acoustic_hidden_size = acoustic_hidden_size
        self.guidance_scale = guidance_scale
        self.acoustic_mel_head = torch.nn.Linear(acoustic_hidden_size, mel_dim)
        self.prior_dropout = torch.nn.Dropout(prior_dropout_rate)
        self.guidance_score = torch.nn.Linear(acoustic_hidden_size, 1)
        self.fusion_network = torch.nn.Sequential(
            torch.nn.Linear(acoustic_hidden_size * 2, fusion_hidden_size),
            torch.nn.SiLU(),
            torch.nn.Dropout(fusion_dropout_rate),
            torch.nn.Linear(fusion_hidden_size, acoustic_hidden_size),
        )
        self.gate_network = torch.nn.Sequential(
            torch.nn.Linear(acoustic_hidden_size * 2, gate_hidden_size),
            torch.nn.SiLU(),
            torch.nn.Dropout(gate_dropout_rate),
            torch.nn.Linear(gate_hidden_size, 1),
        )
        if gate_bias_init != 0.0:
            torch.nn.init.constant_(self.gate_network[-1].bias, gate_bias_init)
        self.text_mel_head = torch.nn.Linear(acoustic_hidden_size, mel_dim)
        self.fused_mel_head = torch.nn.Linear(acoustic_hidden_size, mel_dim)
        if zero_init_guidance:
            torch.nn.init.zeros_(self.fusion_network[-1].weight)
            torch.nn.init.zeros_(self.fusion_network[-1].bias)
            torch.nn.init.zeros_(self.fused_mel_head.weight)
            torch.nn.init.zeros_(self.fused_mel_head.bias)

        self.output_size = mel_dim

    def set_device(self, device: torch.device) -> None:
        if hasattr(self.predictor, "set_device"):
            self.predictor.set_device(device)

    def _aggregate_prior(
        self,
        predictor_hidden: torch.Tensor,
        predictor_lengths: torch.Tensor,
        acoustic_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pairwise_prior = self.prior_joint(
            acoustic_hidden.unsqueeze(2),
            predictor_hidden.unsqueeze(1),
        )
        pairwise_prior = self.prior_dropout(pairwise_prior)
        gate_logits = self.guidance_score(pairwise_prior)

        max_u = predictor_hidden.size(1)
        predictor_mask = (
            torch.arange(max_u, device=predictor_hidden.device)
            .unsqueeze(0)
            .expand(predictor_hidden.size(0), max_u)
            >= predictor_lengths.unsqueeze(1)
        ).unsqueeze(1).unsqueeze(-1)
        gate_logits = gate_logits.masked_fill(predictor_mask, float("-inf"))
        gate_alpha = torch.softmax(gate_logits, dim=2)
        prior = torch.sum(gate_alpha * pairwise_prior, dim=2)

        return prior, gate_alpha.squeeze(-1)

    def acoustic_forward(
        self,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.acoustic_mode == "encoder":
            acoustic_hidden, acoustic_lengths = self.acoustic_encoder(feats, feats_lengths)
            acoustic_mel = self.acoustic_mel_head(acoustic_hidden)
        else:
            acoustic_hidden = self.acoustic_input_proj(feats)
            acoustic_lengths = feats_lengths
            acoustic_mel = self.acoustic_mel_head(acoustic_hidden)
            if feats.size(-1) == self.output_size:
                # Keep the enhanced frontend mel as the acoustic anchor while
                # always using a learnable residual so DDP sees all params.
                acoustic_mel = feats + acoustic_mel
        return acoustic_hidden, acoustic_lengths, acoustic_mel

    def acoustic_only_mel(
        self,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, acoustic_lengths, acoustic_mel = self.acoustic_forward(feats, feats_lengths)
        return acoustic_mel, acoustic_lengths

    def forward(
        self,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        decoder_in: torch.Tensor,
        predictor_lengths: torch.Tensor,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        acoustic_hidden, acoustic_lengths, acoustic_mel = self.acoustic_forward(
            feats, feats_lengths
        )

        self.set_device(acoustic_hidden.device)
        predictor_hidden = self.predictor(decoder_in)
        prior, prior_alpha = self._aggregate_prior(
            predictor_hidden, predictor_lengths, acoustic_hidden
        )
        fusion_input = torch.cat([acoustic_hidden, prior], dim=-1)
        delta_hidden = torch.tanh(self.fusion_network(fusion_input))
        gate_logits = self.gate_network(fusion_input)
        gate_alpha = torch.sigmoid(gate_logits)
        guided_alpha = self.guidance_scale * gate_alpha
        fused_hidden = acoustic_hidden + guided_alpha * delta_hidden
        text_mel = self.text_mel_head(prior)
        fused_mel = acoustic_mel + self.fused_mel_head(fused_hidden)

        return {
            "acoustic_hidden": acoustic_hidden,
            "acoustic_lengths": acoustic_lengths,
            "predictor_hidden": predictor_hidden,
            "prior": prior,
            "gate_alpha": gate_alpha,
            "guided_alpha": guided_alpha,
            "prior_alpha": prior_alpha,
            "gate_logits": gate_logits,
            "fused_hidden": fused_hidden,
            "acoustic_mel": acoustic_mel,
            "text_mel_tokens": None,
            "text_mel": text_mel,
            "fused_mel": fused_mel,
        }


class ESPnetASRDualTransducerModel(ESPnetASRTransducerModel):
    """Dual-transducer model with lower text-guided acoustic refinement."""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: torch.nn.Module,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        lower_decoder: torch.nn.Module,
        lower_joint_network: JointNetwork,
        transducer_weight: float = 1.0,
        use_k2_pruned_loss: bool = False,
        k2_pruned_loss_args: Dict = {},
        warmup_steps: int = 25000,
        validation_nstep: int = 2,
        fastemit_lambda: float = 0.0,
        auxiliary_ctc_weight: float = 0.0,
        auxiliary_ctc_dropout_rate: float = 0.0,
        auxiliary_lm_loss_weight: float = 0.0,
        auxiliary_lm_loss_smoothing: float = 0.05,
        ignore_id: int = -1,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        report_cer: bool = False,
        report_wer: bool = False,
        extract_feats_in_collect_stats: bool = True,
        upper_encoder_type: str = "transducer",
        upper_encoder_conf: Optional[Dict] = None,
        lower_aggregator_type: str = "softmax",
        lower_prior_dropout_rate: float = 0.1,
        lower_gate_hidden_size: int = 256,
        lower_gate_dropout_rate: float = 0.0,
        lower_gate_bias_init: float = 0.0,
        lower_text_decoder_num_layers: int = 2,
        lower_text_decoder_num_heads: int = 4,
        lower_text_decoder_linear_size: int = 1024,
        lower_text_decoder_dropout_rate: float = 0.1,
        lower_text_decoder_max_len: int = 4096,
        lower_fusion_hidden_size: int = 256,
        lower_fusion_dropout_rate: float = 0.1,
        lower_acoustic_mode: str = "encoder",
        lower_guidance_scale: float = 1.0,
        lower_zero_init_guidance: bool = False,
        lower_mel_dim: int = 80,
        lower_mel_loss_weight: float = 0.0,
        lower_mel_loss_type: str = "l1",
        lower_mel_acoustic_weight: float = 1.0,
        lower_mel_text_weight: float = 1.0,
        lower_mel_fused_weight: float = 1.0,
        lower_gate_reg_weight: float = 0.0,
        training_stage: str = "joint",
        freeze_lower: bool = False,
        freeze_upper: bool = False,
    ) -> None:
        upper_encoder_conf = upper_encoder_conf or {}
        lower_encoder = encoder
        self.lower_mel_dim = lower_mel_dim
        self.lower_mel_loss_weight = lower_mel_loss_weight
        self.lower_mel_loss_type = lower_mel_loss_type
        self.lower_mel_acoustic_weight = lower_mel_acoustic_weight
        self.lower_mel_text_weight = lower_mel_text_weight
        self.lower_mel_fused_weight = lower_mel_fused_weight
        self.lower_gate_reg_weight = lower_gate_reg_weight
        if training_stage not in ("joint", "lower", "upper"):
            raise ValueError(f"Unsupported training_stage: {training_stage}")
        self.training_stage = training_stage
        self.freeze_lower = freeze_lower
        self.freeze_upper = freeze_upper

        if upper_encoder_type == "identity":
            upper_encoder = IdentityUpperEncoder(lower_encoder.output_size)
        elif upper_encoder_type == "linear":
            upper_encoder = LinearUpperEncoder(lower_encoder.output_size, **upper_encoder_conf)
        elif upper_encoder_type == "transducer":
            upper_encoder = Encoder(lower_mel_dim, **upper_encoder_conf)
        else:
            raise ValueError(f"Unsupported upper_encoder_type: {upper_encoder_type}")

        lower_transducer = LowerDualTransducer(
            acoustic_encoder=lower_encoder,
            predictor=lower_decoder,
            prior_joint=lower_joint_network,
            mel_dim=lower_mel_dim,
            aggregator_type=lower_aggregator_type,
            prior_dropout_rate=lower_prior_dropout_rate,
            gate_hidden_size=lower_gate_hidden_size,
            gate_dropout_rate=lower_gate_dropout_rate,
            gate_bias_init=lower_gate_bias_init,
            text_decoder_num_layers=lower_text_decoder_num_layers,
            text_decoder_num_heads=lower_text_decoder_num_heads,
            text_decoder_linear_size=lower_text_decoder_linear_size,
            text_decoder_dropout_rate=lower_text_decoder_dropout_rate,
            text_decoder_max_len=lower_text_decoder_max_len,
            fusion_hidden_size=lower_fusion_hidden_size,
            fusion_dropout_rate=lower_fusion_dropout_rate,
            acoustic_mode=lower_acoustic_mode,
            guidance_scale=lower_guidance_scale,
            zero_init_guidance=lower_zero_init_guidance,
        )
        inference_encoder = AcousticOnlyEncoderWrapper(
            lower_transducer,
            upper_encoder,
        )

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=lower_encoder,
            decoder=decoder,
            joint_network=joint_network,
            transducer_weight=transducer_weight,
            use_k2_pruned_loss=use_k2_pruned_loss,
            k2_pruned_loss_args=k2_pruned_loss_args,
            warmup_steps=warmup_steps,
            validation_nstep=validation_nstep,
            fastemit_lambda=fastemit_lambda,
            auxiliary_ctc_weight=auxiliary_ctc_weight,
            auxiliary_ctc_dropout_rate=auxiliary_ctc_dropout_rate,
            auxiliary_lm_loss_weight=auxiliary_lm_loss_weight,
            auxiliary_lm_loss_smoothing=auxiliary_lm_loss_smoothing,
            ignore_id=ignore_id,
            sym_space=sym_space,
            sym_blank=sym_blank,
            report_cer=report_cer,
            report_wer=report_wer,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
        )

        self.lower_transducer = lower_transducer
        self.encoder = inference_encoder
        if self.use_k2_pruned_loss and self.am_proj.in_features != self.upper_encoder.output_size:
            self.am_proj = torch.nn.Linear(self.upper_encoder.output_size, vocab_size)

        if self.use_auxiliary_ctc and self.ctc_lin.in_features != self.upper_encoder.output_size:
            self.ctc_lin = torch.nn.Linear(self.upper_encoder.output_size, vocab_size)

        if self.freeze_lower:
            self._set_module_requires_grad(self.lower_transducer, False)
        if self.freeze_upper:
            self._set_module_requires_grad(self.upper_encoder, False)
            self._set_module_requires_grad(self.decoder, False)
            self._set_module_requires_grad(self.joint_network, False)
            if hasattr(self, "am_proj") and self.am_proj is not None:
                self._set_module_requires_grad(self.am_proj, False)
            if self.use_auxiliary_ctc and hasattr(self, "ctc_lin") and self.ctc_lin is not None:
                self._set_module_requires_grad(self.ctc_lin, False)

    @property
    def lower_encoder(self):
        return self.lower_transducer.acoustic_encoder

    @property
    def upper_encoder(self):
        return self.encoder.upper_encoder

    @staticmethod
    def _set_module_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)

    def _calc_lower_mel_loss(
        self,
        denoised_mel: torch.Tensor,
        denoised_mel_lens: torch.Tensor,
        clean_feats: torch.Tensor,
        clean_feats_lens: torch.Tensor,
    ) -> torch.Tensor:
        max_len = min(denoised_mel.size(1), clean_feats.size(1))
        feat_dim = min(denoised_mel.size(2), clean_feats.size(2))

        denoised_mel = denoised_mel[:, :max_len, :feat_dim]
        clean_feats = clean_feats[:, :max_len, :feat_dim]

        valid_lens = torch.minimum(denoised_mel_lens, clean_feats_lens).clamp_max(max_len)
        mask = (
            torch.arange(max_len, device=denoised_mel.device)
            .unsqueeze(0)
            .expand(denoised_mel.size(0), max_len)
            < valid_lens.unsqueeze(1)
        ).unsqueeze(-1)

        if self.lower_mel_loss_type == "mse":
            frame_loss = (denoised_mel - clean_feats) ** 2
        else:
            frame_loss = torch.abs(denoised_mel - clean_feats)

        frame_loss = frame_loss * mask
        denom = mask.sum().clamp_min(1)
        return frame_loss.sum() / denom

    def _run_lower_transducer(
        self,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        decoder_in: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        predictor_lens = decoder_in.ne(0).sum(dim=1) + 1
        predictor_lens = predictor_lens.clamp_max(decoder_in.size(1))

        if self.freeze_lower and self.training_stage == "upper" and self.training:
            with torch.no_grad():
                lower_out = self.lower_transducer(
                    feats,
                    feats_lengths,
                    decoder_in,
                    predictor_lens,
                    training=False,
                )
        else:
            lower_out = self.lower_transducer(
                feats,
                feats_lengths,
                decoder_in,
                predictor_lens,
                training=self.training and self.training_stage != "upper",
            )

        return lower_out

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        assert text_lengths.dim() == 1, text_lengths.shape
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        batch_size = speech.shape[0]
        text = text[:, : text_lengths.max()]

        feats, feats_lengths = self.encode_features(speech, speech_lengths)

        decoder_in, target, _, u_len = get_transducer_task_io(
            text,
            feats_lengths,
            ignore_id=self.ignore_id,
        )
        lower_out = self._run_lower_transducer(feats, feats_lengths, decoder_in)
        acoustic_mel = lower_out["acoustic_mel"]
        text_mel = lower_out["text_mel"]
        denoised_mel = lower_out["fused_mel"]
        upper_enc_out, upper_enc_lens, upper_dec_out = None, None, None
        loss_trans = speech.new_tensor(0.0)

        if self.training_stage != "lower":
            upper_enc_out, upper_enc_lens = self.upper_encoder(
                denoised_mel, lower_out["acoustic_lengths"]
            )

            self.decoder.set_device(upper_enc_out.device)
            upper_dec_out = self.decoder(decoder_in)

            if self.use_k2_pruned_loss:
                loss_trans = self._calc_k2_transducer_pruned_loss(
                    upper_enc_out,
                    upper_dec_out,
                    text,
                    upper_enc_lens,
                    u_len,
                    **self.k2_pruned_loss_args,
                )
            else:
                joint_out = self.joint_network(
                    upper_enc_out.unsqueeze(2), upper_dec_out.unsqueeze(1)
                )
                loss_trans = self._calc_transducer_loss(
                    upper_enc_out,
                    joint_out,
                    target,
                    upper_enc_lens.int(),
                    u_len,
                )

        loss_ctc, loss_lm = 0.0, 0.0
        loss_lower_mel_acoustic, loss_lower_mel_text, loss_lower_mel_fused = 0.0, 0.0, 0.0
        loss_lower_gate_reg = speech.new_tensor(0.0)

        if self.training_stage != "lower" and self.use_auxiliary_ctc:
            loss_ctc = self._calc_ctc_loss(upper_enc_out, target, upper_enc_lens.int(), u_len)

        if self.training_stage != "lower" and self.use_auxiliary_lm_loss:
            loss_lm = self._calc_lm_loss(upper_dec_out, target)

        clean_speech = kwargs.get("clean_speech", kwargs.get("speech_ref1"))
        clean_speech_lengths = kwargs.get(
            "clean_speech_lengths", kwargs.get("speech_ref1_lengths")
        )
        if (
            self.lower_mel_loss_weight > 0.0
            and clean_speech is not None
            and clean_speech_lengths is not None
        ):
            clean_speech = clean_speech.to(dtype=speech.dtype)
            clean_feats, clean_feats_lens = self._extract_feats(
                clean_speech, clean_speech_lengths
            )
            if self.normalize is not None:
                clean_feats, clean_feats_lens = self.normalize(clean_feats, clean_feats_lens)
            loss_lower_mel_acoustic = self._calc_lower_mel_loss(
                acoustic_mel,
                lower_out["acoustic_lengths"].int(),
                clean_feats,
                clean_feats_lens.int(),
            )
            loss_lower_mel_text = self._calc_lower_mel_loss(
                text_mel,
                lower_out["acoustic_lengths"].int(),
                clean_feats,
                clean_feats_lens.int(),
            )
            loss_lower_mel_fused = self._calc_lower_mel_loss(
                denoised_mel,
                lower_out["acoustic_lengths"].int(),
                clean_feats,
                clean_feats_lens.int(),
            )
            if self.lower_gate_reg_weight > 0.0:
                loss_lower_gate_reg = lower_out["gate_alpha"].mean()
        elif self.training_stage == "lower":
            raise RuntimeError(
                "Lower-stage training requires clean_speech or speech_ref1 targets"
            )

        lower_loss = self.lower_mel_loss_weight * (
            self.lower_mel_acoustic_weight * loss_lower_mel_acoustic
            + self.lower_mel_text_weight * loss_lower_mel_text
            + self.lower_mel_fused_weight * loss_lower_mel_fused
        )
        lower_loss = lower_loss + self.lower_gate_reg_weight * loss_lower_gate_reg

        if self.training_stage == "lower":
            loss = lower_loss
        else:
            loss = (
                self.transducer_weight * loss_trans
                + self.auxiliary_ctc_weight * loss_ctc
                + self.auxiliary_lm_loss_weight * loss_lm
                + lower_loss
            )

        if (
            self.training_stage != "lower"
            and not self.training
            and (self.report_cer or self.report_wer)
        ):
            if self.error_calculator is None:
                from espnet2.asr_transducer.error_calculator import ErrorCalculator

                self.error_calculator = ErrorCalculator(
                    self.decoder,
                    self.joint_network,
                    self.token_list,
                    self.sym_space,
                    self.sym_blank,
                    nstep=self.validation_nstep,
                    report_cer=self.report_cer,
                    report_wer=self.report_wer,
                )

            cer_transducer, wer_transducer = self.error_calculator(
                upper_enc_out, target, upper_enc_lens.int()
            )
        else:
            cer_transducer, wer_transducer = None, None

        stats = dict(
            loss=loss.detach(),
            loss_transducer=(
                loss_trans.detach() if isinstance(loss_trans, torch.Tensor) else None
            ),
            loss_aux_ctc=loss_ctc.detach() if loss_ctc > 0.0 else None,
            loss_aux_lm=loss_lm.detach() if loss_lm > 0.0 else None,
            loss_lower_mel_acoustic=(
                loss_lower_mel_acoustic.detach()
                if isinstance(loss_lower_mel_acoustic, torch.Tensor)
                else None
            ),
            loss_lower_mel_text=(
                loss_lower_mel_text.detach()
                if isinstance(loss_lower_mel_text, torch.Tensor)
                else None
            ),
            loss_lower_mel_fused=(
                loss_lower_mel_fused.detach()
                if isinstance(loss_lower_mel_fused, torch.Tensor)
                else None
            ),
            loss_lower_gate_reg=(
                loss_lower_gate_reg.detach()
                if isinstance(loss_lower_gate_reg, torch.Tensor)
                and self.lower_gate_reg_weight > 0.0
                else None
            ),
            cer_transducer=cer_transducer,
            wer_transducer=wer_transducer,
            lower_gate_mean=lower_out["gate_alpha"].detach().mean(),
            lower_guidance_mean=lower_out["guided_alpha"].detach().mean(),
            lower_prior_abs=lower_out["prior"].detach().abs().mean(),
            acoustic_mel_abs=acoustic_mel.detach().abs().mean(),
            text_mel_abs=text_mel.detach().abs().mean(),
            denoised_mel_abs=denoised_mel.detach().abs().mean(),
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, feats_lengths = self.encode_features(speech, speech_lengths)
        return self.encoder(feats, feats_lengths)

    def encode_with_text_history(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text_history: Union[torch.Tensor, List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode speech using lower text-mel conditioning at inference time.

        This supports a two-pass inference path:
        1. decode once with acoustic-only lower path
        2. reuse the predicted token history to regenerate lower text mel
        3. run the upper encoder again on the fused mel representation
        """
        feats, feats_lengths = self.encode_features(speech, speech_lengths)

        if isinstance(text_history, list):
            decoder_in = torch.tensor(
                [text_history], device=feats.device, dtype=torch.long
            )
        else:
            decoder_in = text_history.to(device=feats.device, dtype=torch.long)
            if decoder_in.dim() == 1:
                decoder_in = decoder_in.unsqueeze(0)

        lower_out = self._run_lower_transducer(feats, feats_lengths, decoder_in)
        return self.upper_encoder(
            lower_out["fused_mel"], lower_out["acoustic_lengths"]
        )

    def encode_with_prefix_features(
        self,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        prefix: Union[torch.Tensor, List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run lower+upper encoding using a predicted token prefix.

        Args:
            feats: Normalized acoustic features. (B, T, F)
            feats_lengths: Acoustic feature lengths. (B,)
            prefix: Decoder input prefix including the initial blank token, or
                a list of token ids.
        """
        if isinstance(prefix, list):
            decoder_in = torch.tensor([prefix], device=feats.device, dtype=torch.long)
        else:
            decoder_in = prefix.to(device=feats.device, dtype=torch.long)
            if decoder_in.dim() == 1:
                decoder_in = decoder_in.unsqueeze(0)

        lower_out = self._run_lower_transducer(feats, feats_lengths, decoder_in)
        return self.upper_encoder(
            lower_out["fused_mel"], lower_out["acoustic_lengths"]
        )

    def encode_features(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with autocast(False):
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        return feats, feats_lengths
