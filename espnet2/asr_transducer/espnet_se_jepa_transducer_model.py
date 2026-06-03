"""SE + JEPA + ASR Transducer model."""

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import soundfile as sf
import torch
import torch.nn.functional as F
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


class SEMelRefiner(torch.nn.Module):
    """Lightweight mel-space refiner before the ASR encoder."""

    def __init__(self, mel_dim: int, hidden_size: int = 256, dropout_rate: float = 0.1):
        super().__init__()
        self.pre_norm = torch.nn.LayerNorm(mel_dim)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(mel_dim, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, mel_dim),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return feats + self.network(self.pre_norm(feats))


class ESPnetASRSEJEPATransducerModel(ESPnetASRTransducerModel):
    """Enhanced-speech transducer with JEPA regularization."""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: Encoder,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
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
        input_size: int = 80,
        se_hidden_size: int = 256,
        se_dropout_rate: float = 0.1,
        jepa_loss_weight: float = 0.1,
        jepa_mask_prob: float = 0.3,
        jepa_mask_span: int = 4,
        jepa_hidden_size: int = 256,
        jepa_predictor_hidden_size: int = 512,
        use_model_jepa_modules: bool = True,
        se_loss_weight: float = 0.0,
        clean_feature_loss_weight: float = 0.0,
        clean_feature_cos_weight: float = 0.25,
        clean_train_scp: Optional[str] = None,
        clean_valid_scp: Optional[str] = None,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
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

        self.use_model_jepa_modules = bool(use_model_jepa_modules)
        if self.use_model_jepa_modules:
            self.se_refiner = SEMelRefiner(
                mel_dim=input_size,
                hidden_size=se_hidden_size,
                dropout_rate=se_dropout_rate,
            )
        else:
            self.se_refiner = torch.nn.Identity()
        self.jepa_loss_weight = jepa_loss_weight
        self.jepa_mask_prob = jepa_mask_prob
        self.jepa_mask_span = max(1, jepa_mask_span)
        if self.use_model_jepa_modules:
            self.jepa_input_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(input_size),
                torch.nn.Linear(input_size, jepa_hidden_size),
                torch.nn.SiLU(),
            )
            self.jepa_projector = torch.nn.Sequential(
                torch.nn.Linear(jepa_hidden_size, jepa_hidden_size),
                torch.nn.LayerNorm(jepa_hidden_size),
                torch.nn.SiLU(),
                torch.nn.Linear(jepa_hidden_size, jepa_hidden_size),
            )
            self.jepa_predictor = torch.nn.Sequential(
                torch.nn.Linear(jepa_hidden_size, jepa_predictor_hidden_size),
                torch.nn.SiLU(),
                torch.nn.Linear(jepa_predictor_hidden_size, jepa_hidden_size),
            )
            # Decode JEPA latent back to mel space and optimize masked-region reconstruction.
            self.jepa_decoder = torch.nn.Sequential(
                torch.nn.Linear(jepa_hidden_size, jepa_predictor_hidden_size),
                torch.nn.SiLU(),
                torch.nn.Linear(jepa_predictor_hidden_size, input_size),
            )
        self.se_loss_weight = float(se_loss_weight)
        self.clean_feature_loss_weight = float(clean_feature_loss_weight)
        self.clean_feature_cos_weight = float(clean_feature_cos_weight)
        self.clean_utt2wav = {}
        for scp in [clean_train_scp, clean_valid_scp]:
            if scp is None:
                continue
            self.clean_utt2wav.update(self._read_scp(scp))

    def _read_scp(self, scp_path: str) -> Dict[str, str]:
        table = {}
        p = Path(scp_path)
        if not p.is_file():
            return table
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                table[parts[0]] = parts[1]
        return table

    def _load_clean_batch(
        self,
        utt_ids: List[str],
        speech_lengths: torch.Tensor,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.clean_utt2wav:
            return None, None

        waveforms: List[torch.Tensor] = []
        lengths: List[int] = []
        for uid, n_len in zip(utt_ids, speech_lengths.tolist()):
            wav_path = self.clean_utt2wav.get(uid, None)
            if wav_path is None:
                return None, None
            wav_np, _sr = sf.read(wav_path, dtype="float32")
            if wav_np.ndim > 1:
                wav_np = wav_np[:, 0]
            wav = torch.from_numpy(wav_np)
            cur_len = int(min(int(n_len), wav.numel()))
            waveforms.append(wav[:cur_len])
            lengths.append(cur_len)

        if not lengths:
            return None, None

        max_len = max(lengths)
        clean = torch.zeros(len(waveforms), max_len, dtype=torch.float32)
        for i, w in enumerate(waveforms):
            clean[i, : w.numel()] = w
        return clean.to(device), torch.tensor(lengths, dtype=torch.long, device=device)

    def _calc_clean_refine_loss(
        self,
        refined_feats: torch.Tensor,
        speech_lengths: torch.Tensor,
        feats_lengths: torch.Tensor,
        utt_ids: Optional[List[str]],
    ) -> torch.Tensor:
        if (not self.training) or self.clean_feature_loss_weight <= 0.0 or not utt_ids:
            return refined_feats.new_tensor(0.0)

        clean_speech, clean_lengths = self._load_clean_batch(
            utt_ids=utt_ids,
            speech_lengths=speech_lengths,
            device=refined_feats.device,
        )
        if clean_speech is None:
            return refined_feats.new_tensor(0.0)

        with torch.no_grad():
            clean_feats, clean_feats_lengths = self._extract_feats(clean_speech, clean_lengths)
            if self.normalize is not None:
                clean_feats, clean_feats_lengths = self.normalize(
                    clean_feats, clean_feats_lengths
                )

        min_len = torch.minimum(clean_feats_lengths, feats_lengths)
        max_t = int(min_len.max().item())
        if max_t <= 0:
            return refined_feats.new_tensor(0.0)

        ref = refined_feats[:, :max_t]
        tgt = clean_feats[:, :max_t]
        b, t, _d = ref.shape
        idx = torch.arange(t, device=ref.device).unsqueeze(0).expand(b, -1)
        valid = (idx < min_len.unsqueeze(1)).unsqueeze(-1).to(ref.dtype)
        denom = valid.sum().clamp_min(1.0)

        l1 = torch.abs(ref - tgt).mul(valid).sum() / denom
        cos = 1.0 - F.cosine_similarity(ref, tgt, dim=-1)
        cos = (cos.unsqueeze(-1) * valid).sum() / denom
        return l1 + self.clean_feature_cos_weight * cos

    def _compute_jepa_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = lengths.size(0)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        for batch_idx in range(batch_size):
            valid_len = int(lengths[batch_idx].item())
            if valid_len <= 0:
                continue

            base_mask = torch.rand(valid_len, device=device) < self.jepa_mask_prob
            if self.jepa_mask_span > 1 and base_mask.any():
                for shift in range(1, self.jepa_mask_span):
                    previous_mask = base_mask[:-shift].clone()
                    base_mask[shift:] |= previous_mask

            if not base_mask.any():
                forced_index = torch.randint(valid_len, (1,), device=device)
                base_mask[forced_index] = True

            mask[batch_idx, :valid_len] = base_mask

        return mask

    def _calc_jepa_loss(
        self,
        refined_feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        clean_feats: Optional[torch.Tensor] = None,
        clean_feats_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.jepa_loss_weight <= 0.0:
            zero = refined_feats.new_tensor(0.0)
            return zero, zero

        target_feats = refined_feats
        target_lens = feats_lengths
        max_t = refined_feats.size(1)
        if clean_feats is not None and clean_feats_lengths is not None:
            max_t = min(max_t, clean_feats.size(1))
            if max_t <= 0:
                zero = refined_feats.new_tensor(0.0)
                return zero, zero
            refined_feats = refined_feats[:, :max_t]
            clean_feats = clean_feats[:, :max_t]
            target_feats = clean_feats
            target_lens = torch.minimum(feats_lengths, clean_feats_lengths)
        else:
            refined_feats = refined_feats[:, :max_t]

        time_mask = self._compute_jepa_mask(target_lens, max_t, refined_feats.device)
        masked_feats = refined_feats.masked_fill(time_mask.unsqueeze(-1), 0.0)
        student_hidden = self.jepa_input_proj(masked_feats)  # B,T,H
        student_latent = self.jepa_predictor(self.jepa_projector(student_hidden))  # B,T,H
        reconstructed = self.jepa_decoder(student_latent)  # B,T,F

        masked_positions = time_mask.unsqueeze(-1)
        if not masked_positions.any():
            zero = refined_feats.new_tensor(0.0)
            return zero, zero

        # JEPA masked reconstruction objective: predict masked clean-aware mel targets.
        l2 = (reconstructed - target_feats).pow(2)
        loss = l2.masked_select(masked_positions).mean()
        mask_ratio = time_mask.float().sum() / target_lens.sum().clamp_min(1)
        return loss, mask_ratio

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, feats_lengths = self.encode_features(speech, speech_lengths)
        refined_feats = self.se_refiner(feats)
        return self.encoder(refined_feats, feats_lengths)

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
        utt_ids = kwargs.get("utt_id", None)
        clean_speech = kwargs.get("clean_speech", None)
        clean_speech_lengths = kwargs.get("clean_speech_lengths", None)

        # If frontend is SE_JEPAFrontend, pass clean speech so frontend JEPA/SE
        # objectives are computed on the same forward path as AED/CTC recipes.
        if self.frontend is not None and hasattr(self.frontend, "jepa_frontend"):
            with autocast(False):
                feats, feats_lengths = self.frontend(
                    speech,
                    speech_lengths,
                    clean_input=clean_speech,
                    clean_input_lengths=clean_speech_lengths,
                )
                if self.specaug is not None and self.training:
                    feats, feats_lengths = self.specaug(feats, feats_lengths)
                if self.normalize is not None:
                    feats, feats_lengths = self.normalize(feats, feats_lengths)
        else:
            feats, feats_lengths = self.encode_features(speech, speech_lengths)
        refined_feats = self.se_refiner(feats)
        encoder_out, encoder_out_lens = self.encoder(refined_feats, feats_lengths)

        decoder_in, target, t_len, u_len = get_transducer_task_io(
            text,
            encoder_out_lens,
            ignore_id=self.ignore_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        if self.use_k2_pruned_loss:
            loss_trans, _, _ = self._calc_k2_transducer_pruned_loss(
                encoder_out, decoder_out, text, t_len, u_len, **self.k2_pruned_loss_args
            )
        else:
            joint_out = self.joint_network(
                encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
            )
            loss_trans = self._calc_transducer_loss(
                encoder_out,
                joint_out,
                target,
                t_len,
                u_len,
            )

        loss_ctc, loss_lm = 0.0, 0.0
        if self.use_auxiliary_ctc:
            loss_ctc = self._calc_ctc_loss(encoder_out, target, t_len, u_len)
        if self.use_auxiliary_lm_loss:
            loss_lm = self._calc_lm_loss(decoder_out, target)

        # Prefer frontend JEPA objective when frontend exposes it (SE_JEPAFrontend).
        # Fallback to model-side masked JEPA objective otherwise.
        loss_jepa = refined_feats.new_tensor(0.0)
        jepa_mask_ratio = None
        used_frontend_jepa = False
        if self.training and self.jepa_loss_weight > 0.0 and hasattr(self.frontend, "compute_jepa_loss"):
            frontend_jepa = self.frontend.compute_jepa_loss()
            if frontend_jepa is not None:
                loss_jepa = frontend_jepa
                used_frontend_jepa = True
        clean_feats = None
        clean_feats_lengths = None
        need_clean_feats = (
            self.clean_feature_loss_weight > 0.0
            or (self.jepa_loss_weight > 0.0 and not used_frontend_jepa)
        )
        if need_clean_feats and clean_speech is not None and clean_speech_lengths is not None:
            with torch.no_grad():
                clean_feats, clean_feats_lengths = self.encode_features(
                    clean_speech, clean_speech_lengths
                )
        if self.use_model_jepa_modules and (not used_frontend_jepa) and self.jepa_loss_weight > 0.0:
            loss_jepa, jepa_mask_ratio = self._calc_jepa_loss(
                refined_feats,
                feats_lengths.int(),
                clean_feats=clean_feats,
                clean_feats_lengths=(
                    clean_feats_lengths.int() if clean_feats_lengths is not None else None
                ),
            )
        loss_clean_refine = self._calc_clean_refine_loss(
            refined_feats=refined_feats,
            speech_lengths=speech_lengths,
            feats_lengths=feats_lengths.int(),
            utt_ids=utt_ids,
        )
        loss_se = refined_feats.new_tensor(0.0)
        if self.training and self.se_loss_weight > 0.0 and hasattr(self.frontend, "compute_se_loss"):
            frontend_se = self.frontend.compute_se_loss()
            if frontend_se is not None:
                loss_se = frontend_se

        loss = (
            self.transducer_weight * loss_trans
            + self.auxiliary_ctc_weight * loss_ctc
            + self.auxiliary_lm_loss_weight * loss_lm
            + self.jepa_loss_weight * loss_jepa
            + self.clean_feature_loss_weight * loss_clean_refine
            + self.se_loss_weight * loss_se
        )

        if not self.training and (self.report_cer or self.report_wer):
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
                encoder_out, target, t_len
            )
        else:
            cer_transducer, wer_transducer = None, None

        stats = dict(
            loss=loss.detach(),
            loss_transducer=loss_trans.detach(),
            loss_aux_ctc=loss_ctc.detach() if loss_ctc > 0.0 else None,
            loss_aux_lm=loss_lm.detach() if loss_lm > 0.0 else None,
            loss_jepa=loss_jepa.detach() if self.jepa_loss_weight > 0.0 else None,
            loss_clean_refine=(
                loss_clean_refine.detach()
                if self.clean_feature_loss_weight > 0.0
                else None
            ),
            loss_se=loss_se.detach() if self.se_loss_weight > 0.0 else None,
            cer_transducer=cer_transducer,
            wer_transducer=wer_transducer,
            refined_mel_abs=refined_feats.detach().abs().mean(),
            jepa_mask_ratio=(
                jepa_mask_ratio.detach()
                if (self.jepa_loss_weight > 0.0 and jepa_mask_ratio is not None)
                else None
            ),
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

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
