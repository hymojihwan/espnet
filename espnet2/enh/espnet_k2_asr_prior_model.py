"""Enhancement model with k2 ASR posterior prior."""

from typing import Dict, List, Optional, OrderedDict, Tuple

import torch

from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.tasks.asr_transducer import ASRTransducerTask


class ESPnetEnhancementK2ASRPriorModel(ESPnetEnhancementModel):
    """Speech enhancement model with on-the-fly k2 ASR prior."""

    def __init__(
        self,
        *args,
        use_k2_asr_prior: bool = False,
        asr_train_config: Optional[str] = None,
        asr_model_file: Optional[str] = None,
        asr_prior_chunk_length: int = 0,
        asr_prior_blank_id: int = 0,
        asr_prior_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_k2_asr_prior = use_k2_asr_prior
        self.asr_prior_chunk_length = asr_prior_chunk_length
        self.asr_prior_blank_id = asr_prior_blank_id
        self.asr_prior_scale = asr_prior_scale
        self.asr_model = None

        if self.use_k2_asr_prior:
            if asr_train_config is None or asr_model_file is None:
                raise ValueError(
                    "asr_train_config and asr_model_file are required for k2 ASR prior"
                )
            self.asr_model, _ = ASRTransducerTask.build_model_from_file(
                config_file=asr_train_config, model_file=asr_model_file, device="cpu"
            )
            if not getattr(self.asr_model, "use_k2_pruned_loss", False):
                raise ValueError(
                    "ASR model must be trained with use_k2_pruned_loss for k2 prior"
                )
            self.asr_model.eval()
            for p in self.asr_model.parameters():
                p.requires_grad = False

    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        additional: Optional[Dict] = None,
        fs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_mix, flens = self.encoder(speech_mix, speech_lengths, fs=fs)

        if self.use_text_prior and additional is not None:
            prior = self._build_text_prior(
                token_int=additional.get("token_int", None),
                token_int_lengths=additional.get("token_int_lengths", None),
                token_time_ms=additional.get("token_time_ms", None),
                token_time_ms_lengths=additional.get("token_time_ms_lengths", None),
                flens=flens,
                speech_lengths=speech_lengths,
                fs=fs,
                dtype=feature_mix.dtype,
            )
            if prior is not None:
                feature_mix = feature_mix + prior

        if self.use_chunk_text_prior and additional is not None:
            chunk_prior = self._build_chunk_text_prior(
                token_chunk_int=additional.get("token_chunk_int", None),
                token_chunk_int_lengths=additional.get(
                    "token_chunk_int_lengths", None
                ),
                flens=flens,
                dtype=feature_mix.dtype,
            )
            if chunk_prior is not None:
                feature_mix = feature_mix + chunk_prior

        if self.use_k2_asr_prior and additional is not None:
            k2_prior = self._build_k2_asr_chunk_prior(
                speech_mix_raw=additional.get("speech_mix_raw", None),
                speech_lengths=speech_lengths,
                token_int=additional.get("token_int", None),
                token_int_lengths=additional.get("token_int_lengths", None),
                flens=flens,
                dtype=feature_mix.dtype,
            )
            if k2_prior is not None:
                feature_mix = feature_mix + k2_prior

        if self.mask_module is None:
            feature_pre, flens, others = self.separator(feature_mix, flens, additional)
        else:
            bottleneck_feats, bottleneck_feats_lengths = self.separator(
                feature_mix, flens
            )
            if additional.get("num_spk") is not None:
                feature_pre, flens, others = self.mask_module(
                    feature_mix, flens, bottleneck_feats, additional["num_spk"]
                )
                others["bottleneck_feats"] = bottleneck_feats
                others["bottleneck_feats_lengths"] = bottleneck_feats_lengths
            else:
                feature_pre = None
                others = {
                    "bottleneck_feats": bottleneck_feats,
                    "bottleneck_feats_lengths": bottleneck_feats_lengths,
                }

        if feature_pre is not None:
            pre_is_multi_list = isinstance(feature_pre[0], (list, tuple))
            if pre_is_multi_list:
                speech_pre = [
                    [self.decoder(p, speech_lengths, fs=fs)[0] for p in ps]
                    for ps in feature_pre
                ]
            else:
                speech_pre = [
                    self.decoder(ps, speech_lengths, fs=fs)[0] for ps in feature_pre
                ]
        else:
            speech_pre = None

        return speech_pre, feature_mix, feature_pre, others

    def _build_k2_asr_chunk_prior(
        self,
        speech_mix_raw: Optional[torch.Tensor],
        speech_lengths: torch.Tensor,
        token_int: Optional[torch.Tensor],
        token_int_lengths: Optional[torch.Tensor],
        flens: torch.Tensor,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if speech_mix_raw is None:
            return None
        if token_int is None or token_int_lengths is None:
            return None
        if self.asr_model is None:
            return None
        if self.asr_prior_chunk_length <= 0:
            return None

        device = speech_mix_raw.device
        self.asr_model = self.asr_model.to(device)

        max_u = token_int.size(1)
        mask = (
            torch.arange(max_u, device=device).unsqueeze(0)
            >= token_int_lengths.unsqueeze(1)
        )
        token_int_fixed = token_int.masked_fill(mask, self.asr_model.ignore_id)

        with torch.no_grad():
            encoder_out, encoder_out_lens = self.asr_model.encode(
                speech_mix_raw, speech_lengths
            )
            decoder_in, _, t_len, u_len = get_transducer_task_io(
                token_int_fixed,
                encoder_out_lens,
                ignore_id=self.asr_model.ignore_id,
                blank_id=self.asr_prior_blank_id,
            )
            self.asr_model.decoder.set_device(encoder_out.device)
            decoder_out = self.asr_model.decoder(decoder_in)
            loss_out = self.asr_model._calc_k2_transducer_pruned_loss(
                encoder_out,
                decoder_out,
                token_int_fixed,
                t_len,
                u_len,
                reduction="none",
                return_px_grad=True,
            )
            _, _, _, px_grad = loss_out

            posterior = -px_grad
            top1 = posterior.argmax(dim=-1)  # (B, T)

            hop_length = getattr(
                getattr(self.asr_model, "frontend", None), "hop_length", 160
            )
            subsampling = getattr(
                getattr(getattr(self.asr_model, "encoder", None), "embed", None),
                "subsampling_factor",
                1,
            )
            frame_shift = max(1, int(hop_length * max(subsampling, 1)))
            frames_per_chunk = max(
                1, int(round(self.asr_prior_chunk_length / float(frame_shift)))
            )

            batch_size = top1.size(0)
            max_chunks = int(
                torch.ceil(
                    speech_lengths.float() / float(self.asr_prior_chunk_length)
                ).max()
            )
            token_chunk_int = torch.full(
                (batch_size, max_chunks),
                fill_value=self.asr_prior_blank_id,
                dtype=torch.long,
                device=device,
            )
            token_chunk_int_lengths = torch.zeros(
                (batch_size,), dtype=torch.long, device=device
            )

            for b in range(batch_size):
                t_len_b = int(t_len[b].item())
                frames = top1[b, :t_len_b]
                num_chunks = int(
                    torch.ceil(
                        speech_lengths[b].float() / float(self.asr_prior_chunk_length)
                    ).item()
                )
                token_chunk_int_lengths[b] = num_chunks
                for c in range(num_chunks):
                    s = c * frames_per_chunk
                    e = min((c + 1) * frames_per_chunk, t_len_b)
                    if s >= t_len_b:
                        break
                    chunk = frames[s:e]
                    non_blank = chunk[chunk != self.asr_prior_blank_id]
                    if non_blank.numel() == 0:
                        tok = self.asr_prior_blank_id
                    else:
                        counts = torch.bincount(non_blank)
                        tok = int(torch.argmax(counts).item())
                    token_chunk_int[b, c] = tok

        chunk_prior = self._build_chunk_text_prior(
            token_chunk_int=token_chunk_int,
            token_chunk_int_lengths=token_chunk_int_lengths,
            flens=flens,
            dtype=dtype,
        )
        if chunk_prior is None:
            return None
        return chunk_prior * float(self.asr_prior_scale)
