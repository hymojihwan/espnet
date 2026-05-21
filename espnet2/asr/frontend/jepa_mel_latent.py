"""Mel-domain JEPA frontend for noise-robust ASR.

This frontend learns clean-aware latent representations directly from noisy speech:
    noisy mel -> context latent -> predictor -> clean target latent

Unlike reconstruction-focused SE, this module does not reconstruct clean mel.
It predicts latent targets on masked regions and outputs latent features to ASR.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft


class JEPAMelLatentFrontend(AbsFrontend):
    """Mel-domain JEPA latent frontend for ASR."""

    def __init__(
        self,
        output_dim: int = 256,
        embedding_dim: int = 256,
        predictor_dim: int = 512,
        num_predictor_layers: int = 2,
        dropout_rate: float = 0.1,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
        fs: int = 16000,
        n_mels: int = 80,
        mask_ratio: float = 0.3,
        ema_decay: float = 0.999,
        embedding_loss_weight: float = 1.0,
        var_loss_weight: float = 0.0,
        cov_loss_weight: float = 0.0,
        var_target: float = 1.0,
        loss_type: str = "l2",
        min_mask_frames: int = 5,
        min_num_mask_blocks: int = 1,
        max_num_mask_blocks: int = 1,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        if loss_type not in ("cosine", "mse", "l2"):
            raise ValueError(
                f"loss_type must be 'cosine', 'mse', or 'l2', got: {loss_type}"
            )
        if isinstance(fs, str):
            fs_lower = fs.strip().lower()
            if fs_lower.endswith("k"):
                fs = int(float(fs_lower[:-1]) * 1000)
            else:
                fs = int(float(fs_lower))

        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.embedding_loss_weight = embedding_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.var_target = var_target
        self.loss_type = loss_type
        self.min_mask_frames = min_mask_frames
        self.min_num_mask_blocks = max(1, min_num_mask_blocks)
        self.max_num_mask_blocks = max(self.min_num_mask_blocks, max_num_mask_blocks)

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )
        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=None,
            fmax=None,
            htk=False,
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(n_mels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(n_mels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.target_encoder.load_state_dict(self.context_encoder.state_dict(), strict=False)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

        predictor_layers = []
        in_dim = embedding_dim
        for _ in range(num_predictor_layers):
            predictor_layers.extend(
                [
                    nn.Linear(in_dim, predictor_dim),
                    nn.LayerNorm(predictor_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_dim = predictor_dim
        predictor_layers.append(nn.Linear(in_dim, embedding_dim))
        self.predictor = nn.Sequential(*predictor_layers)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.out_proj = (
            nn.Identity()
            if output_dim == embedding_dim
            else nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, output_dim))
        )

        self._last_predicted_embeddings = None
        self._last_target_embeddings = None
        self._last_time_mask = None
        self._last_valid_mask = None

    def output_size(self) -> int:
        return self.output_dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def update_target_encoder(self):
        with torch.no_grad():
            tau = self.ema_decay
            src = dict(self.context_encoder.named_parameters())
            for name_t, p_t in self.target_encoder.named_parameters():
                p_s = src[name_t]
                # theta_tgt <- tau * theta_tgt + (1 - tau) * theta_ctx
                p_t.data.mul_(tau).add_(p_s.data, alpha=1 - tau)

    def _extract_logmel(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_stft, feats_lens = self.stft(input, input_lengths)
        if isinstance(input_stft, torch.Tensor):
            input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        input_power = input_stft.real**2 + input_stft.imag**2
        feats, feats_lens = self.logmel(input_power, feats_lens)
        return feats, feats_lens

    def _generate_block_mask(
        self, lengths: torch.Tensor, max_t: int, device: torch.device
    ) -> torch.Tensor:
        batch_size = lengths.size(0)
        time_mask = torch.zeros(batch_size, max_t, dtype=torch.bool, device=device)
        for b in range(batch_size):
            valid_length = int(lengths[b].item())
            if valid_length <= 0:
                continue
            num_mask = int(valid_length * self.mask_ratio)
            num_mask = max(num_mask, min(self.min_mask_frames, valid_length))
            num_mask = min(num_mask, valid_length)
            if num_mask <= 0:
                continue
            num_blocks = int(
                torch.randint(
                    self.min_num_mask_blocks,
                    self.max_num_mask_blocks + 1,
                    (1,),
                    device=device,
                ).item()
            )
            num_blocks = max(1, min(num_blocks, num_mask))
            remaining = num_mask
            for ib in range(num_blocks):
                blocks_left = num_blocks - ib
                block_len = max(1, remaining // blocks_left)
                if blocks_left > 1:
                    jitter = int(
                        torch.randint(0, max(1, block_len), (1,), device=device).item()
                    )
                    block_len = min(remaining - (blocks_left - 1), block_len + jitter)
                start_max = max(1, valid_length - block_len + 1)
                start = torch.randint(0, start_max, (1,), device=device).item()
                time_mask[b, start : start + block_len] = True
                remaining -= block_len
                if remaining <= 0:
                    break
        return time_mask

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        clean_input: Optional[torch.Tensor] = None,
        clean_input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_mel, feats_lens = self._extract_logmel(input, input_lengths)
        context_embeddings = self.context_encoder(noisy_mel)
        batch_size, max_t, _ = context_embeddings.shape

        valid_mask = torch.arange(max_t, device=feats_lens.device)[None, :] < feats_lens[:, None]
        if self.training:
            time_mask = self._generate_block_mask(feats_lens, max_t, context_embeddings.device)
            time_mask = time_mask & valid_mask
        else:
            time_mask = torch.zeros(batch_size, max_t, dtype=torch.bool, device=context_embeddings.device)

        masked_context = context_embeddings.clone()
        if time_mask.any():
            masked_context[time_mask] = self.mask_token
        predicted_embeddings = self.predictor(masked_context)

        if self.training and clean_input is not None:
            clean_lens = clean_input_lengths if clean_input_lengths is not None else input_lengths
            clean_mel, clean_feats_lens = self._extract_logmel(clean_input, clean_lens)
            tgt_t = min(predicted_embeddings.size(1), clean_mel.size(1))
            predicted_for_loss = predicted_embeddings[:, :tgt_t]
            valid_for_loss = valid_mask[:, :tgt_t]
            mask_for_loss = time_mask[:, :tgt_t]
            clean_mel = clean_mel[:, :tgt_t]
            clean_feats_lens = torch.minimum(clean_feats_lens, feats_lens).clamp(max=tgt_t)
            clean_valid = (
                torch.arange(tgt_t, device=clean_feats_lens.device)[None, :]
                < clean_feats_lens[:, None]
            )
            joint_valid = valid_for_loss & clean_valid

            with torch.no_grad():
                target_embeddings = self.target_encoder(clean_mel)

            self._last_predicted_embeddings = predicted_for_loss
            self._last_target_embeddings = target_embeddings
            self._last_time_mask = mask_for_loss
            self._last_valid_mask = joint_valid
        else:
            self._last_predicted_embeddings = None
            self._last_target_embeddings = None
            self._last_time_mask = None
            self._last_valid_mask = None

        output = self.out_proj(predicted_embeddings)
        return output, feats_lens

    def compute_jepa_loss(self) -> Optional[torch.Tensor]:
        if (
            self._last_predicted_embeddings is None
            or self._last_target_embeddings is None
            or self._last_time_mask is None
            or self._last_valid_mask is None
        ):
            return None

        masked_valid = self._last_time_mask & self._last_valid_mask
        if not masked_valid.any():
            return None

        pred = self._last_predicted_embeddings[masked_valid]
        target = self._last_target_embeddings[masked_valid].detach()
        if self.loss_type == "l2":
            diff = pred - target
            loss_align = diff.pow(2).sum(dim=-1).mean()
        elif self.loss_type == "mse":
            loss_align = F.mse_loss(pred, target, reduction="mean")
        else:
            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)
            loss_align = (1 - (pred * target).sum(dim=-1)).mean()

        total = loss_align * self.embedding_loss_weight

        if self.var_loss_weight > 0 and pred.size(0) > 1:
            std = torch.sqrt(pred.var(dim=0, unbiased=False) + 1.0e-4)
            loss_var = torch.mean(F.relu(self.var_target - std))
            total = total + self.var_loss_weight * loss_var

        if self.cov_loss_weight > 0 and pred.size(0) > 1:
            pred_centered = pred - pred.mean(dim=0, keepdim=True)
            cov = (pred_centered.T @ pred_centered) / (pred_centered.size(0) - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            loss_cov = (off_diag.pow(2).sum() / pred.size(1))
            total = total + self.cov_loss_weight * loss_cov

        return total
