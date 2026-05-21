
"""JEPA Hybrid Masked-Patch Frontend for ASR.

Same as jepa_masked + unmasked loss on mel reconstruction.
Predictor predicts ALL patches (masked + unmasked), decoder reconstructs ALL.
Loss: L_mask + λ*L_unmask (fixed target: clean mel).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class JEPA_HybridFrontend(AbsFrontend):
    """JEPA Hybrid Masked-Patch Frontend for ASR.

    Same as jepa_masked but adds unmasked loss:
    - Predictor predicts ALL patches (masked + unmasked)
    - Decoder reconstructs ALL patches to mel
    - Output to ASR: unmasked from noisy, masked from decoded (same as masked)
    - Loss: L_mask + λ*L_unmask on mel reconstruction (fixed target: clean mel)
    """

    def __init__(
        self,
        output_dim: int = 80,
        embedding_dim: int = 256,
        context_encoder_dim: int = 256,
        target_encoder_dim: int = 256,
        predictor_dim: int = 256,
        decoder_dim: int = 256,
        num_context_encoder_layers: int = 2,
        num_target_encoder_layers: int = 2,
        num_predictor_layers: int = 2,
        num_decoder_layers: int = 2,
        patch_size: Any = (4, 4),
        mask_ratio: float = 0.3,
        dropout_rate: float = 0.1,
        encoder_type: str = "mlp",
        encoder_conf: Optional[Dict[str, Any]] = None,
        predictor_type: str = "mlp",
        predictor_conf: Optional[Dict[str, Any]] = None,
        decoder_type: str = "mlp",
        decoder_conf: Optional[Dict[str, Any]] = None,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
        fs: int = 16000,
        n_mels: int = 80,
        embedding_loss_weight: float = 1.0,
        unmask_loss_weight: float = 0.01,
        ema_decay: float = 0.999,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        if isinstance(patch_size, list):
            patch_size = tuple(patch_size)

        super().__init__()

        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.context_encoder_dim = context_encoder_dim
        self.target_encoder_dim = target_encoder_dim
        self.predictor_dim = predictor_dim
        self.decoder_dim = decoder_dim
        self.num_context_encoder_layers = num_context_encoder_layers
        self.num_target_encoder_layers = num_target_encoder_layers
        self.num_predictor_layers = num_predictor_layers
        self.num_decoder_layers = num_decoder_layers
        self.patch_size: Tuple[int, int] = patch_size
        self.mask_ratio = mask_ratio
        self.dropout_rate = dropout_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.embedding_loss_weight = embedding_loss_weight
        self.unmask_loss_weight = unmask_loss_weight
        self.ema_decay = ema_decay
        self.encoder_type = encoder_type
        self.encoder_conf = encoder_conf or {}
        self.predictor_type = predictor_type
        self.predictor_conf = predictor_conf or {}
        self.decoder_type = decoder_type
        self.decoder_conf = decoder_conf or {}

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

        patch_dim = patch_size[0] * patch_size[1]

        if encoder_type == "transformer":
            encoder_num_layers = self.encoder_conf.get("num_layers", 6)
            encoder_num_heads = self.encoder_conf.get("num_heads", 4)
            encoder_ff_dim = self.encoder_conf.get("ff_dim", 1024)
            encoder_dropout = self.encoder_conf.get("dropout_rate", dropout_rate)
            encoder_attn_dropout = self.encoder_conf.get("attn_dropout_rate", 0.1)
            use_pos_enc = self.encoder_conf.get("use_positional_encoding", True)

            self.context_patch_embed = nn.Linear(patch_dim, embedding_dim)
            self.context_pos_enc = (
                PositionalEncoding(embedding_dim, encoder_dropout)
                if use_pos_enc
                else None
            )
            self.context_encoder = self._build_transformer_encoder(
                embedding_dim,
                encoder_num_layers,
                encoder_num_heads,
                encoder_ff_dim,
                encoder_dropout,
                encoder_attn_dropout,
            )

            self.target_patch_embed = nn.Linear(patch_dim, embedding_dim)
            self.target_pos_enc = (
                PositionalEncoding(embedding_dim, encoder_dropout)
                if use_pos_enc
                else None
            )
            self.target_encoder = self._build_transformer_encoder(
                embedding_dim,
                encoder_num_layers,
                encoder_num_heads,
                encoder_ff_dim,
                encoder_dropout,
                encoder_attn_dropout,
            )
        else:
            context_encoder_layers = []
            input_dim = patch_dim
            for i in range(num_context_encoder_layers):
                context_encoder_layers.extend([
                    nn.Linear(input_dim, context_encoder_dim),
                    nn.LayerNorm(context_encoder_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ])
                input_dim = context_encoder_dim
            context_encoder_layers.append(nn.Linear(input_dim, embedding_dim))
            self.context_encoder = nn.Sequential(*context_encoder_layers)
            self.context_patch_embed = None
            self.context_pos_enc = None

            target_encoder_layers = []
            input_dim = patch_dim
            for i in range(num_target_encoder_layers):
                target_encoder_layers.extend([
                    nn.Linear(input_dim, target_encoder_dim),
                    nn.LayerNorm(target_encoder_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ])
                input_dim = target_encoder_dim
            target_encoder_layers.append(nn.Linear(input_dim, embedding_dim))
            self.target_encoder = nn.Sequential(*target_encoder_layers)
            self.target_patch_embed = None
            self.target_pos_enc = None

        if predictor_type == "transformer":
            pred_num_layers = self.predictor_conf.get("num_layers", num_predictor_layers)
            pred_num_heads = self.predictor_conf.get("num_heads", 4)
            pred_ff_dim = self.predictor_conf.get("ff_dim", 1024)
            pred_dropout = self.predictor_conf.get("dropout_rate", dropout_rate)
            pred_attn_dropout = self.predictor_conf.get("attn_dropout_rate", 0.1)
            use_pos_enc = self.predictor_conf.get("use_positional_encoding", True)

            self.predictor_pos_enc = (
                PositionalEncoding(embedding_dim, pred_dropout) if use_pos_enc else None
            )
            self.predictor = self._build_transformer_encoder(
                embedding_dim,
                pred_num_layers,
                pred_num_heads,
                pred_ff_dim,
                pred_dropout,
                pred_attn_dropout,
            )
        elif predictor_type == "mlp":
            pred_num_layers = self.predictor_conf.get("num_layers", num_predictor_layers)
            pred_hidden_dim = self.predictor_conf.get("hidden_dim", predictor_dim)
            pred_dropout = self.predictor_conf.get("dropout_rate", dropout_rate)

            predictor_layers = []
            input_dim = embedding_dim
            for i in range(pred_num_layers):
                predictor_layers.extend([
                    nn.Linear(input_dim, pred_hidden_dim),
                    nn.LayerNorm(pred_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(pred_dropout),
                ])
                input_dim = pred_hidden_dim
            predictor_layers.append(nn.Linear(input_dim, embedding_dim))
            self.predictor = nn.Sequential(*predictor_layers)
            self.predictor_pos_enc = None
        else:
            raise ValueError(f"Unsupported predictor_type: {predictor_type}")

        if decoder_type == "mlp":
            dec_num_layers = self.decoder_conf.get("num_layers", num_decoder_layers)
            dec_hidden_dim = self.decoder_conf.get("hidden_dim", decoder_dim)
            dec_dropout = self.decoder_conf.get("dropout_rate", dropout_rate)

            decoder_layers = []
            input_dim = embedding_dim
            for i in range(dec_num_layers):
                decoder_layers.extend([
                    nn.Linear(input_dim, dec_hidden_dim),
                    nn.LayerNorm(dec_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dec_dropout),
                ])
                input_dim = dec_hidden_dim
            decoder_layers.append(nn.Linear(input_dim, patch_dim))
            self.decoder = nn.Sequential(*decoder_layers)
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

        if encoder_type == "transformer":
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_dim))

        if encoder_type == "transformer":
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
            self.target_patch_embed.load_state_dict(self.context_patch_embed.state_dict())
            if self.target_pos_enc is not None and self.context_pos_enc is not None:
                self.target_pos_enc.load_state_dict(self.context_pos_enc.state_dict())
        else:
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        for param in self.target_encoder.parameters():
            param.requires_grad = False
        if self.target_patch_embed is not None:
            for param in self.target_patch_embed.parameters():
                param.requires_grad = False
        if self.target_pos_enc is not None:
            for param in self.target_pos_enc.parameters():
                param.requires_grad = False

        self._last_x_n = None
        self._last_x_c = None
        self._last_x_hat = None
        self._last_mask = None
        self._last_patch_grid = None
        self._last_noisy_patches = None
        self._last_clean_patches = None
        self._last_decoded_patches = None
        self._last_patch_mask = None

    def _build_transformer_encoder(
        self,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float,
        attn_dropout_rate: float,
    ) -> nn.Module:
        encoder_layers = repeat(
            num_layers,
            lambda lnum: EncoderLayer(
                embedding_dim,
                MultiHeadedAttention(
                    num_heads,
                    embedding_dim,
                    attn_dropout_rate,
                    qk_norm=False,
                    use_flash_attn=False,
                    causal=False,
                    cross_attn=False,
                ),
                PositionwiseFeedForward(embedding_dim, ff_dim, dropout_rate),
                dropout_rate,
                normalize_before=True,
                concat_after=False,
            ),
        )
        after_norm = LayerNorm(embedding_dim)
        return nn.ModuleDict({"encoders": encoder_layers, "after_norm": after_norm})

    def output_size(self) -> int:
        return self.output_dim

    def _extract_logmel(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_stft, feats_lens = self.stft(input, input_lengths)
        if isinstance(input_stft, torch.Tensor):
            assert input_stft.shape[-1] == 2
            input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        input_power = input_stft.real**2 + input_stft.imag**2
        log_mel, feats_lens = self.logmel(input_power, feats_lens)
        return log_mel, feats_lens

    def _create_patches(
        self,
        x: torch.Tensor,
        feats_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        B, T, F = x.shape
        patch_time, patch_freq = self.patch_size

        num_patches_time = (T + patch_time - 1) // patch_time
        num_patches_freq = (F + patch_freq - 1) // patch_freq
        num_patches = num_patches_time * num_patches_freq

        pad_time = num_patches_time * patch_time - T
        pad_freq = num_patches_freq * patch_freq - F
        if pad_time > 0 or pad_freq > 0:
            x = torch.nn.functional.pad(x, (0, pad_freq, 0, pad_time))

        x = x.view(B, num_patches_time, patch_time, num_patches_freq, patch_freq)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, num_patches, patch_time, patch_freq)
        patches = x.view(B, num_patches, patch_time * patch_freq)

        patch_mask = torch.ones(B, num_patches, dtype=torch.bool, device=x.device)
        for b in range(B):
            valid_patches_time = (feats_lens[b] + patch_time - 1) // patch_time
            if valid_patches_time < num_patches_time:
                invalid_start = valid_patches_time * num_patches_freq
                patch_mask[b, invalid_start:] = False

        return patches, patch_mask, (num_patches_time, num_patches_freq)

    def _unpatch(
        self,
        patches: torch.Tensor,
        original_shape: Tuple[int, int],
        patch_grid: Tuple[int, int],
    ) -> torch.Tensor:
        B, num_patches, patch_dim = patches.shape
        T, F = original_shape
        patch_time, patch_freq = self.patch_size
        num_patches_time, num_patches_freq = patch_grid

        patches = patches.view(B, num_patches_time, num_patches_freq, patch_time, patch_freq)
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()
        x = patches.view(B, num_patches_time * patch_time, num_patches_freq * patch_freq)

        if x.shape[1] > T:
            x = x[:, :T, :]
        if x.shape[2] > F:
            x = x[:, :, :F]

        return x

    def _random_mask_patches(
        self,
        num_patches: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_masked = int(num_patches * self.mask_ratio)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        for b in range(batch_size):
            indices = torch.randperm(num_patches, device=device)[:num_masked]
            mask[b, indices] = True
        return mask

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def update_target_encoder(self):
        with torch.no_grad():
            if self.encoder_type == "transformer":
                for target_param, context_param in zip(
                    self.target_encoder.parameters(),
                    self.context_encoder.parameters(),
                ):
                    target_param.data.mul_(self.ema_decay).add_(
                        context_param.data, alpha=1 - self.ema_decay
                    )
                for target_param, context_param in zip(
                    self.target_patch_embed.parameters(),
                    self.context_patch_embed.parameters(),
                ):
                    target_param.data.mul_(self.ema_decay).add_(
                        context_param.data, alpha=1 - self.ema_decay
                    )
                if self.target_pos_enc is not None and self.context_pos_enc is not None:
                    for target_param, context_param in zip(
                        self.target_pos_enc.parameters(),
                        self.context_pos_enc.parameters(),
                    ):
                        target_param.data.mul_(self.ema_decay).add_(
                            context_param.data, alpha=1 - self.ema_decay
                        )
            else:
                for target_param, context_param in zip(
                    self.target_encoder.parameters(),
                    self.context_encoder.parameters(),
                ):
                    target_param.data.mul_(self.ema_decay).add_(
                        context_param.data, alpha=1 - self.ema_decay
                    )

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        clean_input: Optional[torch.Tensor] = None,
        clean_input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_n, feats_lens = self._extract_logmel(input, input_lengths)
        B, T, F = x_n.shape
        original_shape = (T, F)

        noisy_patches, patch_mask, patch_grid = self._create_patches(x_n, feats_lens)
        num_patches = noisy_patches.shape[1]

        if self.training:
            mask = self._random_mask_patches(num_patches, B, x_n.device)
            mask = mask & patch_mask
        else:
            mask = torch.zeros(B, num_patches, dtype=torch.bool, device=x_n.device)

        if self.encoder_type == "transformer":
            noisy_patches_embedded = self.context_patch_embed(noisy_patches)
            if mask.any():
                mask_token_expanded = self.mask_token.expand(B, num_patches, -1)
                mask_3d = mask.unsqueeze(-1)
                noisy_patches_embedded = torch.where(
                    mask_3d, mask_token_expanded, noisy_patches_embedded
                )
        else:
            noisy_patches_masked = noisy_patches.clone()
            if mask.any():
                mask_token_expanded = self.mask_token.expand(B, num_patches, -1)
                mask_3d = mask.unsqueeze(-1)
                noisy_patches_masked = torch.where(
                    mask_3d, mask_token_expanded, noisy_patches_masked
                )

        if self.encoder_type == "transformer":
            context_features = noisy_patches_embedded
            if self.context_pos_enc is not None:
                context_features = self.context_pos_enc(context_features)
            valid_mask = patch_mask.unsqueeze(1)
            for encoder_layer in self.context_encoder["encoders"]:
                context_features, valid_mask = encoder_layer(context_features, valid_mask)
            context_features = self.context_encoder["after_norm"](context_features)
        else:
            context_features = self.context_encoder(noisy_patches_masked)

        # Always run predictor/decoder for all patches.
        # This makes inference produce denoised-like reconstructed features,
        # instead of bypassing to raw noisy mel when mask is empty.
        if self.predictor_type == "transformer":
            pred_features = context_features
            if self.predictor_pos_enc is not None:
                pred_features = self.predictor_pos_enc(pred_features)
            valid_mask = patch_mask.unsqueeze(1)
            for encoder_layer in self.predictor["encoders"]:
                pred_features, valid_mask = encoder_layer(pred_features, valid_mask)
            predicted_latents = self.predictor["after_norm"](pred_features)
        else:
            predicted_latents = self.predictor(context_features)
        decoded_patches = self.decoder(predicted_latents)

        if mask.any() and self.training and clean_input is not None:
            lens = clean_input_lengths if clean_input_lengths is not None else input_lengths
            x_c, _ = self._extract_logmel(clean_input, lens)
            clean_patches, _, _ = self._create_patches(x_c, lens)

            full_reconstructed_patches = noisy_patches.clone()
            full_reconstructed_patches[mask] = decoded_patches[mask]
            x_hat = self._unpatch(full_reconstructed_patches, original_shape, patch_grid)

            self._last_x_n = x_n
            self._last_x_c = x_c
            self._last_x_hat = x_hat
            self._last_mask = mask
            self._last_patch_grid = patch_grid
            self._last_noisy_patches = noisy_patches
            self._last_clean_patches = clean_patches
            self._last_decoded_patches = decoded_patches
            self._last_patch_mask = patch_mask
        else:
            # In eval, mask is empty by design. Reconstruct all patches so
            # frontend acts as a denoising mapper instead of passthrough.
            x_hat = self._unpatch(decoded_patches, original_shape, patch_grid)

            self._last_x_n = None
            self._last_x_c = None
            self._last_x_hat = None
            self._last_mask = None
            self._last_patch_grid = None
            self._last_noisy_patches = None
            self._last_clean_patches = None
            self._last_decoded_patches = None
            self._last_patch_mask = None

        return x_hat, feats_lens

    def compute_jepa_loss(self) -> Optional[torch.Tensor]:
        """L_mask + λ*L_unmask on mel reconstruction (fixed target: clean mel)."""
        if (
            self._last_decoded_patches is None
            or self._last_clean_patches is None
            or self._last_mask is None
        ):
            return None

        if not self._last_patch_mask.any():
            return None

        valid_mask = self._last_patch_mask

        masked_valid = self._last_mask & valid_mask
        if masked_valid.any():
            masked_decoded = self._last_decoded_patches[masked_valid]
            masked_clean = self._last_clean_patches[masked_valid]
            loss_mask = F.mse_loss(masked_decoded, masked_clean, reduction="mean")
        else:
            loss_mask = torch.tensor(0.0, device=self._last_decoded_patches.device)

        unmasked_valid = (~self._last_mask) & valid_mask
        if unmasked_valid.any() and self.unmask_loss_weight > 0:
            unmasked_decoded = self._last_decoded_patches[unmasked_valid]
            unmasked_clean = self._last_clean_patches[unmasked_valid]
            loss_unmask = F.mse_loss(unmasked_decoded, unmasked_clean, reduction="mean")
        else:
            loss_unmask = torch.tensor(0.0, device=self._last_decoded_patches.device)

        total_loss = (
            loss_mask * self.embedding_loss_weight
            + loss_unmask * self.unmask_loss_weight
        )
        return total_loss

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ):
        mask_token_key = prefix + "mask_token"
        if mask_token_key in missing_keys:
            missing_keys.remove(mask_token_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
