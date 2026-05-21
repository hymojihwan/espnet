from typing import Any, Dict, Optional, Tuple

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
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class JEPA_MaskedPatchLatentFrontend(AbsFrontend):
    """JEPA Masked-Patch Latent Frontend for ASR.

    This frontend implements a JEPA-style self-supervised learning approach
    with clean reconstruction loss and alignment loss.

    Architecture:
        Noisy Speech → STFT → Log-Mel (x_n) → Mask patches
                                                      ↓
        All patches → Context Encoder → Context Features
                                                      ↓
                                                 Predictor
                                                      ↓
        Clean Speech → STFT → Log-Mel (x_c) → Target Encoder (EMA) → Target Latents
                                                      ↓
                                                 Target Decoder
                                                      ↓
        Loss: L_align = MSE(pred_latents[masked], target_latents[masked])
              L_clean_recon = MSE(clean_patches_hat, clean_patches)
    """

    def __init__(
        self,
        output_dim: int = 256,  # Output dimension of the frontend (ASR encoder input dim)
        embedding_dim: int = 256,
        context_encoder_dim: int = 256,
        target_encoder_dim: int = 256,
        predictor_dim: int = 256,
        decoder_dim: int = 256,
        num_context_encoder_layers: int = 2,
        num_target_encoder_layers: int = 2,
        num_predictor_layers: int = 2,
        num_decoder_layers: int = 2,
        patch_size: Any = (4, 4),  # (time, freq) - can accept list/tuple from YAML
        mask_ratio: float = 0.3,
        dropout_rate: float = 0.1,
        encoder_type: str = "transformer",  # "mlp" or "transformer"
        encoder_conf: Optional[Dict[str, Any]] = None,
        predictor_type: str = "transformer",  # "mlp" or "transformer"
        predictor_conf: Optional[Dict[str, Any]] = None,
        decoder_type: str = "transformer",  # "mlp" or "transformer"
        decoder_conf: Optional[Dict[str, Any]] = None,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
        fs: int = 16000,
        n_mels: int = 80,
        embedding_loss_weight: float = 1.0,
        ema_decay: float = 0.999,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        # Convert patch_size to tuple if it's a list (from YAML config)
        if isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        
        super().__init__()

        # Output dimension is n_mels (mel spectrogram, same as masked)
        self.output_dim = n_mels
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
        self.ema_decay = ema_decay
        self.encoder_type = encoder_type
        self.encoder_conf = encoder_conf or {}
        self.predictor_type = predictor_type
        self.predictor_conf = predictor_conf or {}
        self.decoder_type = decoder_type
        self.decoder_conf = decoder_conf or {}

        # STFT + LogMel
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

        # Context Encoder: Processes all patches (masked with mask token)
        patch_dim = patch_size[0] * patch_size[1]  # patch_time * patch_freq
        
        if encoder_type == "transformer":
            # Transformer encoder
            encoder_num_layers = self.encoder_conf.get("num_layers", num_context_encoder_layers)
            encoder_num_heads = self.encoder_conf.get("num_heads", 4)
            encoder_ff_dim = self.encoder_conf.get("ff_dim", 1024)
            encoder_dropout = self.encoder_conf.get("dropout_rate", dropout_rate)
            encoder_attn_dropout = self.encoder_conf.get("attn_dropout_rate", 0.1)
            use_pos_enc = self.encoder_conf.get("use_positional_encoding", False)  # Use fixed 2D sincos instead
            
            # Patch embedding: project patch_dim to embedding_dim
            self.context_patch_embed = nn.Linear(patch_dim, embedding_dim)
            
            # Positional encoding (optional, we use fixed 2D sincos in forward)
            if use_pos_enc:
                self.context_pos_enc = PositionalEncoding(embedding_dim, encoder_dropout)
            else:
                self.context_pos_enc = None
            
            # Transformer encoder layers
            self.context_encoder = self._build_transformer_encoder(
                embedding_dim, encoder_num_layers, encoder_num_heads,
                encoder_ff_dim, encoder_dropout, encoder_attn_dropout
            )
            
            # Target encoder (same structure, will be updated via EMA)
            self.target_patch_embed = nn.Linear(patch_dim, embedding_dim)
            if use_pos_enc:
                self.target_pos_enc = PositionalEncoding(embedding_dim, encoder_dropout)
            else:
                self.target_pos_enc = None
            self.target_encoder = self._build_transformer_encoder(
                embedding_dim, encoder_num_layers, encoder_num_heads,
                encoder_ff_dim, encoder_dropout, encoder_attn_dropout
            )
        else:
            # MLP encoder (not typically used, but kept for compatibility)
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
            
            # Target Encoder: Processes clean patches (with EMA)
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

        # Predictor: Predicts target latents from context features
        if predictor_type == "transformer":
            # Transformer predictor
            pred_num_layers = self.predictor_conf.get("num_layers", num_predictor_layers)
            pred_num_heads = self.predictor_conf.get("num_heads", 4)
            pred_ff_dim = self.predictor_conf.get("ff_dim", 1024)
            pred_dropout = self.predictor_conf.get("dropout_rate", dropout_rate)
            pred_attn_dropout = self.predictor_conf.get("attn_dropout_rate", 0.1)
            use_pos_enc = self.predictor_conf.get("use_positional_encoding", False)  # Use fixed 2D sincos instead
            
            # Positional encoding (optional)
            if use_pos_enc:
                self.predictor_pos_enc = PositionalEncoding(embedding_dim, pred_dropout)
            else:
                self.predictor_pos_enc = None
            
            # Transformer encoder layers
            self.predictor = self._build_transformer_encoder(
                embedding_dim, pred_num_layers, pred_num_heads,
                pred_ff_dim, pred_dropout, pred_attn_dropout
            )
        elif predictor_type == "mlp":
            # MLP predictor
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

        # Decoder: Reconstructs mel patches from latents (for noisy path output)
        if decoder_type == "mlp":
            # MLP decoder
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

        # Target Decoder: Reconstructs clean patches from target latents (for clean reconstruction loss)
        # Same structure as decoder but separate instance
        if decoder_type == "mlp":
            target_decoder_layers = []
            input_dim = embedding_dim
            for i in range(dec_num_layers):
                target_decoder_layers.extend([
                    nn.Linear(input_dim, dec_hidden_dim),
                    nn.LayerNorm(dec_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dec_dropout),
                ])
                input_dim = dec_hidden_dim
            target_decoder_layers.append(nn.Linear(input_dim, patch_dim))
            self.target_decoder = nn.Sequential(*target_decoder_layers)
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

        # Mask token for replacing masked patches (learnable)
        # For transformer: mask token is in embedding space
        # For MLP: mask token is in patch space
        if encoder_type == "transformer":
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_dim))

        # No target_decoder_head or out_proj needed - we output mel spectrogram like jepa_masked

        # init target = context
        self._init_teacher()

        # buffers for loss
        self._last_mask = None
        self._last_patch_mask = None
        self._last_pred_latents = None
        self._last_tgt_latents = None
        self._last_clean_patches = None
        self._last_clean_patches_hat = None

    def _build_transformer_encoder(
        self,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float,
        attn_dropout_rate: float,
    ) -> nn.Module:
        """Build transformer encoder for patch sequences."""
        # Build encoder layers
        encoder_layers = repeat(
            num_layers,
            lambda _: EncoderLayer(
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
        
        # Final layer norm
        after_norm = LayerNorm(embedding_dim)
        
        return nn.ModuleDict({
            "encoders": encoder_layers,
            "after_norm": after_norm,
        })

    def output_size(self) -> int:
        return self.output_dim

    def _extract_logmel(self, wav: torch.Tensor, wav_lens: torch.Tensor):
        """Initialize target encoder as copy of context encoder (will be updated via EMA)."""
        if self.encoder_type == "transformer":
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
            self.target_patch_embed.load_state_dict(self.context_patch_embed.state_dict())
            if self.target_pos_enc is not None and self.context_pos_enc is not None:
                self.target_pos_enc.load_state_dict(self.context_pos_enc.state_dict())
        else:
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        
        # Freeze target encoder (no gradients)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        if self.target_patch_embed is not None:
            for param in self.target_patch_embed.parameters():
                param.requires_grad = False
        if self.target_pos_enc is not None:
            for param in self.target_pos_enc.parameters():
                param.requires_grad = False
        # target_decoder is trainable (not frozen)

    def update_target_encoder(self):
        """Update target encoder using EMA of context encoder.
        
        This should be called after each optimizer step during training.
        """
        with torch.no_grad():
            if self.encoder_type == "transformer":
                # Update transformer encoder
                for target_param, context_param in zip(
                    self.target_encoder.parameters(),
                    self.context_encoder.parameters(),
                ):
                    target_param.data.mul_(self.ema_decay).add_(
                        context_param.data, alpha=1 - self.ema_decay
                    )
                # Update patch embedding
                for target_param, context_param in zip(
                    self.target_patch_embed.parameters(),
                    self.context_patch_embed.parameters(),
                ):
                    target_param.data.mul_(self.ema_decay).add_(
                        context_param.data, alpha=1 - self.ema_decay
                    )
                # Update positional encoding if exists
                if self.target_pos_enc is not None and self.context_pos_enc is not None:
                    for target_param, context_param in zip(
                        self.target_pos_enc.parameters(),
                        self.context_pos_enc.parameters(),
                    ):
                        target_param.data.mul_(self.ema_decay).add_(
                            context_param.data, alpha=1 - self.ema_decay
                        )
            else:
                # Update MLP encoder
                for target_param, context_param in zip(
                    self.target_encoder.parameters(),
                    self.context_encoder.parameters(),
                ):
                    target_param.data.mul_(self.ema_decay).add_(
                        context_param.data, alpha=1 - self.ema_decay
                    )
        # target_decoder is not updated via EMA (it's trainable)

    def _extract_logmel(self, wav: torch.Tensor, wav_lens: torch.Tensor):
        stft, feats_lens = self.stft(wav, wav_lens)
        if isinstance(stft, torch.Tensor):
            stft = ComplexTensor(stft[..., 0], stft[..., 1])
        power = stft.real ** 2 + stft.imag ** 2
        logmel, feats_lens = self.logmel(power, feats_lens)
        return logmel, feats_lens  # (B,T,F), (B,)

    def _create_patches(
        self,
        x: torch.Tensor,
        feats_lens: torch.Tensor,
    ):
        """
        2D patching (time,freq) -> flatten patches.
        Returns:
          patches: (B, num_patches, patch_time * patch_freq) - Flattened patches
          patch_mask: (B, num_patches) - Boolean mask indicating which patches are valid
          patch_grid: (num_patches_time, num_patches_freq) - Grid dimensions
          idx_t, idx_f: (num_patches,) long  (2D position index for each patch in flattened order)
        """
        B, T, F = x.shape
        patch_time, patch_freq = self.patch_size

        # Calculate number of patches
        num_patches_time = (T + patch_time - 1) // patch_time  # Ceiling division
        num_patches_freq = (F + patch_freq - 1) // patch_freq
        num_patches = num_patches_time * num_patches_freq

        # Pad if necessary
        pad_time = num_patches_time * patch_time - T
        pad_freq = num_patches_freq * patch_freq - F
        if pad_time > 0 or pad_freq > 0:
            x = torch.nn.functional.pad(x, (0, pad_freq, 0, pad_time))

        # Reshape to patches: (B, num_patches_time, patch_time, num_patches_freq, patch_freq)
        x = x.view(B, num_patches_time, patch_time, num_patches_freq, patch_freq)
        # Permute to group patches: (B, num_patches_time, num_patches_freq, patch_time, patch_freq)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        # Reshape to (B, num_patches, patch_time, patch_freq)
        x = x.view(B, num_patches, patch_time, patch_freq)
        # Flatten patches: (B, num_patches, patch_time * patch_freq)
        patches = x.view(B, num_patches, patch_time * patch_freq)

        # Create mask for valid patches based on sequence lengths
        patch_mask = torch.ones(B, num_patches, dtype=torch.bool, device=x.device)
        for b in range(B):
            valid_patches_time = (feats_lens[b] + patch_time - 1) // patch_time
            if valid_patches_time < num_patches_time:
                # Mark invalid patches (those beyond valid time steps)
                invalid_start = valid_patches_time * num_patches_freq
                patch_mask[b, invalid_start:] = False

        # 2D indices for each patch in flattened order (t-major then f)
        idx = torch.arange(num_patches, device=x.device)
        idx_t = idx // num_patches_freq
        idx_f = idx % num_patches_freq

        return patches, patch_mask, (num_patches_time, num_patches_freq), idx_t, idx_f

    def _unpatch(
        self,
        patches: torch.Tensor,
        original_shape: Tuple[int, int],
        patch_grid: Tuple[int, int],
    ) -> torch.Tensor:
        """Reconstruct spectrogram from patches.
        
        Args:
            patches: (B, num_patches, patch_time * patch_freq) - Flattened patches
            original_shape: (T, F) - Original shape (without batch dimension)
            patch_grid: (num_patches_time, num_patches_freq) - Grid dimensions
        
        Returns:
            x: (B, T, F) - Reconstructed spectrogram
        """
        B, num_patches, patch_dim = patches.shape
        T, F = original_shape
        patch_time, patch_freq = self.patch_size
        num_patches_time, num_patches_freq = patch_grid

        # Reshape patches to (B, num_patches_time, num_patches_freq, patch_time, patch_freq)
        patches = patches.view(B, num_patches_time, num_patches_freq, patch_time, patch_freq)
        # Permute to (B, num_patches_time, patch_time, num_patches_freq, patch_freq)
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()
        # Reshape to (B, num_patches_time * patch_time, num_patches_freq * patch_freq)
        x = patches.view(B, num_patches_time * patch_time, num_patches_freq * patch_freq)

        # Crop to original size
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
        """Create random mask for patches.

        Args:
            num_patches: Number of patches
            batch_size: Batch size
            device: Device

        Returns:
            mask: (B, num_patches) - Boolean mask (True for masked patches)
        """
        num_masked = int(num_patches * self.mask_ratio)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        for b in range(batch_size):
            # Randomly select patches to mask
            indices = torch.randperm(num_patches, device=device)[:num_masked]
            mask[b, indices] = True
        return mask

    def train(self, mode: bool = True):
        """Set training mode.
        
        Override to ensure target_encoder always stays in eval mode.
        """
        super().train(mode)
        self.target_encoder.eval()  # Always keep target encoder in eval mode
        return self

    def _init_teacher(self):
        """Initialize target encoder as copy of context encoder (will be updated via EMA)."""
        if self.encoder_type == "transformer":
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
            self.target_patch_embed.load_state_dict(self.context_patch_embed.state_dict())
            if self.target_pos_enc is not None and self.context_pos_enc is not None:
                self.target_pos_enc.load_state_dict(self.context_pos_enc.state_dict())
        else:
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        
        # Freeze target encoder (no gradients)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        if self.target_patch_embed is not None:
            for param in self.target_patch_embed.parameters():
                param.requires_grad = False
        if self.target_pos_enc is not None:
            for param in self.target_pos_enc.parameters():
                param.requires_grad = False
        # target_decoder is trainable (not frozen)

    def update_target_encoder(self):
        """Update target encoder using EMA of context encoder.
        
        This should be called after each optimizer step during training.
        """
        with torch.no_grad():
            if self.encoder_type == "transformer":
                # Update transformer encoder
                for target_param, context_param in zip(
                    self.target_encoder.parameters(),
                    self.context_encoder.parameters(),
                ):
                    target_param.data.mul_(self.ema_decay).add_(
                        context_param.data, alpha=1 - self.ema_decay
                    )
                # Update patch embedding
                for target_param, context_param in zip(
                    self.target_patch_embed.parameters(),
                    self.context_patch_embed.parameters(),
                ):
                    target_param.data.mul_(self.ema_decay).add_(
                        context_param.data, alpha=1 - self.ema_decay
                    )
                # Update positional encoding if exists
                if self.target_pos_enc is not None and self.context_pos_enc is not None:
                    for target_param, context_param in zip(
                        self.target_pos_enc.parameters(),
                        self.context_pos_enc.parameters(),
                    ):
                        target_param.data.mul_(self.ema_decay).add_(
                            context_param.data, alpha=1 - self.ema_decay
                        )
            else:
                # Update MLP encoder
                for target_param, context_param in zip(
                    self.target_encoder.parameters(),
                    self.context_encoder.parameters(),
                ):
                    target_param.data.mul_(self.ema_decay).add_(
                        context_param.data, alpha=1 - self.ema_decay
                    )
        # target_decoder is not updated via EMA (it's trainable)

    @staticmethod
    def _sincos_1d(pos: torch.Tensor, dim: int):
        """
        pos: (M,)
        return: (M, dim)
        """
        assert dim % 2 == 0
        omega = torch.arange(dim // 2, device=pos.device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (dim // 2)))
        out = pos.float().unsqueeze(1) * omega.unsqueeze(0)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    def _fixed_2d_sincos_pos(self, idx_t: torch.Tensor, idx_f: torch.Tensor, dim: int):
        """
        idx_t, idx_f: (N,)
        returns: (N, dim) fixed (no grad)
        """
        assert dim % 2 == 0
        d_half = dim // 2
        # split dim half for time, half for freq
        pe_t = self._sincos_1d(idx_t, d_half)
        pe_f = self._sincos_1d(idx_f, d_half)
        return torch.cat([pe_t, pe_f], dim=1).to(dtype=torch.float32)

    def _run_stack(self, stack: nn.ModuleDict, x: torch.Tensor, valid_mask_1: torch.Tensor):
        """
        x: (B,L,D)
        valid_mask_1: (B,1,L) bool
        """
        h = x
        m = valid_mask_1
        for layer in stack["encoders"]:
            h, m = layer(h, m)
        h = stack["after_norm"](h)
        return h

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor,
        clean_input: Optional[torch.Tensor] = None,
        clean_input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            input: (Batch, Nsamples) - Raw audio waveform (noisy speech)
            input_lengths: (Batch,) - Length of each sequence
            clean_input: Optional (Batch, Nsamples) - Clean speech for training
            clean_input_lengths: Optional (Batch,) - Length of clean sequences

        Returns:
            output: (Batch, T, output_dim) - Denoised log-mel features
            output_lengths: (Batch,) - Length of each sequence
        """
        # 1. Extract log-mel features from noisy input
        x_n, feats_lens = self._extract_logmel(input, input_lengths)
        B, T, F = x_n.shape
        original_shape = (T, F)

        # 2. Create patches from noisy log-mel
        noisy_patches, patch_mask, patch_grid, idx_t, idx_f = self._create_patches(x_n, feats_lens)
        num_patches = noisy_patches.shape[1]

        # 3. Create mask for patches (random masking during training)
        if self.training:
            mask = self._random_mask_patches(num_patches, B, x_n.device)
            # Combine with patch_mask (only mask valid patches)
            mask = mask & patch_mask
        else:
            # During inference, don't mask (or use a fixed pattern)
            mask = torch.zeros(B, num_patches, dtype=torch.bool, device=x_n.device)

        # 4. Replace masked patches with mask token before encoding
        if self.encoder_type == "transformer":
            # For transformer: embed all patches first
            noisy_patches_embedded = self.context_patch_embed(noisy_patches)  # (B, num_patches, embedding_dim)
            # Replace masked patches with mask token in embedding space
            if mask.any():
                mask_token_expanded = self.mask_token.expand(B, num_patches, -1)  # (B, num_patches, embedding_dim)
                mask_3d = mask.unsqueeze(-1)  # (B, num_patches, 1)
                noisy_patches_embedded = torch.where(
                    mask_3d, mask_token_expanded, noisy_patches_embedded
                )
            # Add fixed 2D sincos positional embedding
            pe = self._fixed_2d_sincos_pos(idx_t, idx_f, self.embedding_dim).to(
                device=noisy_patches_embedded.device, dtype=noisy_patches_embedded.dtype
            )  # (num_patches, embedding_dim)
            noisy_patches_embedded = noisy_patches_embedded + pe.unsqueeze(0)  # (B, num_patches, embedding_dim)
        else:
            # For MLP: replace masked patches with mask token in patch space
            noisy_patches_masked = noisy_patches.clone()
            if mask.any():
                mask_token_expanded = self.mask_token.expand(B, num_patches, -1)  # (B, num_patches, patch_dim)
                mask_3d = mask.unsqueeze(-1)  # (B, num_patches, 1)
                noisy_patches_masked = torch.where(
                    mask_3d, mask_token_expanded, noisy_patches_masked
                )
        
        # 5. Process all patches through context encoder
        if self.encoder_type == "transformer":
            # Use embedded patches with mask tokens
            context_features = noisy_patches_embedded  # (B, num_patches, embedding_dim)
            
            # Create mask for valid patches (for transformer attention)
            # mask shape: (B, 1, num_patches) - True for valid patches
            valid_mask = patch_mask.unsqueeze(1)  # (B, 1, num_patches)
            
            # Apply transformer encoder layers
            context_features = self._run_stack(self.context_encoder, context_features, valid_mask)
        else:
            # MLP encoder
            context_features = self.context_encoder(noisy_patches_masked)  # (B, num_patches, embedding_dim)
        
        if mask.any() and self.training and clean_input is not None:
            # Training mode: use target encoder for clean patches
            lens = clean_input_lengths if clean_input_lengths is not None else input_lengths
            x_c, _ = self._extract_logmel(clean_input, lens)
            clean_patches, _, _, _, _ = self._create_patches(x_c, lens)
            
            # Encode clean patches with target encoder (EMA)
            # Note: Clean patches are NOT masked - we encode all clean patches
            # The valid_mask is only for transformer attention to ignore padding
            with torch.no_grad():
                if self.encoder_type == "transformer":
                    # Embed patches
                    target_latents = self.target_patch_embed(clean_patches)  # (B, num_patches, embedding_dim)
                    
                    # Add fixed 2D sincos positional embedding
                    pe_c = self._fixed_2d_sincos_pos(idx_t, idx_f, self.embedding_dim).to(
                        device=target_latents.device, dtype=target_latents.dtype
                    )  # (num_patches, embedding_dim)
                    target_latents = target_latents + pe_c.unsqueeze(0)  # (B, num_patches, embedding_dim)
                    
                    # Create mask for valid patches (for transformer attention only)
                    # Use same patch_mask as noisy to ensure alignment
                    valid_mask = patch_mask.unsqueeze(1)  # (B, 1, num_patches)
                    
                    # Apply transformer encoder layers
                    target_latents = self._run_stack(self.target_encoder, target_latents, valid_mask)
                else:
                    target_latents = self.target_encoder(clean_patches)  # (B, num_patches, embedding_dim)
            
            # Predict target latents for masked regions
            if self.predictor_type == "transformer":
                # Transformer predictor: process all context features
                pred_features = context_features  # (B, num_patches, embedding_dim)
                
                # Create mask for valid patches
                valid_mask = patch_mask.unsqueeze(1)  # (B, 1, num_patches)
                
                # Apply transformer encoder layers
                predicted_latents = self._run_stack(self.predictor, pred_features, valid_mask)  # (B, num_patches, embedding_dim)
            else:
                # MLP predictor
                predicted_latents = self.predictor(context_features)  # (B, num_patches, embedding_dim)
            
            # Select masked patches
            masked_predicted_latents = predicted_latents[mask.unsqueeze(-1).expand_as(predicted_latents)].view(-1, self.embedding_dim)
            
            # Decode masked patches
            reconstructed_patches = self.decoder(masked_predicted_latents)  # (num_masked, patch_dim)
            
            # Merge: unmasked from x_n, masked from reconstruction
            # Reconstruct full patch structure with masked patches replaced
            full_reconstructed_patches = noisy_patches.clone()
            # Replace masked patches with reconstructed ones
            full_reconstructed_patches[mask] = reconstructed_patches
            
            # Reconstruct spectrogram from patches
            x_hat = self._unpatch(full_reconstructed_patches, original_shape, patch_grid)
            
            # Target decoder: reconstruct clean patches from target latents
            # (target_decoder is trainable, but target_latents are stopgrad)
            clean_patches_hat = self.target_decoder(target_latents.detach())  # (B, num_patches, patch_dim)
            
            # Store for loss computation
            self._last_mask = mask
            self._last_patch_mask = patch_mask
            self._last_pred_latents = predicted_latents
            self._last_tgt_latents = target_latents
            self._last_clean_patches = clean_patches
            self._last_clean_patches_hat = clean_patches_hat
        else:
            # Inference mode: predict and decode masked patches
            if mask.any():
                if self.predictor_type == "transformer":
                    # Transformer predictor: process all context features
                    pred_features = context_features  # (B, num_patches, embedding_dim)
                    
                    # Create mask for valid patches
                    valid_mask = patch_mask.unsqueeze(1)  # (B, 1, num_patches)
                    
                    # Apply transformer encoder layers
                    predicted_latents = self._run_stack(self.predictor, pred_features, valid_mask)  # (B, num_patches, embedding_dim)
                    
                    # Select masked patches
                    masked_predicted_latents = predicted_latents[mask.unsqueeze(-1).expand_as(predicted_latents)].view(-1, self.embedding_dim)
                else:
                    # MLP predictor: process only masked patches
                    masked_context_features = context_features[mask.unsqueeze(-1).expand_as(context_features)].view(-1, self.embedding_dim)
                    masked_predicted_latents = self.predictor(masked_context_features)
                
                reconstructed_patches = self.decoder(masked_predicted_latents)
                
                # Merge patches back into spectrogram
                full_reconstructed_patches = noisy_patches.clone()
                full_reconstructed_patches[mask] = reconstructed_patches
                x_hat = self._unpatch(full_reconstructed_patches, original_shape, patch_grid)
            else:
                x_hat = x_n
            
            self._clear_cache()

        return x_hat, feats_lens

    def _clear_cache(self):
        self._last_mask = None
        self._last_patch_mask = None
        self._last_pred_latents = None
        self._last_tgt_latents = None
        self._last_clean_patches = None
        self._last_clean_patches_hat = None

    def compute_jepa_loss(self) -> Optional[torch.Tensor]:
        """
        Computes two losses:
        1. L_align = MSE(pred_latents[masked], stopgrad(target_latents[masked]))
        2. L_clean_recon = MSE(clean_patches_hat, clean_patches) (masked-only or all)
        """
        if (
            self._last_mask is None
            or self._last_patch_mask is None
            or self._last_pred_latents is None
            or self._last_tgt_latents is None
            or self._last_clean_patches is None
            or self._last_clean_patches_hat is None
        ):
            return None

        if not self._last_patch_mask.any():
            return None

        masked_valid = self._last_mask & self._last_patch_mask  # (B,N)
        
        # 1. Alignment loss: L_align = MSE(pred_latents[masked], stopgrad(target_latents[masked]))
        if masked_valid.any():
            pred_masked = self._last_pred_latents[masked_valid]  # (M,D)
            tgt_masked = self._last_tgt_latents[masked_valid].detach()  # (M,D) - stopgrad
            loss_align = F.mse_loss(pred_masked, tgt_masked, reduction="mean")
        else:
            loss_align = torch.tensor(0.0, device=self._last_pred_latents.device)
        
        # 2. Clean reconstruction loss: L_clean_recon = MSE(clean_patches_hat, clean_patches)
        # Use masked-only for now (can be changed to all patches)
        if masked_valid.any():
            clean_hat_masked = self._last_clean_patches_hat[masked_valid]  # (M, patch_dim)
            clean_masked = self._last_clean_patches[masked_valid]  # (M, patch_dim)
            loss_clean_recon = F.mse_loss(clean_hat_masked, clean_masked, reduction="mean")
        else:
            loss_clean_recon = torch.tensor(0.0, device=self._last_clean_patches_hat.device)
        
        total_loss = loss_align + loss_clean_recon
        return total_loss * self.embedding_loss_weight


