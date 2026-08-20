
"""JEPA Masked-Patch Inpainting Frontend for ASR.

This module implements a JEPA-style masked-patch inpainting denoiser that
predicts masked regions of log-mel spectrograms.
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

class JEPA_MaskedPatchFrontend(AbsFrontend):
    """JEPA Masked-Patch Inpainting Frontend for ASR.

    This frontend implements a JEPA-style masked-patch inpainting denoiser
    that predicts masked regions of log-mel spectrograms.

    Architecture:
        Noisy Speech → STFT → Log-Mel (x_n) → Mask patches
                                                      ↓
        Unmasked patches → Context Encoder → Context Features
                                                      ↓
                                                 Predictor
                                                      ↓
        Clean Speech → STFT → Log-Mel (x_c) → Target Encoder (EMA) → Target Latents
                                                      ↓
                                                 (for masked regions)
                                                      ↓
                                                 Decoder → Reconstructed patches
                                                      ↓
        Merge: unmasked from x_n, masked from reconstruction → x_hat
                                                      ↓
        Feed x_hat to Conformer-CTC

    During training:
        - Mask patches on noisy log-mel (x_n)
        - Context encoder processes unmasked noisy patches
        - Target encoder (EMA) processes clean log-mel (x_c)
        - Predictor predicts target latents for masked regions
        - Decoder reconstructs mel patches from latents
        - Loss: Cosine distance between predicted and clean target latents

    During inference:
        - Only noisy speech is used
        - Masking, target encoder, and latent prediction loss are disabled
        - The configured ASR input path is fed to the encoder

    Args:
        output_dim: Output dimension of the frontend (should match encoder input, e.g., 80 for log-mel)
        embedding_dim: Dimension of the embedding/latent space (default: 256)
        context_encoder_dim: Dimension of the context encoder hidden layers (default: 256)
        target_encoder_dim: Dimension of the target encoder hidden layers (default: 256)
        predictor_dim: Dimension of the predictor hidden layers (default: 256)
        decoder_dim: Dimension of the decoder hidden layers (default: 256)
        num_context_encoder_layers: Number of context encoder layers (default: 2)
        num_target_encoder_layers: Number of target encoder layers (default: 2)
        num_predictor_layers: Number of predictor layers (default: 2)
        num_decoder_layers: Number of decoder layers (default: 2)
        patch_size: Size of patches in (time, freq) dimensions (default: (4, 4))
        mask_ratio: Ratio of patches to mask (default: 0.3)
        random_mask_ratio: Randomly sample the training mask ratio between
            mask_ratio_min and mask_ratio (default: False)
        mask_ratio_min: Minimum mask ratio used when random_mask_ratio is true
            (default: 0.0)
        zero_mask_prob: Probability of forcing mask_ratio=0.0 during training
            (default: 0.0)
        dropout_rate: Dropout rate (default: 0.1)
        n_fft: FFT size for STFT (default: 512)
        hop_length: Hop length for STFT (default: 128)
        win_length: Window length for STFT (default: None, defaults to n_fft)
        fs: Sampling rate (default: 16000)
        n_mels: Number of mel bins (default: 80)
        embedding_loss_weight: Weight for the reconstruction loss (default: 1.0)
        ema_decay: EMA decay rate for target encoder updates (default: 0.999)
        window: Window function for STFT (default: "hann")
        center: Whether to center STFT frames (default: True)
        normalized: Whether to normalize STFT (default: False)
        onesided: Whether to return onesided STFT (default: True)
    """

    def __init__(
        self,
        output_dim: int = 80,  # Output dimension of the frontend (log-mel dim)
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
        random_mask_ratio: bool = False,
        mask_ratio_min: float = 0.0,
        zero_mask_prob: float = 0.0,
        dropout_rate: float = 0.1,
        encoder_type: str = "mlp",  # "mlp" or "transformer"
        encoder_conf: Optional[Dict[str, Any]] = None,
        predictor_type: str = "mlp",  # "mlp", "transformer", or "identity"
        predictor_conf: Optional[Dict[str, Any]] = None,
        decoder_type: str = "mlp",  # "mlp" or "transformer"
        decoder_conf: Optional[Dict[str, Any]] = None,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
        fs: int = 16000,
        n_mels: int = 80,
        embedding_loss_weight: float = 1.0,
        base_reconstruction_loss_weight: float = 0.0,
        enable_jepa_loss: bool = True,
        latent_loss_type: str = "cosine",
        latent_prediction_target: str = "clean",
        mel_reconstruction_target: str = "clean",
        soft_clean_beta: float = 0.25,
        mel_reconstruction_loss_type: str = "mse",
        clean_shape_delta_weight: float = 0.5,
        clean_shape_base_anchor_weight: float = 0.05,
        clean_shape_huber_beta: float = 0.5,
        clean_shape_normalize_eps: float = 1.0e-5,
        ema_decay: float = 0.999,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        whisper_compatible_mel: bool = False,
        inference_full_reconstruction: bool = False,
        asr_input_mode: str = "masked",
        bridge_residual_weight: float = 0.1,
        full_bridge_loss_weight: float = 0.0,
    ):
        # Convert patch_size to tuple if it's a list (from YAML config)
        # This must be done before type checking
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
        self.patch_size: Tuple[int, int] = patch_size  # Now guaranteed to be tuple
        self.mask_ratio = mask_ratio
        self.random_mask_ratio = random_mask_ratio
        self.mask_ratio_min = mask_ratio_min
        self.zero_mask_prob = zero_mask_prob
        self.dropout_rate = dropout_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.embedding_loss_weight = embedding_loss_weight
        self.base_reconstruction_loss_weight = base_reconstruction_loss_weight
        self.enable_jepa_loss = enable_jepa_loss
        if latent_loss_type not in ("cosine", "l2", "mel_reconstruction"):
            raise ValueError(
                "latent_loss_type must be 'cosine', 'l2', or "
                "'mel_reconstruction', "
                f"got {latent_loss_type}"
            )
        self.latent_loss_type = latent_loss_type
        if latent_prediction_target not in ("clean", "base"):
            raise ValueError(
                "latent_prediction_target must be 'clean' or 'base', "
                f"got {latent_prediction_target}"
            )
        self.latent_prediction_target = latent_prediction_target
        if mel_reconstruction_target not in (
            "clean",
            "base",
            "input",
            "soft_clean",
        ):
            raise ValueError(
                "mel_reconstruction_target must be 'clean', 'base', 'input', "
                "or 'soft_clean', "
                f"got {mel_reconstruction_target}"
            )
        self.mel_reconstruction_target = mel_reconstruction_target
        if not 0.0 <= soft_clean_beta <= 1.0:
            raise ValueError(
                "soft_clean_beta must be between 0.0 and 1.0, "
                f"got {soft_clean_beta}"
            )
        self.soft_clean_beta = soft_clean_beta
        if mel_reconstruction_loss_type not in ("mse", "clean_shape_huber"):
            raise ValueError(
                "mel_reconstruction_loss_type must be 'mse' or "
                f"'clean_shape_huber', got {mel_reconstruction_loss_type}"
            )
        self.mel_reconstruction_loss_type = mel_reconstruction_loss_type
        self.clean_shape_delta_weight = clean_shape_delta_weight
        self.clean_shape_base_anchor_weight = clean_shape_base_anchor_weight
        self.clean_shape_huber_beta = clean_shape_huber_beta
        self.clean_shape_normalize_eps = clean_shape_normalize_eps
        if self.mel_reconstruction_loss_type == "clean_shape_huber":
            if self.mel_reconstruction_target != "clean":
                raise ValueError(
                    "clean_shape_huber requires mel_reconstruction_target='clean'"
                )
            if self.clean_shape_delta_weight < 0.0:
                raise ValueError("clean_shape_delta_weight must be non-negative")
            if self.clean_shape_base_anchor_weight < 0.0:
                raise ValueError(
                    "clean_shape_base_anchor_weight must be non-negative"
                )
            if self.clean_shape_huber_beta <= 0.0:
                raise ValueError("clean_shape_huber_beta must be positive")
            if self.clean_shape_normalize_eps <= 0.0:
                raise ValueError("clean_shape_normalize_eps must be positive")
        if self.base_reconstruction_loss_weight < 0.0:
            raise ValueError(
                "Expected base_reconstruction_loss_weight >= 0.0, "
                f"got {self.base_reconstruction_loss_weight}"
            )
        self.ema_decay = ema_decay
        self.encoder_type = encoder_type
        self.encoder_conf = encoder_conf or {}
        self.predictor_type = predictor_type
        self.predictor_conf = predictor_conf or {}
        self.decoder_type = decoder_type
        self.decoder_conf = decoder_conf or {}
        self.whisper_compatible_mel = whisper_compatible_mel
        self.inference_full_reconstruction = inference_full_reconstruction
        if asr_input_mode not in (
            "base",
            "masked",
            "zero_mask",
            "bridge",
            "residual_bridge",
            "teacher_bridge",
            "context_projection",
            "context_residual",
            "context_latent",
        ):
            raise ValueError(
                "asr_input_mode must be 'base', 'masked', 'zero_mask', 'bridge', "
                "'residual_bridge', 'teacher_bridge', or "
                "'context_projection', 'context_residual', or "
                "'context_latent', "
                f"got {asr_input_mode}"
            )
        self.asr_input_mode = asr_input_mode
        if self.asr_input_mode == "context_latent":
            self.output_dim = embedding_dim
        self.bridge_residual_weight = bridge_residual_weight
        self.full_bridge_loss_weight = full_bridge_loss_weight
        self.inference_mask_ratio = 0.0
        self.inference_residual_weight = 1.0
        self.inference_mask_seed = 0

        if self.full_bridge_loss_weight < 0.0:
            raise ValueError(
                "Expected full_bridge_loss_weight >= 0.0, "
                f"got {full_bridge_loss_weight}"
            )

        if not 0.0 <= self.mask_ratio_min <= self.mask_ratio <= 1.0:
            raise ValueError(
                "Expected 0.0 <= mask_ratio_min <= mask_ratio <= 1.0, "
                f"got mask_ratio_min={mask_ratio_min}, mask_ratio={mask_ratio}"
            )
        if not 0.0 <= self.zero_mask_prob <= 1.0:
            raise ValueError(
                "Expected 0.0 <= zero_mask_prob <= 1.0, "
                f"got {zero_mask_prob}"
            )

        if self.whisper_compatible_mel:
            try:
                import whisper
                from whisper.audio import HOP_LENGTH, N_FFT, N_MELS
            except Exception as e:
                raise RuntimeError(
                    "whisper_compatible_mel requires openai-whisper"
                ) from e

            expected = (N_FFT, N_FFT, HOP_LENGTH, N_MELS, 16000)
            configured = (n_fft, win_length, hop_length, n_mels, fs)
            if configured != expected:
                raise ValueError(
                    "Whisper-compatible mel requires "
                    f"n_fft={N_FFT}, win_length={N_FFT}, "
                    f"hop_length={HOP_LENGTH}, n_mels={N_MELS}, fs=16000; "
                    f"got {configured}"
                )
            self.register_buffer(
                "whisper_mel_filters",
                whisper.audio.mel_filters("cpu", n_mels),
                persistent=False,
            )
            self.stft = None
            self.logmel = None
            self.n_fft = n_fft
            self.win_length = win_length
        else:
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

        # Context Encoder: Processes unmasked noisy patches
        # Input: flattened patches (B, num_patches, patch_time * patch_freq)
        # Note: patch_freq is a subset of mel bins (n_mels), not all mel bins
        # Output: context features (B, num_patches, embedding_dim)
        patch_dim = patch_size[0] * patch_size[1]  # patch_time * patch_freq
        
        if encoder_type == "transformer":
            # Transformer encoder
            encoder_num_layers = self.encoder_conf.get("num_layers", 6)
            encoder_num_heads = self.encoder_conf.get("num_heads", 4)
            encoder_ff_dim = self.encoder_conf.get("ff_dim", 1024)
            encoder_dropout = self.encoder_conf.get("dropout_rate", dropout_rate)
            encoder_attn_dropout = self.encoder_conf.get("attn_dropout_rate", 0.1)
            use_pos_enc = self.encoder_conf.get("use_positional_encoding", True)
            
            # Patch embedding: project patch_dim to embedding_dim
            self.context_patch_embed = nn.Linear(patch_dim, embedding_dim)
            
            # Positional encoding
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
            # MLP encoder (original implementation)
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

        # Predictor: Predicts target latents for masked regions from context features
        # Input: context features (B, num_patches, embedding_dim)
        # Output: predicted target latents (B, num_patches, embedding_dim)
        if predictor_type == "transformer":
            # Transformer predictor
            pred_num_layers = self.predictor_conf.get("num_layers", num_predictor_layers)
            pred_num_heads = self.predictor_conf.get("num_heads", 4)
            pred_ff_dim = self.predictor_conf.get("ff_dim", 1024)
            pred_dropout = self.predictor_conf.get("dropout_rate", dropout_rate)
            pred_attn_dropout = self.predictor_conf.get("attn_dropout_rate", 0.1)
            use_pos_enc = self.predictor_conf.get("use_positional_encoding", True)
            
            # Positional encoding
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
            # Use config if provided, otherwise use default parameters
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
        elif predictor_type == "identity":
            self.predictor = nn.Identity()
            self.predictor_pos_enc = None
        else:
            raise ValueError(f"Unsupported predictor_type: {predictor_type}")

        # Context projection modes perform latent JEPA prediction and do not
        # reconstruct masked mel patches for the ASR path.
        self.decoder = None
        if self.asr_input_mode not in (
            "context_projection",
            "context_residual",
            "context_latent",
        ):
            if decoder_type == "mlp":
                dec_num_layers = self.decoder_conf.get(
                    "num_layers", num_decoder_layers
                )
                dec_hidden_dim = self.decoder_conf.get(
                    "hidden_dim", decoder_dim
                )
                dec_dropout = self.decoder_conf.get(
                    "dropout_rate", dropout_rate
                )

                decoder_layers = []
                input_dim = embedding_dim
                for i in range(dec_num_layers):
                    decoder_layers.extend(
                        [
                            nn.Linear(input_dim, dec_hidden_dim),
                            nn.LayerNorm(dec_hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dec_dropout),
                        ]
                    )
                    input_dim = dec_hidden_dim
                decoder_layers.append(nn.Linear(input_dim, patch_dim))
                self.decoder = nn.Sequential(*decoder_layers)
            else:
                raise ValueError(f"Unsupported decoder_type: {decoder_type}")

        self.asr_projector = None
        if self.asr_input_mode in ("context_projection", "context_residual"):
            self.asr_projector = nn.Linear(embedding_dim, patch_dim)
            if self.asr_input_mode == "context_residual":
                nn.init.zeros_(self.asr_projector.weight)
                nn.init.zeros_(self.asr_projector.bias)

        # Mask token for replacing masked patches (learnable)
        # For transformer: mask token is in embedding space
        # For MLP: mask token is in patch space
        if encoder_type == "transformer":
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_dim))

        # Initialize target encoder as copy of context encoder (will be updated via EMA)
        if encoder_type == "transformer":
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

        # Storage for loss computation
        self._last_x_n = None
        self._last_x_c = None
        self._last_x_hat = None
        self._last_mask = None
        self._last_patch_grid = None
        self._last_noisy_patches = None
        self._last_asr_patches = None
        self._last_clean_patches = None
        self._last_reconstruction_target_patches = None
        self._last_reconstructed_patches = None
        self._last_full_reconstructed_patches = None
        self._last_patch_mask = None
        self._last_predicted_latents = None
        self._last_target_latents = None
        self._last_jepa_loss_stats = {}

    def _build_transformer_encoder(
        self,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float,
        attn_dropout_rate: float,
    ) -> nn.Module:
        """Build transformer encoder for patch sequences.
        
        Args:
            embedding_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout_rate: Dropout rate
            attn_dropout_rate: Attention dropout rate
            
        Returns:
            Transformer encoder module
        """
        # Build encoder layers
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
        
        # Final layer norm
        after_norm = LayerNorm(embedding_dim)
        
        return nn.ModuleDict({
            "encoders": encoder_layers,
            "after_norm": after_norm,
        })

    def output_size(self) -> int:
        """Return the output dimension of the frontend."""
        return self.output_dim

    def _extract_logmel(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract log-mel features from audio waveform."""
        if self.whisper_compatible_mel:
            window = torch.hann_window(
                self.win_length, device=input.device, dtype=input.dtype
            )
            stft = torch.stft(
                input,
                self.n_fft,
                self.hop_length,
                window=window,
                return_complex=True,
            )
            magnitudes = stft[..., :-1].abs() ** 2
            mel_spec = self.whisper_mel_filters.to(magnitudes.dtype) @ magnitudes
            log_mel = torch.clamp(mel_spec, min=1e-10).log10()
            log_mel = torch.maximum(
                log_mel,
                log_mel.reshape(input.size(0), -1).max(dim=-1)[0][
                    :, None, None
                ]
                - 8.0,
            )
            log_mel = (log_mel + 4.0) / 4.0
            feats_lens = input_lengths // self.hop_length
            feats_lens = torch.clamp(feats_lens, max=log_mel.size(-1))
            return log_mel.transpose(1, 2), feats_lens

        input_stft, feats_lens = self.stft(input, input_lengths)
        if isinstance(input_stft, torch.Tensor):
            assert input_stft.shape[-1] == 2, f"Expected last dim to be 2, got {input_stft.shape}"
            input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        input_power = input_stft.real**2 + input_stft.imag**2
        log_mel, feats_lens = self.logmel(input_power, feats_lens)
        return log_mel, feats_lens

    def _create_patches(
        self,
        x: torch.Tensor,
        feats_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Create patches from log-mel spectrogram.

        Args:
            x: (B, T, F) - Log-mel spectrogram where F is mel bins
            feats_lens: (B,) - Length of each sequence

        Returns:
            patches: (B, num_patches, patch_time * patch_freq) - Flattened patches
            patch_mask: (B, num_patches) - Boolean mask indicating which patches are valid
            patch_grid: (num_patches_time, num_patches_freq) - Grid dimensions
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

        return patches, patch_mask, (num_patches_time, num_patches_freq)

    def _unpatch(
        self,
        patches: torch.Tensor,
        original_shape: Tuple[int, int, int],
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
        mask_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """Create random mask for patches.

        Args:
            num_patches: Number of patches
            batch_size: Batch size
            device: Device

        Returns:
            mask: (B, num_patches) - Boolean mask (True for masked patches)
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        num_masked = int(num_patches * mask_ratio)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        if num_masked <= 0:
            return mask
        for b in range(batch_size):
            # Randomly select patches to mask
            indices = torch.randperm(num_patches, device=device)[:num_masked]
            mask[b, indices] = True
        return mask

    def configure_masked_residual_inference(
        self,
        mask_ratio: float,
        residual_weight: float,
        seed: int = 0,
    ) -> None:
        """Enable deterministic masked residual reconstruction at inference."""
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(
                f"Expected 0.0 <= mask_ratio <= 1.0, got {mask_ratio}"
            )
        if not 0.0 <= residual_weight <= 1.0:
            raise ValueError(
                "Expected 0.0 <= residual_weight <= 1.0, "
                f"got {residual_weight}"
            )
        self.inference_mask_ratio = mask_ratio
        self.inference_residual_weight = residual_weight
        self.inference_mask_seed = seed

    def _deterministic_inference_mask(
        self,
        num_patches: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a reproducible pseudo-random patch mask for inference."""
        num_masked = int(num_patches * self.inference_mask_ratio)
        mask = torch.zeros(
            batch_size,
            num_patches,
            dtype=torch.bool,
            device=device,
        )
        if num_masked <= 0:
            return mask

        generator = torch.Generator(device=device)
        generator.manual_seed(self.inference_mask_seed)
        for batch_index in range(batch_size):
            indices = torch.randperm(
                num_patches,
                generator=generator,
                device=device,
            )[:num_masked]
            mask[batch_index, indices] = True
        return mask

    def _sample_training_mask_ratio(self, device: torch.device) -> float:
        """Sample the mask ratio used for the current training batch."""
        sample = torch.rand((), device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(sample, src=0)

        sample_value = sample.item()
        if self.zero_mask_prob > 0.0:
            if sample_value < self.zero_mask_prob:
                return 0.0

            sample_value = (
                (sample_value - self.zero_mask_prob)
                / max(1.0 - self.zero_mask_prob, 1.0e-8)
            )

        if self.random_mask_ratio:
            return self.mask_ratio_min + sample_value * (
                self.mask_ratio - self.mask_ratio_min
            )

        return self.mask_ratio

    def _encode_context_patches(
        self,
        noisy_patches: torch.Tensor,
        patch_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode noisy patches, optionally replacing masked patches first."""
        B, num_patches, _ = noisy_patches.shape
        if self.encoder_type == "transformer":
            context_features = self.context_patch_embed(noisy_patches)
            if mask.any():
                mask_token_expanded = self.mask_token.expand(B, num_patches, -1)
                context_features = torch.where(
                    mask.unsqueeze(-1), mask_token_expanded, context_features
                )
            elif self.training:
                context_features = context_features + self.mask_token.sum() * 0.0

            if self.context_pos_enc is not None:
                context_features = self.context_pos_enc(context_features)

            valid_mask = patch_mask.unsqueeze(1)
            for encoder_layer in self.context_encoder["encoders"]:
                context_features, valid_mask = encoder_layer(
                    context_features, valid_mask
                )
            return self.context_encoder["after_norm"](context_features)

        noisy_patches_masked = noisy_patches
        if mask.any():
            mask_token_expanded = self.mask_token.expand(B, num_patches, -1)
            noisy_patches_masked = torch.where(
                mask.unsqueeze(-1), mask_token_expanded, noisy_patches
            )
        return self.context_encoder(noisy_patches_masked)

    def _predict_latents(
        self,
        context_features: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target latents from context features."""
        if self.predictor_type == "transformer":
            pred_features = context_features
            if self.predictor_pos_enc is not None:
                pred_features = self.predictor_pos_enc(pred_features)

            valid_mask = patch_mask.unsqueeze(1)
            for encoder_layer in self.predictor["encoders"]:
                pred_features, valid_mask = encoder_layer(pred_features, valid_mask)
            return self.predictor["after_norm"](pred_features)

        return self.predictor(context_features)

    def _encode_target_patches(
        self,
        patches: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode patches with the EMA target encoder."""
        with torch.no_grad():
            if self.encoder_type == "transformer":
                target_latents = self.target_patch_embed(patches)
                if self.target_pos_enc is not None:
                    target_latents = self.target_pos_enc(target_latents)

                valid_mask = patch_mask.unsqueeze(1)
                for encoder_layer in self.target_encoder["encoders"]:
                    target_latents, valid_mask = encoder_layer(
                        target_latents, valid_mask
                    )
                return self.target_encoder["after_norm"](target_latents)

            return self.target_encoder(patches)

    def _decode_teacher_bridge(
        self,
        noisy_patches: torch.Tensor,
        patch_mask: torch.Tensor,
        original_shape: Tuple[int, int],
        patch_grid: Tuple[int, int],
    ) -> torch.Tensor:
        """Decode enhanced patches through EMA target encoder for ASR input."""
        target_latents = self._encode_target_patches(noisy_patches, patch_mask)
        reconstructed_patches = self.decoder(
            target_latents.reshape(-1, self.embedding_dim)
        ).view_as(noisy_patches)
        full_reconstructed_patches = torch.where(
            patch_mask.unsqueeze(-1),
            reconstructed_patches.to(noisy_patches.dtype),
            noisy_patches,
        )
        return self._unpatch(full_reconstructed_patches, original_shape, patch_grid)

    def _decode_full_bridge(
        self,
        context_features: torch.Tensor,
        noisy_patches: torch.Tensor,
        patch_mask: torch.Tensor,
        original_shape: Tuple[int, int],
        patch_grid: Tuple[int, int],
    ) -> torch.Tensor:
        """Decode every valid patch for the ASR input bridge path."""
        predicted_latents = self._predict_latents(context_features, patch_mask)
        reconstructed_patches = self.decoder(
            predicted_latents.reshape(-1, self.embedding_dim)
        ).view_as(noisy_patches)
        full_reconstructed_patches = torch.where(
            patch_mask.unsqueeze(-1),
            reconstructed_patches.to(noisy_patches.dtype),
            noisy_patches,
        )
        return self._unpatch(full_reconstructed_patches, original_shape, patch_grid)

    def _decode_unmasked_bridge(
        self,
        context_features: torch.Tensor,
        noisy_patches: torch.Tensor,
        patch_mask: torch.Tensor,
        mask: torch.Tensor,
        original_shape: Tuple[int, int],
        patch_grid: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode all patches from an unmasked context path."""
        unmasked_context_features = context_features
        if mask.any():
            no_mask = torch.zeros_like(mask)
            unmasked_context_features = self._encode_context_patches(
                noisy_patches, patch_mask, no_mask
            )
        predicted_latents = self._predict_latents(
            unmasked_context_features, patch_mask
        )
        reconstructed_patches = self.decoder(
            predicted_latents.reshape(-1, self.embedding_dim)
        ).view_as(noisy_patches)
        full_reconstructed_patches = torch.where(
            patch_mask.unsqueeze(-1),
            reconstructed_patches.to(noisy_patches.dtype),
            noisy_patches,
        )
        bridge_x = self._unpatch(
            full_reconstructed_patches, original_shape, patch_grid
        )
        return bridge_x, reconstructed_patches

    def _project_unmasked_context(
        self,
        context_features: torch.Tensor,
        noisy_patches: torch.Tensor,
        patch_mask: torch.Tensor,
        mask: torch.Tensor,
        original_shape: Tuple[int, int],
        patch_grid: Tuple[int, int],
        residual: bool = False,
    ) -> torch.Tensor:
        """Project unmasked context latents to the ASR feature sequence."""
        if self.asr_projector is None:
            raise RuntimeError("ASR projector is not initialized")

        unmasked_context_features = self._get_unmasked_context(
            context_features,
            noisy_patches,
            patch_mask,
            mask,
        )
        projected_patches = self.asr_projector(unmasked_context_features)
        invalid_patches = (
            torch.zeros_like(noisy_patches) if residual else noisy_patches
        )
        projected_patches = torch.where(
            patch_mask.unsqueeze(-1),
            projected_patches.to(noisy_patches.dtype),
            invalid_patches,
        )
        return self._unpatch(projected_patches, original_shape, patch_grid)

    def _get_unmasked_context(
        self,
        context_features: torch.Tensor,
        noisy_patches: torch.Tensor,
        patch_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return context latents computed without patch masking."""
        if not mask.any():
            return context_features

        no_mask = torch.zeros_like(mask)
        return self._encode_context_patches(noisy_patches, patch_mask, no_mask)

    def train(self, mode: bool = True):
        """Set training mode.
        
        Override to ensure target_encoder always stays in eval mode.
        """
        super().train(mode)
        self.target_encoder.eval()  # Always keep target encoder in eval mode
        return self

    def update_target_encoder(self):
        """Update target encoder using EMA of context encoder.
        
        This should be called after each optimizer step during training.
        """
        if not self.enable_jepa_loss or self.latent_loss_type == "mel_reconstruction":
            return

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

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor,
        clean_input: Optional[torch.Tensor] = None,
        clean_input_lengths: Optional[torch.Tensor] = None,
        asr_input: Optional[torch.Tensor] = None,
        asr_input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            input: (Batch, Nsamples) - Raw audio waveform (noisy speech)
            input_lengths: (Batch,) - Length of each sequence
            clean_input: Optional (Batch, Nsamples) - Clean speech for training
            clean_input_lengths: Optional (Batch,) - Length of clean sequences
            asr_input: Optional (Batch, Nsamples) - Separate waveform used as
                the base ASR features while JEPA encodes ``input``
            asr_input_lengths: Optional (Batch,) - Lengths for ``asr_input``

        Returns:
            output: (Batch, T, output_dim) - Denoised log-mel features
            output_lengths: (Batch,) - Length of each sequence
        """
        # 1. Extract log-mel features from noisy input
        x_n, feats_lens = self._extract_logmel(input, input_lengths)
        B, T, F = x_n.shape
        original_shape = (T, F)

        # 2. Create patches from noisy log-mel
        noisy_patches, patch_mask, patch_grid = self._create_patches(x_n, feats_lens)
        num_patches = noisy_patches.shape[1]
        if asr_input is None:
            x_asr = x_n
            asr_patches = noisy_patches
        else:
            asr_lengths = (
                asr_input_lengths
                if asr_input_lengths is not None
                else input_lengths
            )
            x_asr, asr_feats_lens = self._extract_logmel(asr_input, asr_lengths)
            if x_asr.shape != x_n.shape or not torch.equal(asr_feats_lens, feats_lens):
                raise ValueError(
                    "JEPA and ASR-base features must have matching shapes and lengths: "
                    f"JEPA={tuple(x_n.shape)}, ASR={tuple(x_asr.shape)}"
                )
            asr_patches, asr_patch_mask, asr_patch_grid = self._create_patches(
                x_asr, asr_feats_lens
            )
            if (
                asr_patches.shape != noisy_patches.shape
                or not torch.equal(asr_patch_mask, patch_mask)
                or asr_patch_grid != patch_grid
            ):
                raise ValueError("JEPA and ASR-base patch layouts must match")

        # 3. Create mask for patches (random masking during training)
        if self.training:
            current_mask_ratio = self._sample_training_mask_ratio(x_n.device)
            mask = self._random_mask_patches(
                num_patches, B, x_n.device, current_mask_ratio
            )
            # Combine with patch_mask (only mask valid patches)
            mask = mask & patch_mask
        else:
            mask = self._deterministic_inference_mask(
                num_patches,
                B,
                x_n.device,
            )
            mask = mask & patch_mask

        # 4. Process patches through context encoder for the JEPA masked branch
        context_features = self._encode_context_patches(noisy_patches, patch_mask, mask)
        
        if mask.any() and self.training:
            x_c = None
            clean_patches = None
            reconstruction_target_patches = (
                asr_patches.detach().clone()
                if self.base_reconstruction_loss_weight > 0.0
                else None
            )
            target_latents = None
            if (
                self.enable_jepa_loss
                and self.latent_loss_type == "mel_reconstruction"
                and self.mel_reconstruction_target == "base"
            ):
                reconstruction_target_patches = asr_patches.detach().clone()
            elif (
                self.enable_jepa_loss
                and self.latent_loss_type == "mel_reconstruction"
                and self.mel_reconstruction_target == "input"
            ):
                reconstruction_target_patches = noisy_patches.detach().clone()
            elif (
                self.enable_jepa_loss
                and self.latent_loss_type != "mel_reconstruction"
                and self.latent_prediction_target == "base"
            ):
                target_latents = self._encode_target_patches(
                    asr_patches, patch_mask
                )
            elif self.enable_jepa_loss and clean_input is not None:
                lens = (
                    clean_input_lengths
                    if clean_input_lengths is not None
                    else input_lengths
                )
                x_c, _ = self._extract_logmel(clean_input, lens)
                clean_patches, _, _ = self._create_patches(x_c, lens)
                if self.latent_loss_type == "mel_reconstruction":
                    if self.mel_reconstruction_target == "soft_clean":
                        if clean_patches.shape != asr_patches.shape:
                            raise ValueError(
                                "Clean and base patches must match for soft-clean "
                                "target interpolation: "
                                f"clean={tuple(clean_patches.shape)}, "
                                f"base={tuple(asr_patches.shape)}"
                            )
                        base_target = asr_patches.detach().clone()
                        clean_target = clean_patches.detach()
                        reconstruction_target_patches = base_target + (
                            self.soft_clean_beta * (clean_target - base_target)
                        )
                    else:
                        reconstruction_target_patches = clean_patches.detach()
                else:
                    target_latents = self._encode_target_patches(
                        clean_patches, patch_mask
                    )
            
            # Predict target latents for masked regions
            predicted_latents = self._predict_latents(context_features, patch_mask)
            
            if self.asr_input_mode in (
                "context_projection",
                "context_residual",
                "context_latent",
            ):
                reconstructed_patches = None
                x_hat = x_asr
            elif self.asr_input_mode == "zero_mask":
                masked_predicted_latents = predicted_latents[
                    mask.unsqueeze(-1).expand_as(predicted_latents)
                ].view(-1, self.embedding_dim)
                reconstructed_patches = self.decoder(masked_predicted_latents)
                zero_masked_patches = asr_patches.masked_fill(
                    mask.unsqueeze(-1), 0.0
                )
                x_hat = self._unpatch(
                    zero_masked_patches, original_shape, patch_grid
                )
            else:
                masked_predicted_latents = predicted_latents[
                    mask.unsqueeze(-1).expand_as(predicted_latents)
                ].view(-1, self.embedding_dim)
                reconstructed_patches = self.decoder(masked_predicted_latents)
                if self.asr_input_mode == "base":
                    x_hat = x_asr
                else:
                    full_reconstructed_patches = asr_patches.clone()
                    full_reconstructed_patches[mask] = reconstructed_patches.to(
                        full_reconstructed_patches.dtype
                    )
                    x_hat = self._unpatch(
                        full_reconstructed_patches, original_shape, patch_grid
                    )
            
            # Store for loss computation
            self._last_x_n = x_n
            self._last_x_c = x_c
            self._last_x_hat = x_hat
            self._last_mask = mask
            self._last_patch_grid = patch_grid
            self._last_noisy_patches = noisy_patches
            self._last_asr_patches = asr_patches
            self._last_clean_patches = clean_patches
            self._last_reconstruction_target_patches = (
                reconstruction_target_patches
            )
            self._last_reconstructed_patches = reconstructed_patches
            self._last_patch_mask = patch_mask
            self._last_predicted_latents = predicted_latents
            self._last_target_latents = target_latents
        else:
            # Inference mode: predict and decode masked patches
            if self.inference_full_reconstruction:
                x_hat = self._decode_full_bridge(
                    context_features,
                    noisy_patches,
                    patch_mask,
                    original_shape,
                    patch_grid,
                )
            elif mask.any():
                predicted_latents = self._predict_latents(context_features, patch_mask)
                masked_predicted_latents = predicted_latents[
                    mask.unsqueeze(-1).expand_as(predicted_latents)
                ].view(-1, self.embedding_dim)
                
                reconstructed_patches = self.decoder(masked_predicted_latents)
                
                # Merge patches back into spectrogram (match dtype for AMP)
                full_reconstructed_patches = asr_patches.clone()
                base_masked_patches = full_reconstructed_patches[mask]
                reconstructed_patches = reconstructed_patches.to(
                    full_reconstructed_patches.dtype
                )
                residual_patches = base_masked_patches + (
                    self.inference_residual_weight
                    * (reconstructed_patches - base_masked_patches)
                )
                full_reconstructed_patches[mask] = residual_patches
                x_hat = self._unpatch(full_reconstructed_patches, original_shape, patch_grid)
            else:
                x_hat = x_asr
            
            self._last_x_n = None
            self._last_x_c = None
            self._last_x_hat = None
            self._last_mask = None
            self._last_patch_grid = None
            self._last_noisy_patches = None
            self._last_asr_patches = None
            self._last_clean_patches = None
            self._last_reconstruction_target_patches = None
            self._last_reconstructed_patches = None
            self._last_full_reconstructed_patches = None
            self._last_patch_mask = None
            self._last_predicted_latents = None
            self._last_target_latents = None
            self._last_jepa_loss_stats = {}

        output_lengths = feats_lens
        if self.asr_input_mode == "base":
            x_hat = x_asr.clone()
        elif self.asr_input_mode == "teacher_bridge" and not self.training:
            x_hat = self._decode_teacher_bridge(
                noisy_patches,
                patch_mask,
                original_shape,
                patch_grid,
            )
        elif self.asr_input_mode == "context_latent":
            x_hat = self._get_unmasked_context(
                context_features,
                noisy_patches,
                patch_mask,
                mask,
            )
            output_lengths = patch_mask.sum(dim=1).to(feats_lens.dtype)
        elif self.asr_input_mode == "context_projection":
            x_hat = self._project_unmasked_context(
                context_features,
                noisy_patches,
                patch_mask,
                mask,
                original_shape,
                patch_grid,
            )
        elif self.asr_input_mode == "context_residual":
            residual = self._project_unmasked_context(
                context_features,
                noisy_patches,
                patch_mask,
                mask,
                original_shape,
                patch_grid,
                residual=True,
            )
            x_hat = x_asr + self.bridge_residual_weight * residual
        elif self.asr_input_mode in ("bridge", "residual_bridge"):
            bridge_x, full_reconstructed_patches = self._decode_unmasked_bridge(
                context_features,
                noisy_patches,
                patch_mask,
                mask,
                original_shape,
                patch_grid,
            )
            if self.training and self._last_clean_patches is not None:
                self._last_full_reconstructed_patches = (
                    full_reconstructed_patches
                )
            if self.asr_input_mode == "bridge":
                x_hat = bridge_x
            else:
                x_hat = x_asr + self.bridge_residual_weight * (bridge_x - x_asr)

        return x_hat, output_lengths

    def compute_jepa_loss(self) -> Optional[torch.Tensor]:
        """Compute the configured masked prediction loss.
        
        Returns:
            loss: Scalar latent cosine loss, or None if unavailable
        """
        if not self.enable_jepa_loss:
            return None

        self._last_jepa_loss_stats = {}

        if self.latent_loss_type == "mel_reconstruction":
            mask = self._last_mask
            target_patches = self._last_reconstruction_target_patches
            reconstructed = self._last_reconstructed_patches
            if (
                mask is None
                or target_patches is None
                or reconstructed is None
                or mask.dim() != 2
                or target_patches.dim() != 3
                or reconstructed.dim() != 2
            ):
                return None
            n_patches = min(mask.size(1), target_patches.size(1))
            masked_target = target_patches[:, :n_patches, :][
                mask[:, :n_patches]
            ].detach()
            n_masked = min(reconstructed.size(0), masked_target.size(0))
            if n_masked == 0:
                return None
            reconstructed = reconstructed[:n_masked].float()
            masked_target = masked_target[:n_masked].float()
            if self.mel_reconstruction_loss_type == "clean_shape_huber":
                base_patches = self._last_asr_patches
                if base_patches is None or base_patches.dim() != 3:
                    return None
                masked_base = base_patches[:, :n_patches, :][
                    mask[:, :n_patches]
                ].detach()
                n_masked = min(n_masked, masked_base.size(0))
                if n_masked == 0:
                    return None
                return self.embedding_loss_weight * (
                    self._compute_clean_shape_reconstruction_loss(
                        reconstructed[:n_masked],
                        masked_target[:n_masked],
                        masked_base[:n_masked].float(),
                    )
                )

            reconstruction_loss = F.mse_loss(
                reconstructed,
                masked_target,
                reduction="mean",
            )
            self._last_jepa_loss_stats = {
                "loss_jepa_mel_reconstruction": reconstruction_loss.detach()
            }
            if self.mel_reconstruction_target == "soft_clean":
                self._last_jepa_loss_stats["soft_clean_beta"] = (
                    reconstruction_loss.new_tensor(self.soft_clean_beta)
                )
            return self.embedding_loss_weight * reconstruction_loss

        predicted_latents = self._last_predicted_latents
        target_latents = self._last_target_latents
        mask = self._last_mask
        if predicted_latents is None or target_latents is None or mask is None:
            return None

        n_patches = min(
            predicted_latents.size(1),
            target_latents.size(1),
            mask.size(1),
        )
        masked_valid = mask[:, :n_patches]
        patch_mask = self._last_patch_mask
        if patch_mask is not None:
            masked_valid = masked_valid & patch_mask[:, :n_patches]
        if not masked_valid.any():
            return None

        predicted_masked = predicted_latents[:, :n_patches, :][masked_valid]
        target_masked = target_latents[:, :n_patches, :][masked_valid].detach()
        predicted_masked = predicted_masked.float()
        target_masked = target_masked.float()
        if self.latent_loss_type == "cosine":
            cosine_similarity = F.cosine_similarity(
                predicted_masked,
                target_masked,
                dim=-1,
                eps=1.0e-8,
            )
            latent_loss = 1.0 - cosine_similarity.mean()
        else:
            latent_loss = F.mse_loss(
                predicted_masked,
                target_masked,
                reduction="mean",
            )
        loss = self.embedding_loss_weight * latent_loss

        full_reconstructed = self._last_full_reconstructed_patches
        clean_patches = self._last_clean_patches
        if (
            self.full_bridge_loss_weight > 0.0
            and full_reconstructed is not None
            and patch_mask is not None
            and clean_patches is not None
        ):
            n_patches = min(
                patch_mask.size(1),
                clean_patches.size(1),
                full_reconstructed.size(1),
            )
            valid_mask = patch_mask[:, :n_patches]
            if valid_mask.any():
                full_loss = self.full_bridge_loss_weight * F.mse_loss(
                    full_reconstructed[:, :n_patches, :][valid_mask],
                    clean_patches[:, :n_patches, :][valid_mask],
                    reduction="mean",
                )
                loss = full_loss if loss is None else loss + full_loss

        return loss

    def _compute_clean_shape_reconstruction_loss(
        self,
        reconstructed: torch.Tensor,
        clean_target: torch.Tensor,
        base_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clean Mel shape, temporal-delta, and Base-anchor losses."""
        patch_time, patch_frequency = self.patch_size
        expected_size = patch_time * patch_frequency
        if reconstructed.size(-1) != expected_size:
            raise ValueError(
                "Reconstructed patch dimension does not match patch_size: "
                f"{reconstructed.size(-1)} != {expected_size}"
            )

        reconstructed_2d = reconstructed.reshape(
            -1,
            patch_time,
            patch_frequency,
        )
        clean_target_2d = clean_target.reshape(
            -1,
            patch_time,
            patch_frequency,
        )
        reconstructed_normalized = self._normalize_mel_patch(
            reconstructed_2d
        )
        clean_target_normalized = self._normalize_mel_patch(clean_target_2d)

        shape_loss = F.smooth_l1_loss(
            reconstructed_normalized,
            clean_target_normalized,
            beta=self.clean_shape_huber_beta,
            reduction="mean",
        )
        if patch_time > 1 and self.clean_shape_delta_weight > 0.0:
            reconstructed_delta = (
                reconstructed_normalized[:, 1:]
                - reconstructed_normalized[:, :-1]
            )
            clean_target_delta = (
                clean_target_normalized[:, 1:]
                - clean_target_normalized[:, :-1]
            )
            delta_loss = F.smooth_l1_loss(
                reconstructed_delta,
                clean_target_delta,
                beta=self.clean_shape_huber_beta,
                reduction="mean",
            )
        else:
            delta_loss = shape_loss.new_zeros(())
        base_anchor_loss = F.smooth_l1_loss(
            reconstructed,
            base_target,
            beta=self.clean_shape_huber_beta,
            reduction="mean",
        )
        total_loss = (
            shape_loss
            + self.clean_shape_delta_weight * delta_loss
            + self.clean_shape_base_anchor_weight * base_anchor_loss
        )
        self._last_jepa_loss_stats = {
            "loss_jepa_clean_shape": shape_loss.detach(),
            "loss_jepa_clean_delta": delta_loss.detach(),
            "loss_jepa_base_anchor": base_anchor_loss.detach(),
        }
        return total_loss

    def _normalize_mel_patch(self, patch: torch.Tensor) -> torch.Tensor:
        mean = patch.mean(dim=(-2, -1), keepdim=True)
        variance = patch.var(
            dim=(-2, -1),
            correction=0,
            keepdim=True,
        )
        return (patch - mean) * torch.rsqrt(
            variance + self.clean_shape_normalize_eps
        )

    def get_jepa_loss_stats(self) -> Dict[str, torch.Tensor]:
        """Return detached component losses from the latest JEPA loss call."""
        return self._last_jepa_loss_stats.copy()

    def compute_base_reconstruction_loss(self) -> Optional[torch.Tensor]:
        """Compute Base Mel reconstruction loss for masked ASR patches."""
        if self.base_reconstruction_loss_weight <= 0.0:
            return None

        mask = self._last_mask
        target_patches = self._last_asr_patches
        reconstructed = self._last_reconstructed_patches
        if (
            mask is None
            or target_patches is None
            or reconstructed is None
            or mask.dim() != 2
            or target_patches.dim() != 3
            or reconstructed.dim() != 2
        ):
            return None

        n_patches = min(mask.size(1), target_patches.size(1))
        masked_target = target_patches[:, :n_patches, :][
            mask[:, :n_patches]
        ].detach()
        n_masked = min(reconstructed.size(0), masked_target.size(0))
        if n_masked == 0:
            return None

        return F.mse_loss(
            reconstructed[:n_masked].float(),
            masked_target[:n_masked].float(),
            reduction="mean",
        )

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
        """Handle loading state dict with backward compatibility for mask_token.
        
        This allows loading checkpoints that don't have mask_token (older versions).
        """
        mask_token_key = prefix + "mask_token"
        if mask_token_key in missing_keys:
            missing_keys.remove(mask_token_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
