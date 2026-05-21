"""JEPA Residual Denoiser Frontend for ASR.

This module implements a JEPA-style residual denoiser that outputs denoised log-mel features.
The predictor outputs a residual delta in log-mel space, which is gradually mixed with the
noisy input using a warm-up schedule.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft


class JEPAResidualFrontend(AbsFrontend):
    """JEPA Residual Denoiser Frontend for ASR.

    This frontend implements a JEPA-style denoiser that works directly in log-mel space:
    - x_n = noisy log-mel (80-dim, same as baseline)
    - x_c = clean log-mel (from paired data)
    - Context encoder processes x_n → context features
    - Predictor outputs delta in mel space: Δx = h(context) (shape = 80)
    - Denoised features: x_hat = x_n + α(t) * Δx
    - α(t) starts at 0 and gradually increases (warm-up)

    Architecture:
        Noisy Speech → STFT → Log-Mel (x_n)
                                         ↓
                                    Context Encoder
                                         ↓
                                      Predictor → Δx (80-dim)
                                         ↓
        x_hat = x_n + α(t) * Δx ←────────┘

    During training, clean speech is used to compute target (x_c) for loss computation.
    The residual mixing weight α(t) starts at 0 and gradually increases to allow
    stable training without shocking the encoder.

    Args:
        predictor_dim: Dimension of the predictor hidden layers (default: 256)
        num_predictor_layers: Number of predictor layers (default: 2)
        dropout_rate: Dropout rate (default: 0.1)
        n_fft: FFT size for STFT (default: 512)
        hop_length: Hop length for STFT (default: 128)
        win_length: Window length for STFT (default: 512)
        fs: Sampling rate (default: 16000)
        n_mels: Number of mel bins (default: 80)
        embedding_loss_weight: Weight for residual loss (default: 1.0)
        residual_mix_start: Initial residual mixing weight (default: 0.0)
        residual_mix_end: Final residual mixing weight (default: 1.0)
        residual_mix_warmup_steps: Number of steps to warm up residual mixing (default: 10000)
        window: Window type for STFT (default: "hann")
        center: Whether to center the STFT (default: True)
        normalized: Whether to normalize the STFT (default: False)
        onesided: Whether to use onesided STFT (default: True)
    """

    @typechecked
    def __init__(
        self,
        predictor_dim: int = 256,
        num_predictor_layers: int = 2,
        dropout_rate: float = 0.1,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
        fs: int = 16000,
        n_mels: int = 80,
        embedding_loss_weight: float = 1.0,
        residual_mix_start: float = 0.0,
        residual_mix_end: float = 1.0,
        residual_mix_warmup_steps: int = 10000,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()

        self.predictor_dim = predictor_dim
        self.num_predictor_layers = num_predictor_layers
        self.dropout_rate = dropout_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.embedding_loss_weight = embedding_loss_weight
        self.residual_mix_start = residual_mix_start
        self.residual_mix_end = residual_mix_end
        self.residual_mix_warmup_steps = residual_mix_warmup_steps

        # STFT for spectral analysis
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        # Log-Mel transform (same as default frontend)
        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=None,
            fmax=None,
            htk=False,
        )

        # Context encoder: Maps noisy log-mel features to context features
        # Input: (B, T, n_mels) -> Output: (B, T, predictor_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.n_mels, predictor_dim),
            nn.LayerNorm(predictor_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(predictor_dim, predictor_dim),
            nn.LayerNorm(predictor_dim),
        )

        # Predictor: Predicts residual delta in log-mel space
        # Input: (B, T, predictor_dim) -> Output: (B, T, n_mels)
        predictor_layers = []
        input_dim = predictor_dim
        for i in range(num_predictor_layers):
            predictor_layers.extend([
                nn.Linear(input_dim, predictor_dim),
                nn.LayerNorm(predictor_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            input_dim = predictor_dim
        
        # Final projection to n_mels (residual delta)
        predictor_layers.append(nn.Linear(input_dim, self.n_mels))
        self.predictor = nn.Sequential(*predictor_layers)

        # Step counter for residual mixing warm-up
        self.register_buffer("_residual_mix_step", torch.tensor(0, dtype=torch.long))

        # For backward compatibility, alias context_encoder as encoder
        self.encoder = self.context_encoder

    def output_size(self) -> int:
        """Return the output dimension of the frontend (n_mels)."""
        return self.n_mels

    def _get_residual_mix_weight(self) -> float:
        """Get current residual mixing weight based on warm-up schedule.
        
        Returns:
            Current mixing weight α(t)
        """
        if not self.training:
            return self.residual_mix_end
        
        step = self._residual_mix_step.item()
        if step < self.residual_mix_warmup_steps:
            # Linear interpolation: start -> end over warmup_steps
            progress = step / self.residual_mix_warmup_steps
            alpha = self.residual_mix_start * (1 - progress) + self.residual_mix_end * progress
        else:
            alpha = self.residual_mix_end
        
        # Update step counter for next iteration
        self._residual_mix_step += 1
        return alpha

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
            output: (Batch, T, n_mels) - Denoised log-mel features
            output_lengths: (Batch,) - Length of each sequence
        """
        # Extract noisy log-mel features: x_n
        x_n, feats_lens = self._extract_logmel(input, input_lengths)

        # Context encoder: process noisy log-mel
        context_features = self.context_encoder(x_n)  # (B, T, predictor_dim)

        # Predictor: predict residual delta in log-mel space
        delta_x = self.predictor(context_features)  # (B, T, n_mels)

        # Get current residual mixing weight
        alpha = self._get_residual_mix_weight()

        # Denoised features: x_hat = x_n + α(t) * Δx
        x_hat = x_n + alpha * delta_x

        # During training with clean speech, store for loss computation
        if clean_input is not None and self.training:
            lens = clean_input_lengths if clean_input_lengths is not None else input_lengths
            # Extract clean log-mel features: x_c
            x_c, _ = self._extract_logmel(clean_input, lens)
            
            # Store for loss computation
            self._last_x_n = x_n
            self._last_x_c = x_c
            self._last_delta_x = delta_x
            self._last_x_hat = x_hat
            self._last_alpha = alpha
        else:
            self._last_x_n = None
            self._last_x_c = None
            self._last_delta_x = None
            self._last_x_hat = None
            self._last_alpha = None

        return x_hat, feats_lens

    def _extract_logmel(
        self, 
        input: torch.Tensor, 
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract log-mel features from audio waveform.

        Args:
            input: (Batch, Nsamples) - Raw audio waveform
            input_lengths: (Batch,) - Length of each sequence

        Returns:
            log_mel: (Batch, T, n_mels) - Log-mel spectrogram
            feats_lens: (Batch,) - Length of each sequence
        """
        # 1. STFT: Convert waveform to spectral features
        input_stft, feats_lens = self.stft(input, input_lengths)
        
        # Convert to ComplexTensor if needed
        if isinstance(input_stft, torch.Tensor):
            assert input_stft.shape[-1] == 2, f"Expected last dim to be 2, got {input_stft.shape}"
            input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        
        # 2. Compute power spectrum: |STFT|^2
        input_power = input_stft.real**2 + input_stft.imag**2

        # 3. Convert to log-mel spectrogram
        log_mel, feats_lens = self.logmel(input_power, feats_lens)

        return log_mel, feats_lens

    def compute_jepa_loss(self) -> Optional[torch.Tensor]:
        """Compute JEPA residual loss.
        
        Loss is computed as: ||x_c - x_hat||^2 = ||x_c - (x_n + α * Δx)||^2
        
        Returns:
            loss: Scalar residual loss, or None if not in training mode
        """
        if self._last_x_c is None or self._last_x_hat is None:
            return None

        # Residual loss: MSE between clean log-mel and denoised log-mel
        loss = F.mse_loss(self._last_x_hat, self._last_x_c, reduction='mean')

        return loss

