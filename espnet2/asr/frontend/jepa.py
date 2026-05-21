"""JEPA (Joint Embedding Predictive Architecture) Frontend for ASR.

This module implements a JEPA-based frontend that learns noise-robust representations
through joint embedding and predictive coding without explicit speech enhancement.
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


class JEPAFrontend(AbsFrontend):
    """JEPA (Joint Embedding Predictive Architecture) Frontend for ASR.

    This frontend implements JEPA for learning noise-robust embeddings through:
    1. Context Encoder: Maps noisy speech to context embeddings
    2. Target Encoder: Maps clean speech to target embeddings
    3. Predictor: Predicts target embeddings from context embeddings
    4. JEPA Loss: Minimizes distance between predicted and target embeddings

    Architecture:
        Noisy Speech → Context Encoder → Context Embeddings
                                                      ↓
                                                 Predictor
                                                      ↓
        Clean Speech → Target Encoder → Target Embeddings ← Predicted Embeddings
                                                      (JEPA Loss)

    During training, both noisy and clean speech are provided to learn the mapping.
    The target encoder is updated via EMA of the context encoder (no gradients).
    During inference, only noisy speech is used, and the predictor output is used for ASR.
    
    Note: Call update_target_encoder() after each optimizer step during training.

    Args:
        embedding_dim: Dimension of the joint embedding space (default: 512)
        predictor_dim: Dimension of the predictor hidden layers (default: 256)
        num_predictor_layers: Number of predictor layers (default: 2)
        dropout_rate: Dropout rate (default: 0.1)
        n_fft: FFT size for STFT (default: 512)
        hop_length: Hop length for STFT (default: 256)
        win_length: Window length for STFT (default: 512)
        fs: Sampling rate (default: 16000)
        noise_reduction_weight: Weight for noise reduction loss (default: 1.0)
        embedding_loss_weight: Weight for embedding learning loss (default: 0.5)
        ema_decay: EMA decay rate for target encoder updates (default: 0.999)
        mask_ratio: Mask ratio for predictive coding (default: 0.3)
    """

    @typechecked
    def __init__(
        self,
        embedding_dim: int = 512,
        predictor_dim: int = 256,
        num_predictor_layers: int = 2,
        dropout_rate: float = 0.1,
        n_fft: int = 512,
        hop_length: int = 128,  # Match default frontend (was 256)
        win_length: Optional[int] = None,
        fs: int = 16000,
        n_mels: int = 80,  # Match default frontend's log-mel dimension
        noise_reduction_weight: float = 1.0,
        embedding_loss_weight: float = 0.5,
        ema_decay: float = 0.999,
        mask_ratio: float = 0.3,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.predictor_dim = predictor_dim
        self.num_predictor_layers = num_predictor_layers
        self.dropout_rate = dropout_rate
        self.hop_length = hop_length
        self.noise_reduction_weight = noise_reduction_weight
        self.embedding_loss_weight = embedding_loss_weight
        self.ema_decay = ema_decay
        self.mask_ratio = mask_ratio

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
            n_mels=n_mels,  # Match default frontend
            fmin=None,
            fmax=None,
            htk=False,
        )

        # Input dimension is n_mels (log-mel spectrogram dimension)
        self.mel_dim = n_mels

        # Context encoder: Maps noisy log-mel features to context embeddings
        # Input: (B, T, n_mels) -> Output: (B, T, embedding_dim)
        # This encoder is trained with gradients
        self.context_encoder = nn.Sequential(
            nn.Linear(self.mel_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Target encoder: Maps clean log-mel features to target embeddings
        # This encoder is updated via EMA of context encoder (no gradients)
        # No dropout: use Identity instead to maintain same structure
        # Initialized with same weights as context encoder using load_state_dict
        self.target_encoder = nn.Sequential(
            nn.Linear(self.mel_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Identity(),  # dropout 자리 (no-op layer to maintain structure)
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        # Initialize target encoder with context encoder weights
        # strict=False allows skipping dropout layer (Identity has no parameters)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict(), strict=False)
        
        # Set requires_grad to False for target encoder (permanently fixed)
        with torch.no_grad():
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            self.target_encoder.eval()  # Initialize in eval mode

        # Predictor: Predicts target embeddings from context embeddings
        # Input: (B, T, embedding_dim) -> Output: (B, T, embedding_dim)
        predictor_layers = []
        input_dim = embedding_dim
        for i in range(num_predictor_layers):
            predictor_layers.extend([
                nn.Linear(input_dim, predictor_dim),
                nn.LayerNorm(predictor_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            input_dim = predictor_dim
        
        # Final projection to embedding_dim
        predictor_layers.append(nn.Linear(input_dim, embedding_dim))
        self.predictor = nn.Sequential(*predictor_layers)

        # JEPA masking: mask token for predictive coding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.mask_ratio = mask_ratio
        self.min_mask_frames = 5  # Minimum number of frames to mask (guarantees meaningful prediction)

        # Gating mechanism for combining predicted and context embeddings
        # Input: concatenation of predicted and context embeddings
        self.gate_linear = nn.Linear(embedding_dim * 2, embedding_dim)
        self.gate_activation = nn.Sigmoid()
        # Initialize bias to 1.0 so initial gate favors predicted embeddings
        # sigmoid(1) ≈ 0.73, meaning initial gate value is ~0.73 (favoring predicted over context)
        nn.init.constant_(self.gate_linear.bias, 1.0)

        # For backward compatibility, alias context_encoder as encoder
        self.encoder = self.context_encoder

    def train(self, mode: bool = True):
        """Set training mode.
        
        Override to ensure target_encoder always stays in eval mode.
        """
        super().train(mode)
        self.target_encoder.eval()  # Always keep target encoder in eval mode
        return self

    def update_target_encoder(self, decay: Optional[float] = None):
        """Update target encoder using EMA of context encoder.
        
        This should be called after each optimizer step during training.
        Dropout layers don't have parameters, so they are automatically skipped.
        
        Args:
            decay: EMA decay rate (default: self.ema_decay)
        """
        if decay is None:
            decay = self.ema_decay
        
        with torch.no_grad():
            src = dict(self.context_encoder.named_parameters())
            for name_t, p_t in self.target_encoder.named_parameters():
                p_s = src[name_t]
                p_t.data.mul_(decay).add_(p_s.data, alpha=1 - decay)

    def output_size(self) -> int:
        """Return the output dimension of the frontend."""
        return self.embedding_dim

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor,
        clean_input: Optional[torch.Tensor] = None,
        clean_input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            input: (Batch, Nsamples) - Raw audio waveform (noisy speech)
            input_lengths: (Batch,) - Length of each sequence
            clean_input: Optional (Batch, Nsamples) - Clean speech for JEPA training
            clean_input_lengths: Optional (Batch,) - Length of clean sequences

        Returns:
            output: (Batch, T, embedding_dim) - Predicted clean embeddings
            output_lengths: (Batch,) - Length of each sequence
        """
        # Process noisy input (context)
        context_embeddings, feats_lens = self._extract_embeddings(
            input, input_lengths, use_context=True
        )

        B, T, D = context_embeddings.shape

        # JEPA masking: block masking (contiguous frames)
        # Create padding mask
        pad_mask = torch.arange(T, device=feats_lens.device)[None, :] >= feats_lens[:, None]  # (B, T)
        
        # Create block mask for predictive coding
        if self.training:
            time_mask = self._generate_block_mask(B, T, feats_lens, pad_mask, device=context_embeddings.device)
            # Ensure padding positions are never masked
            time_mask = time_mask & (~pad_mask)
        else:
            # No masking during inference
            time_mask = torch.zeros(B, T, dtype=torch.bool, device=context_embeddings.device)

        # Apply masking: replace masked positions with mask_token
        context_masked = context_embeddings.clone()
        if time_mask.any():
            context_masked[time_mask] = self.mask_token

        # Predict target embeddings from masked context
        predicted_embeddings = self.predictor(context_masked)

        # During training with clean speech, compute target embeddings (teacher)
        if clean_input is not None and self.training:
            lens = clean_input_lengths if clean_input_lengths is not None else input_lengths
            # Target encoder: no_grad (always in eval mode via train() override)
            with torch.no_grad():
                target_embeddings, _ = self._extract_embeddings(clean_input, lens, use_context=False)
            
            # Store for loss computation
            self._last_target_embeddings = target_embeddings
            self._last_predicted_embeddings = predicted_embeddings
            self._last_time_mask = time_mask
        else:
            self._last_target_embeddings = None
            self._last_predicted_embeddings = None
            self._last_time_mask = None

        # Gating mechanism: learnable combination of predicted and context embeddings
        # Concatenate predicted and context embeddings
        combined = torch.cat([predicted_embeddings, context_embeddings], dim=-1)  # (B, T, 2*D)
        # Compute gate weights
        gate = self.gate_activation(self.gate_linear(combined))  # (B, T, D)
        # Combine using gating
        output = gate * predicted_embeddings + (1 - gate) * context_embeddings

        # Apply layer normalization for stability
        output = F.layer_norm(output, output.shape[-1:])

        return output, feats_lens

    def _extract_embeddings(
        self, 
        input: torch.Tensor, 
        input_lengths: torch.Tensor,
        use_context: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings from audio waveform.

        Args:
            input: (Batch, Nsamples) - Raw audio waveform
            input_lengths: (Batch,) - Length of each sequence
            use_context: If True, use context encoder; if False, use target encoder

        Returns:
            embeddings: (Batch, T, embedding_dim)
            feats_lens: (Batch,)
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

        # 4. Encode: Map log-mel features to embeddings
        if use_context:
            embeddings = self.context_encoder(log_mel)
        else:
            # Target encoder: no dropout needed (deterministic embeddings for stable training)
            embeddings = self.target_encoder(log_mel)

        return embeddings, feats_lens

    def _generate_block_mask(
        self,
        B: int,
        T: int,
        feats_lens: torch.Tensor,
        pad_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate block mask for JEPA predictive coding.
        
        Creates contiguous blocks of masked frames instead of random frame masking.
        This makes the prediction task more challenging.
        
        Args:
            B: Batch size
            T: Sequence length
            feats_lens: (B,) - Length of each sequence
            pad_mask: (B, T) - Boolean mask for padding positions
            device: Device to create mask on
            
        Returns:
            time_mask: (B, T) - Boolean mask indicating masked positions
        """
        time_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        
        for b in range(B):
            valid_length = int(feats_lens[b].item())
            if valid_length == 0:
                continue
                
            # Calculate number of frames to mask
            num_mask = int(valid_length * self.mask_ratio)
            # Guarantee minimum mask frames for meaningful prediction
            num_mask = max(num_mask, min(self.min_mask_frames, valid_length))
            if num_mask == 0:
                continue
            
            # Block masking: create one or more contiguous blocks
            # For small num_mask, use single block to avoid variance
            min_block_len = max(1, num_mask // 3)  # Minimum block length (at least 1/3 of total)
            
            if num_mask <= self.min_mask_frames * 2:
                # Small mask budget: use single block
                block_len = num_mask
                start_pos = torch.randint(0, max(1, valid_length - block_len + 1), (1,), device=device).item()
                time_mask[b, start_pos:start_pos + block_len] = True
            else:
                # Larger mask budget: use 1-3 blocks
                num_blocks = torch.randint(1, min(4, num_mask // min_block_len + 1), (1,), device=device).item()
                num_blocks = min(num_blocks, num_mask // min_block_len)
                
                if num_blocks == 1:
                    # Single block
                    block_len = num_mask
                    start_pos = torch.randint(0, max(1, valid_length - block_len + 1), (1,), device=device).item()
                    time_mask[b, start_pos:start_pos + block_len] = True
                else:
                    # Multiple blocks: distribute mask budget with minimum block length guarantee
                    masked_count = 0
                    for i in range(num_blocks):
                        remaining = num_mask - masked_count
                        if remaining < min_block_len:
                            break
                        
                        # Last block: use all remaining
                        if i == num_blocks - 1:
                            block_len = remaining
                        else:
                            # Block length: at least min_block_len, at most reasonable max
                            max_block_len = min(remaining - min_block_len * (num_blocks - i - 1), valid_length // 3)
                            block_len = torch.randint(min_block_len, max_block_len + 1, (1,), device=device).item()
                        
                        # Random start position (avoid overlapping with already masked regions)
                        max_start = valid_length - block_len
                        if max_start < 0:
                            break
                        start_pos = torch.randint(0, max_start + 1, (1,), device=device).item()
                        
                        # Check for overlap with existing masks
                        if not time_mask[b, start_pos:start_pos + block_len].any():
                            time_mask[b, start_pos:start_pos + block_len] = True
                            masked_count += block_len
                        else:
                            # Try different position (simple retry)
                            for retry in range(10):
                                start_pos = torch.randint(0, max_start + 1, (1,), device=device).item()
                                if not time_mask[b, start_pos:start_pos + block_len].any():
                                    time_mask[b, start_pos:start_pos + block_len] = True
                                    masked_count += block_len
                                    break
        
        return time_mask

    def compute_jepa_loss(
        self,
        predicted_embeddings: Optional[torch.Tensor] = None,
        target_embeddings: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute JEPA loss between predicted and target embeddings.
        
        JEPA loss is computed only on masked positions (predictive coding).

        Args:
            predicted_embeddings: (Batch, T, embedding_dim) - Predicted embeddings
            target_embeddings: (Batch, T, embedding_dim) - Target embeddings  
            time_mask: (Batch, T) - Boolean mask indicating masked positions

        Returns:
            loss: Scalar JEPA loss (computed only on masked positions)
        """
        # Use stored values if not provided
        if predicted_embeddings is None:
            predicted_embeddings = self._last_predicted_embeddings
        if target_embeddings is None:
            target_embeddings = self._last_target_embeddings
        if time_mask is None:
            time_mask = self._last_time_mask

        if predicted_embeddings is None or target_embeddings is None or time_mask is None:
            return torch.tensor(0.0, device=predicted_embeddings.device if predicted_embeddings is not None else 'cpu')

        # Cosine distance between predicted and target embeddings
        # Cosine similarity: normalized dot product
        pred_norm = F.normalize(predicted_embeddings, p=2, dim=-1)  # (Batch, T, D)
        target_norm = F.normalize(target_embeddings.detach(), p=2, dim=-1)  # (Batch, T, D)
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)  # (Batch, T)
        
        # Cosine distance = 1 - cosine similarity (0 when identical, 2 when opposite)
        loss = 1 - cosine_sim  # (Batch, T)

        # JEPA loss: only compute on masked positions
        if time_mask.any():
            loss = loss[time_mask].mean()
        else:
            loss = torch.tensor(0.0, device=loss.device)

        return loss
