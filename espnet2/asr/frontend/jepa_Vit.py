# JEPA-style Masked-Patch Frontend for ASR (without reconstruction)
# - Context encoder: ViT blocks on visible patch tokens only
# - Target encoder: ViT blocks on clean patches (EMA teacher)
# - Predictor: predicts target representations for masked patches
# - Output: denoised mel-like features (B, T, F) to plug into Conformer-CTC

from typing import Any, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft


# -------------------------
# Positional embedding (2D sin-cos)
# -------------------------
def _get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    pos: (M,) positions
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))
    out = pos.float().unsqueeze(1) * omega.unsqueeze(0)  # (M, D/2)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_t: int, grid_f: int, device) -> torch.Tensor:
    """
    returns: (grid_t*grid_f, D)
    """
    assert embed_dim % 2 == 0
    # split dim into t and f
    dim_t = embed_dim // 2
    dim_f = embed_dim - dim_t

    t = torch.arange(grid_t, device=device)
    f = torch.arange(grid_f, device=device)
    tt, ff = torch.meshgrid(t, f, indexing="ij")  # (grid_t, grid_f)

    tt = tt.reshape(-1)  # (M,)
    ff = ff.reshape(-1)  # (M,)

    emb_t = _get_1d_sincos_pos_embed(dim_t, tt)  # (M, dim_t)
    emb_f = _get_1d_sincos_pos_embed(dim_f, ff)  # (M, dim_f)

    return torch.cat([emb_t, emb_f], dim=1)  # (M, D)


# -------------------------
# ViT blocks
# -------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float, proj_drop: float):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        """
        x: (B, N, D)
        key_padding_mask: (B, N) True for valid tokens, False for padded tokens
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,N,N)

        if key_padding_mask is not None:
            # key_padding_mask True=valid, False=pad
            # mask out padded keys
            mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B,H,N,Hd)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float, attn_drop: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# -------------------------
# MAE-style masking
# -------------------------
def random_masking(x, mask_ratio: float, valid_mask: Optional[torch.Tensor] = None):
    """
    x: (B, N, D)
    valid_mask: (B, N) True for valid patches (not time-padding), False otherwise
    Returns:
      x_visible: (B, N_vis, D)
      mask: (B, N) 1=masked, 0=visible (float)
      ids_restore: (B, N) to restore original order
      ids_keep: (B, N_vis)
    """
    B, N, D = x.shape
    device = x.device

    # noise for random shuffling
    noise = torch.rand(B, N, device=device)

    if valid_mask is not None:
        # ensure invalid tokens are always treated as "masked/padded" by giving them huge noise
        noise = noise.masked_fill(~valid_mask, 1.0 + noise.max().detach() + 1.0)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # (B, N)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    N_valid = valid_mask.sum(dim=1) if valid_mask is not None else torch.full((B,), N, device=device)
    N_keep = (N_valid.float() * (1.0 - mask_ratio)).long().clamp(min=1)

    # variable keep per sample -> we will pad to max_keep for batching
    max_keep = int(N_keep.max().item())
    ids_keep_list = []
    x_keep_list = []
    keep_mask_list = []

    for b in range(B):
        keep = ids_shuffle[b, : N_keep[b]]
        ids_keep_list.append(keep)
        x_keep_list.append(x[b, keep])
        keep_mask = torch.ones(N_keep[b], device=device, dtype=torch.bool)
        keep_mask_list.append(keep_mask)

    # pad visible tokens to max_keep
    x_visible = torch.zeros(B, max_keep, D, device=device, dtype=x.dtype)
    ids_keep = torch.zeros(B, max_keep, device=device, dtype=torch.long)
    vis_pad_mask = torch.zeros(B, max_keep, device=device, dtype=torch.bool)  # True=valid visible token

    for b in range(B):
        k = x_keep_list[b].shape[0]
        x_visible[b, :k] = x_keep_list[b]
        ids_keep[b, :k] = ids_keep_list[b]
        vis_pad_mask[b, :k] = True

    # generate mask: 0 is keep, 1 is remove
    mask = torch.ones(B, N, device=device)
    for b in range(B):
        mask[b, ids_keep_list[b]] = 0.0

    if valid_mask is not None:
        # mark invalid patches as masked
        mask = mask.masked_fill(~valid_mask, 1.0)

    # unshuffle to get the binary mask in original order
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_visible, vis_pad_mask, mask, ids_restore, ids_keep


# -------------------------
# Frontend
# -------------------------
class JEPA_MaskedPatchFrontend(AbsFrontend):
    """
    JEPA-style masked-patch frontend (without reconstruction):
      - Context encoder: ViT blocks on visible patch tokens only
      - Target encoder: ViT blocks on clean patches (EMA teacher)
      - Predictor: predicts target representations for masked patches
      - Output: denoised mel-like features (x_hat) to plug into ASR backend
    """

    def __init__(
        self,
        output_dim: int = 80,
        patch_size: Any = (8, 8),   # (time, freq)
        mask_ratio: float = 0.3,

        # ViT sizes
        embed_dim: int = 256,
        enc_depth: int = 4,
        enc_heads: int = 4,
        enc_mlp_ratio: float = 4.0,
        enc_drop: float = 0.1,
        enc_attn_drop: float = 0.1,

        # Predictor configuration
        pred_depth: int = 2,
        pred_heads: int = 4,
        pred_mlp_ratio: float = 4.0,
        pred_drop: float = 0.1,
        pred_attn_drop: float = 0.1,

        # loss
        embedding_loss_weight: float = 1.0,
        ema_decay: float = 0.999,

        # audio
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
        fs: int = 16000,
        n_mels: int = 80,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        if isinstance(patch_size, list):
            patch_size = tuple(patch_size)

        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.embedding_loss_weight = embedding_loss_weight
        self.ema_decay = ema_decay

        self.hop_length = hop_length
        self.n_mels = n_mels

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
        self.logmel = LogMel(fs=fs, n_fft=n_fft, n_mels=n_mels, fmin=None, fmax=None, htk=False)

        # patch dimension
        pt, pf = self.patch_size
        self.patch_dim = pt * pf

        # patch embed (ViT style)
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        # encoder blocks
        self.encoder = nn.ModuleList([
            ViTBlock(embed_dim, enc_heads, enc_mlp_ratio, enc_drop, enc_attn_drop)
            for _ in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

        # target encoder (EMA copy)
        self.target_patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.target_encoder = nn.ModuleList([
            ViTBlock(embed_dim, enc_heads, enc_mlp_ratio, enc_drop, enc_attn_drop)
            for _ in range(enc_depth)
        ])
        self.target_norm = nn.LayerNorm(embed_dim)

        # init target as copy (EMA teacher)
        self._init_target_from_context()
        self._freeze_target()

        # Predictor: predicts target representations for masked patches
        self.predictor = nn.ModuleList([
            ViTBlock(embed_dim, pred_heads, pred_mlp_ratio, pred_drop, pred_attn_drop)
            for _ in range(pred_depth)
        ])
        self.pred_norm = nn.LayerNorm(embed_dim)

        # Projection head to map from context representation to output mel patches
        # This is used to convert representations back to mel features for ASR
        self.proj_to_patches = nn.Linear(embed_dim, self.patch_dim)

        # buffers for loss
        self._last_mask = None
        self._last_context_repr = None  # Context encoder representations
        self._last_target_repr = None  # Target encoder representations (for masked patches)
        self._last_predicted_repr = None  # Predicted target representations
        self._last_patch_mask = None  # validity mask (B,N)
        self._last_noisy_patches = None
        self._last_clean_patches = None

    def output_size(self) -> int:
        return self.output_dim

    def _init_target_from_context(self):
        self.target_patch_embed.load_state_dict(self.patch_embed.state_dict())
        for t, s in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            t.data.copy_(s.data)
        self.target_norm.load_state_dict(self.enc_norm.state_dict())

    def _freeze_target(self):
        for p in self.target_patch_embed.parameters():
            p.requires_grad = False
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_norm.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        # EMA update
        for tp, sp in zip(self.target_patch_embed.parameters(), self.patch_embed.parameters()):
            tp.data.mul_(self.ema_decay).add_(sp.data, alpha=1.0 - self.ema_decay)
        for tp, sp in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.data.mul_(self.ema_decay).add_(sp.data, alpha=1.0 - self.ema_decay)
        for tp, sp in zip(self.target_norm.parameters(), self.enc_norm.parameters()):
            tp.data.mul_(self.ema_decay).add_(sp.data, alpha=1.0 - self.ema_decay)

    def _extract_logmel(self, wav: torch.Tensor, wav_lens: torch.Tensor):
        stft, feats_lens = self.stft(wav, wav_lens)
        if isinstance(stft, torch.Tensor):
            stft = ComplexTensor(stft[..., 0], stft[..., 1])
        power = stft.real**2 + stft.imag**2
        logmel, feats_lens = self.logmel(power, feats_lens)
        return logmel, feats_lens  # (B,T,F), (B,)

    def _create_patches(self, x: torch.Tensor, feats_lens: torch.Tensor):
        """
        x: (B,T,F)
        patches: (B,N,patch_dim)
        patch_mask: (B,N) True=valid, False=invalid (time padding)
        grid: (Nt, Nf)
        """
        B, T, F = x.shape
        pt, pf = self.patch_size

        Nt = (T + pt - 1) // pt
        Nf = (F + pf - 1) // pf
        N = Nt * Nf

        pad_t = Nt * pt - T
        pad_f = Nf * pf - F
        if pad_t > 0 or pad_f > 0:
            x = torch.nn.functional.pad(x, (0, pad_f, 0, pad_t))

        x = x.view(B, Nt, pt, Nf, pf).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, N, pt * pf)

        patch_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        for b in range(B):
            valid_t = (feats_lens[b].item() + pt - 1) // pt
            if valid_t < Nt:
                invalid_start = valid_t * Nf
                patch_mask[b, invalid_start:] = False

        return x, patch_mask, (Nt, Nf), (T, F)

    def _unpatch(self, patches: torch.Tensor, grid: Tuple[int,int], orig_shape: Tuple[int,int]):
        """
        patches: (B,N,patch_dim)
        return: (B,T,F) cropped
        """
        B, N, _ = patches.shape
        Nt, Nf = grid
        T, F = orig_shape
        pt, pf = self.patch_size

        x = patches.view(B, Nt, Nf, pt, pf).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, Nt * pt, Nf * pf)
        x = x[:, :T, :F]
        return x

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        clean_input: Optional[torch.Tensor] = None,
        clean_input_lengths: Optional[torch.Tensor] = None,
    ):
        # noisy log-mel
        x_n, feats_lens = self._extract_logmel(input, input_lengths)
        noisy_patches, patch_mask, grid, orig_shape = self._create_patches(x_n, feats_lens)  # (B,N,P)

        B, N, _ = noisy_patches.shape
        device = noisy_patches.device

        # positional embedding for this grid
        pos = get_2d_sincos_pos_embed(self.embed_dim, grid[0], grid[1], device=device)  # (N,D)
        pos = pos.unsqueeze(0).expand(B, -1, -1)  # (B,N,D)

        # patch embed
        tokens = self.patch_embed(noisy_patches) + pos  # (B,N,D)

        # MAE masking: encoder sees only visible tokens
        if self.training:
            x_vis, vis_pad_mask, mask, ids_restore, ids_keep = random_masking(tokens, self.mask_ratio, valid_mask=patch_mask)
        else:
            # no masking at inference (or you can keep a fixed mask pattern)
            x_vis = tokens
            vis_pad_mask = patch_mask
            mask = torch.zeros(B, N, device=device)
            ids_restore = torch.arange(N, device=device).unsqueeze(0).repeat(B,1)
            ids_keep = torch.arange(N, device=device).unsqueeze(0).repeat(B,1)

        # Context encoder (visible only)
        context_repr = x_vis
        # key_padding_mask True=valid
        for blk in self.encoder:
            context_repr = blk(context_repr, key_padding_mask=vis_pad_mask)
        context_repr = self.enc_norm(context_repr)  # (B, Nvis(max), D)

        # Restore context representations to full sequence order
        # Build full sequence (B,N,D) in the original token order
        full_context = torch.zeros(B, N, self.embed_dim, device=device, dtype=context_repr.dtype)
        
        # Scatter visible context representations back to original positions
        for b in range(B):
            keep = ids_keep[b, :].clone()
            k_valid = vis_pad_mask[b].sum().item()
            keep = keep[:k_valid]
            full_context[b, keep] = context_repr[b, :k_valid]

        # For masked patches, use predictor to predict target representations
        mask_bool = (mask > 0.5) & patch_mask  # only valid masked patches
        
        if mask_bool.any() and self.training and clean_input is not None:
            # Get target encoder representations from clean patches
            lens = clean_input_lengths if clean_input_lengths is not None else input_lengths
            x_c, c_lens = self._extract_logmel(clean_input, lens)
            clean_patches, clean_patch_mask, _, _ = self._create_patches(x_c, c_lens)
            
            # Target encoder: process clean patches
            clean_tokens = self.target_patch_embed(clean_patches) + pos  # (B,N,D)
            target_repr = clean_tokens
            for blk in self.target_encoder:
                target_repr = blk(target_repr, key_padding_mask=clean_patch_mask)
            target_repr = self.target_norm(target_repr)  # (B,N,D)
            
            # Predictor: predict target representations for masked patches
            # Predictor sees full context (visible + masked positions with zero/mask token)
            # For masked positions, we can use zero or learnable mask token
            # Here we use the context representation directly (masked positions are zeros)
            full_context_for_pred = full_context.clone()
            
            # Apply predictor to full context sequence
            pred_full = full_context_for_pred
            for blk in self.predictor:
                pred_full = blk(pred_full, key_padding_mask=patch_mask)
            pred_full = self.pred_norm(pred_full)  # (B, N, D)
            
            # Extract predicted representations for masked patches only
            pred_repr = pred_full[mask_bool]  # (num_masked, D)
            
            # Store for loss computation
            self._last_mask = mask_bool
            self._last_patch_mask = patch_mask
            self._last_context_repr = full_context
            self._last_target_repr = target_repr
            self._last_predicted_repr = pred_repr
            self._last_noisy_patches = noisy_patches
            self._last_clean_patches = clean_patches
            
            # For output: use predicted representations for masked patches, context for visible
            full_repr = full_context.clone()
            full_repr[mask_bool] = pred_repr
        else:
            # Inference or no clean input: use context representations only
            full_repr = full_context
            self._last_mask = None
            self._last_patch_mask = None
            self._last_context_repr = None
            self._last_target_repr = None
            self._last_predicted_repr = None
            self._last_noisy_patches = None
            self._last_clean_patches = None

        # Project representations back to patch space for output
        out_patches = self.proj_to_patches(full_repr)  # (B,N,patch_dim)
        
        # Merge: unmasked from noisy, masked from predicted representations
        if mask_bool.any():
            out_patches_final = noisy_patches.clone()
            out_patches_final[mask_bool] = out_patches[mask_bool]
        else:
            out_patches_final = noisy_patches

        # Reconstruct mel
        x_hat = self._unpatch(out_patches_final, grid, orig_shape)

        return x_hat, feats_lens

    def compute_jepa_loss(self) -> Optional[torch.Tensor]:
        """
        JEPA representation matching loss (without reconstruction):
          MSE(predicted_target_repr[masked], target_repr[masked])
        """
        if self._last_mask is None or self._last_predicted_repr is None or self._last_target_repr is None:
            return None
        if not self._last_mask.any():
            return None
        
        # Get target representations for masked patches
        masked_target_repr = self._last_target_repr[self._last_mask]  # (num_masked, D)
        
        # Predicted target representations
        pred_target_repr = self._last_predicted_repr  # (num_masked, D)
        
        # MSE loss between predicted and target representations
        loss = F.mse_loss(pred_target_repr, masked_target_repr, reduction="mean")
        
        return loss * self.embedding_loss_weight