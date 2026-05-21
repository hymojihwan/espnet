from contextlib import nullcontext
from typing import Optional, Tuple, Union

import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class FrozenEnhLatentFrontend(AbsFrontend):
    """Frozen enhancement latent frontend for ASR.

    Waveform -> frozen enhancement encoder/separator -> latent feature -> projection.
    """

    @typechecked
    def __init__(
        self,
        enh_train_config: str,
        enh_model_file: str,
        fs: Union[int, str] = 16000,
        output_size: int = 80,
        latent_downsample: int = 8,
        enh_no_grad: bool = True,
        n_fft: Optional[int] = None,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_mels: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        from espnet2.tasks.enh import EnhancementTask

        enh_model, _ = EnhancementTask.build_model_from_file(
            enh_train_config, enh_model_file, device="cpu"
        )
        self.enh_model = enh_model.enh_model if hasattr(enh_model, "enh_model") else enh_model
        self.enh_model.eval()
        for param in self.enh_model.parameters():
            param.requires_grad = False

        self.enh_no_grad = enh_no_grad
        self.latent_downsample = max(1, latent_downsample)
        latent_dim = int(getattr(self.enh_model.encoder, "output_dim", 256))
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(latent_dim),
            torch.nn.Linear(latent_dim, output_size),
        )
        self._output_size = output_size

    def output_size(self) -> int:
        return self._output_size

    def _downsample(
        self,
        latent: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.latent_downsample <= 1:
            return latent, lengths

        # [B, T, C] -> [B, C, T]
        pooled = F.avg_pool1d(
            latent.transpose(1, 2),
            kernel_size=self.latent_downsample,
            stride=self.latent_downsample,
            ceil_mode=True,
        ).transpose(1, 2)
        new_lengths = torch.div(
            lengths + self.latent_downsample - 1,
            self.latent_downsample,
            rounding_mode="trunc",
        )
        return pooled, new_lengths

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input.dim() == 3 and input.size(1) == 1:
            input = input.squeeze(1)

        ctx = torch.no_grad() if self.enh_no_grad else nullcontext()
        with ctx:
            feature_mix, flens = self.enh_model.encoder(input, input_lengths)
            feature_pre, _, _ = self.enh_model.separator(feature_mix, flens, {})
            if isinstance(feature_pre, (list, tuple)):
                latent = feature_pre[0]
            else:
                latent = feature_pre

        latent, flens = self._downsample(latent, flens)
        feats = self.proj(latent)
        return feats, flens
