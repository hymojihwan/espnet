from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from espnet2.enh.separator.abs_separator import AbsSeparator


class _DemucsEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 4,
            ),
            nn.GroupNorm(1, out_channels),
            nn.GLU(dim=1),
            nn.Conv1d(out_channels // 2, out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DemucsDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        is_last: bool = False,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 4,
            ),
            nn.Identity() if is_last else nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DemucsSeparator(AbsSeparator):
    """Compact waveform Demucs-style separator.

    This module is intended to be used with ``encoder: same`` and ``decoder: same``.
    It receives waveform samples, predicts enhanced waveform samples, and therefore
    can be swapped against a ConvTasNet enhancement frontend while keeping the ASR
    model fixed.
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_spk: int = 1,
        channels: int = 64,
        depth: int = 5,
        growth: int = 1,
        kernel_size: int = 8,
        stride: int = 4,
        lstm_layers: int = 2,
        rescale: float = 0.1,
    ):
        super().__init__()
        if input_dim != 1:
            raise ValueError(
                "DemucsSeparator expects waveform input. Use encoder: same "
                f"(input_dim=1), got input_dim={input_dim}."
            )
        if num_spk != 1:
            raise ValueError("DemucsSeparator currently supports num_spk=1 only.")

        self._num_spk = num_spk
        self.depth = depth
        self.stride = stride
        self.rescale = rescale

        encoders = []
        encoder_channels = []
        in_channels = 1
        for layer_idx in range(depth):
            hidden = channels * (growth**layer_idx)
            out_channels = hidden * 2
            encoders.append(
                _DemucsEncoderBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            encoder_channels.append(out_channels)
            in_channels = out_channels
        self.encoders = nn.ModuleList(encoders)

        bottleneck_channels = encoder_channels[-1]
        self.lstm = nn.LSTM(
            bottleneck_channels,
            bottleneck_channels,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm_proj = nn.Linear(2 * bottleneck_channels, bottleneck_channels)

        decoders = []
        in_channels = bottleneck_channels
        for layer_idx in reversed(range(depth)):
            skip_channels = encoder_channels[layer_idx]
            out_channels = 1 if layer_idx == 0 else encoder_channels[layer_idx - 1]
            decoders.append(
                _DemucsDecoderBlock(
                    in_channels + skip_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    is_last=layer_idx == 0,
                )
            )
            in_channels = out_channels
        self.decoders = nn.ModuleList(decoders)

    @property
    def num_spk(self):
        return self._num_spk

    def _pad_to_valid_length(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        original_length = x.size(-1)
        valid = self.stride**self.depth
        pad = (valid - original_length % valid) % valid
        if pad > 0:
            x = F.pad(x, (0, pad))
        return x, original_length

    @staticmethod
    def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
        if x.size(-1) > length:
            return x[..., :length]
        if x.size(-1) < length:
            return F.pad(x, (0, length - x.size(-1)))
        return x

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        if input.dim() == 3:
            input = input.squeeze(-1)
        if input.dim() != 2:
            raise RuntimeError(
                "DemucsSeparator expects input shape (B, T) or (B, T, 1), "
                f"got {tuple(input.shape)}."
            )

        original_input = input
        x, original_length = self._pad_to_valid_length(input.unsqueeze(1))

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.lstm_proj(x)
        x = x.transpose(1, 2)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = self._match_length(x, skip.size(-1))
            x = decoder(torch.cat([x, skip], dim=1))

        enhanced = self._match_length(x.squeeze(1), original_length)
        enhanced = original_input + self.rescale * enhanced
        others = OrderedDict()
        return [enhanced], ilens, others

    def forward_streaming(self, input_frame: torch.Tensor, buffer=None):
        raise NotImplementedError("DemucsSeparator does not support streaming.")
