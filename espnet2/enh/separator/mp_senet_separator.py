from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from espnet2.enh.separator.abs_separator import AbsSeparator


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features: int, beta: float = 2.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class SPConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        upscale_factor: int = 2,
    ):
        super().__init__()
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * upscale_factor,
            kernel_size=kernel_size,
            stride=(1, 1),
        )
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        batch_size, channels, height, width = x.shape
        x = x.view(
            batch_size,
            self.upscale_factor,
            channels // self.upscale_factor,
            height,
            width,
        )
        x = x.permute(0, 2, 3, 4, 1)
        return x.contiguous().view(
            batch_size,
            channels // self.upscale_factor,
            height,
            width * self.upscale_factor,
        )


class DenseBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int] = (2, 3),
        depth: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for idx in range(depth):
            dilation = 2**idx
            self.layers.append(
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, dilation, 0), value=0.0),
                    nn.Conv2d(
                        channels * (idx + 1),
                        channels,
                        kernel_size,
                        dilation=(dilation, 1),
                    ),
                    nn.InstanceNorm2d(channels, affine=True),
                    nn.PReLU(channels),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for layer in self.layers:
            x = layer(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, channels: int, in_channels: int = 2, dense_depth: int = 4):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dense_block = DenseBlock(channels, depth=dense_depth)
        self.downsample = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.dense_block(x)
        return self.downsample(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.gru = nn.GRU(
            channels,
            channels * 2,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
        )
        gru_out = channels * 4 if bidirectional else channels * 2
        self.linear = nn.Linear(gru_out, channels)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        x_att, _ = self.attention(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.dropout1(x_att)

        residual = x
        x_norm = self.norm2(x)
        self.gru.flatten_parameters()
        x_ffn, _ = self.gru(x_norm)
        x_ffn = torch.nn.functional.leaky_relu(x_ffn)
        x_ffn = self.linear(x_ffn)
        x = residual + self.dropout2(x_ffn)
        return self.norm3(x)


class TSTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_transformer = TransformerBlock(
            channels,
            num_heads=num_heads,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.freq_transformer = TransformerBlock(
            channels,
            num_heads=num_heads,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, frames, freq_bins = x.size()

        x_time = x.permute(0, 3, 2, 1).contiguous()
        x_time = x_time.view(batch_size * freq_bins, frames, channels)
        x_time = self.time_transformer(x_time) + x_time

        x_freq = x_time.view(batch_size, freq_bins, frames, channels)
        x_freq = x_freq.permute(0, 2, 1, 3).contiguous()
        x_freq = x_freq.view(batch_size * frames, freq_bins, channels)
        x_freq = self.freq_transformer(x_freq) + x_freq

        return x_freq.view(batch_size, frames, freq_bins, channels).permute(0, 3, 1, 2)


class MaskDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        input_dim: int,
        beta: float = 2.0,
        dense_depth: int = 4,
    ):
        super().__init__()
        self.dense_block = DenseBlock(channels, depth=dense_depth)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(channels, channels, (1, 3), upscale_factor=2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, 1, (1, 2)),
        )
        self.learnable_sigmoid = LearnableSigmoid2d(input_dim, beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return self.learnable_sigmoid(x)


class PhaseDecoder(nn.Module):
    def __init__(self, channels: int, dense_depth: int = 4):
        super().__init__()
        self.dense_block = DenseBlock(channels, depth=dense_depth)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(channels, channels, (1, 3), upscale_factor=2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.phase_conv_real = nn.Conv2d(channels, 1, (1, 2))
        self.phase_conv_imag = nn.Conv2d(channels, 1, (1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.phase_conv(x)
        real = self.phase_conv_real(x)
        imag = self.phase_conv_imag(x)
        phase = torch.atan2(imag, real)
        return phase.permute(0, 3, 2, 1).squeeze(-1)


class MPSENetSeparator(AbsSeparator):
    """MP-SENet separator adapted to ESPnet's complex STFT interface.

    This follows the MP-SENet generator design: dense TF encoder, stacked
    time-frequency transformer blocks, parallel magnitude-mask and phase
    decoders, and complex spectrum reconstruction.
    """

    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        dense_channel: int = 64,
        num_tsblocks: int = 4,
        num_heads: int = 4,
        beta: float = 2.0,
        dense_depth: int = 4,
        bidirectional: bool = True,
        dropout: float = 0.0,
        inference_chunk_size: int = 201,
        inference_chunk_overlap: int = 20,
    ):
        super().__init__()
        if num_spk != 1:
            raise ValueError("MPSENetSeparator currently supports num_spk=1 only")
        self._num_spk = num_spk
        self.input_dim = input_dim
        self.inference_chunk_size = inference_chunk_size
        self.inference_chunk_overlap = inference_chunk_overlap
        self.dense_encoder = DenseEncoder(
            dense_channel,
            in_channels=2,
            dense_depth=dense_depth,
        )
        self.ts_transformers = nn.ModuleList(
            [
                TSTransformerBlock(
                    dense_channel,
                    num_heads=num_heads,
                    bidirectional=bidirectional,
                    dropout=dropout,
                )
                for _ in range(num_tsblocks)
            ]
        )
        self.mask_decoder = MaskDecoder(
            dense_channel,
            input_dim=input_dim,
            beta=beta,
            dense_depth=dense_depth,
        )
        self.phase_decoder = PhaseDecoder(
            dense_channel,
            dense_depth=dense_depth,
        )

    def forward(
        self,
        input: Union[torch.Tensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        if self.inference_chunk_size > 0 and input.size(1) > self.inference_chunk_size:
            enhanced = self._forward_chunked(input, ilens)
            return [enhanced], ilens, OrderedDict()

        enhanced, others = self._forward_full(input)
        return [enhanced], ilens, others

    def _forward_full(self, input: torch.Tensor) -> Tuple[torch.Tensor, OrderedDict]:
        specs = input.permute(0, 2, 1)
        real = specs.real
        imag = specs.imag
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-8)
        phase = torch.atan2(imag, real)

        x = torch.stack((magnitude, phase), dim=-1).permute(0, 3, 2, 1)
        x = self.dense_encoder(x)
        for block in self.ts_transformers:
            x = block(x)

        magnitude_mask = self.mask_decoder(x)
        enhanced_magnitude = magnitude * magnitude_mask
        enhanced_phase = self.phase_decoder(x)
        enhanced_real = enhanced_magnitude * torch.cos(enhanced_phase)
        enhanced_imag = enhanced_magnitude * torch.sin(enhanced_phase)
        enhanced = torch.complex(
            enhanced_real.permute(0, 2, 1),
            enhanced_imag.permute(0, 2, 1),
        )

        others = OrderedDict()
        others["mask_spk1"] = magnitude_mask.permute(0, 2, 1)
        others["phase_spk1"] = enhanced_phase.permute(0, 2, 1)
        return enhanced, others

    def _forward_chunked(self, input: torch.Tensor, ilens: torch.Tensor) -> torch.Tensor:
        chunk_size = self.inference_chunk_size
        overlap = min(max(self.inference_chunk_overlap, 0), chunk_size - 1)
        step = chunk_size - overlap
        batch_size, num_frames, freq_bins = input.shape
        enhanced_sum = input.new_zeros(input.shape)
        weight = input.real.new_zeros(batch_size, num_frames, 1)

        for start in range(0, num_frames, step):
            end = min(start + chunk_size, num_frames)
            if end <= start:
                break
            chunk = input[:, start:end]
            chunk_enhanced, _ = self._forward_full(chunk)
            enhanced_sum[:, start:end] = enhanced_sum[:, start:end] + chunk_enhanced
            weight[:, start:end] = weight[:, start:end] + 1.0
            if end == num_frames:
                break

        return enhanced_sum / weight.clamp_min(1.0).to(enhanced_sum.dtype)

    @property
    def num_spk(self):
        return self._num_spk
