from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complexnn import (
    ComplexBatchNorm,
    ComplexConv2d,
    ComplexConvTranspose2d,
    complex_cat,
)
from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")
EPS = torch.finfo(torch.double).eps


class FRCRNSeparator(AbsSeparator):
    """Frequency recurrent complex CRN separator.

    This module follows the single-channel FRCRN family at the ESPnet separator
    interface level: a complex encoder-decoder CRN predicts a complex mask, and
    a frequency-recurrence bottleneck explicitly models spectral dependency
    before decoding.
    """

    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        rnn_layer: int = 2,
        rnn_units: int = 256,
        freq_rnn_units: int = 128,
        freq_rnn_layers: int = 1,
        masking_mode: str = "E",
        bidirectional: bool = False,
        use_cbn: bool = False,
        kernel_size: int = 5,
        kernel_num: List[int] = [32, 64, 128, 256, 256, 256],
        use_builtin_complex: bool = True,
        use_noise_mask: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_builtin_complex = use_builtin_complex
        self._num_spk = num_spk
        self.use_noise_mask = use_noise_mask
        self.predict_noise = use_noise_mask
        if masking_mode not in ["C", "E", "R"]:
            raise ValueError(f"Unsupported masking mode: {masking_mode}")
        self.masking_mode = masking_mode
        self.kernel_size = kernel_size
        self.kernel_num = [2] + kernel_num

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                    ),
                    (
                        nn.BatchNorm2d(self.kernel_num[idx + 1])
                        if not use_cbn
                        else ComplexBatchNorm(self.kernel_num[idx + 1])
                    ),
                    nn.PReLU(),
                )
            )

        hidden_dim = (input_dim - 1 + 2 ** (len(self.kernel_num) - 1) - 1) // (
            2 ** (len(self.kernel_num) - 1)
        )
        hidden_dim = max(hidden_dim, 1)
        bottleneck_channels = self.kernel_num[-1]
        time_fac = 2 if bidirectional else 1
        time_dropout = dropout if rnn_layer > 1 else 0.0
        freq_dropout = dropout if freq_rnn_layers > 1 else 0.0

        self.time_rnn = nn.LSTM(
            input_size=hidden_dim * bottleneck_channels,
            hidden_size=rnn_units,
            num_layers=rnn_layer,
            dropout=time_dropout,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.time_projection = nn.Linear(
            rnn_units * time_fac, hidden_dim * bottleneck_channels
        )

        self.freq_rnn = nn.LSTM(
            input_size=bottleneck_channels,
            hidden_size=freq_rnn_units,
            num_layers=freq_rnn_layers,
            dropout=freq_dropout,
            bidirectional=True,
            batch_first=False,
        )
        self.freq_projection = nn.Linear(freq_rnn_units * 2, bottleneck_channels)
        self.freq_norm = nn.LayerNorm(bottleneck_channels)

        for idx in range(len(self.kernel_num) - 1, 0, -1):
            out_channels = self.kernel_num[idx - 1]
            if idx == 1:
                out_channels = (
                    self.kernel_num[idx - 1] * (self._num_spk + 1)
                    if self.use_noise_mask
                    else self.kernel_num[idx - 1] * self._num_spk
                )
            self.decoder.append(
                nn.Sequential(
                    ComplexConvTranspose2d(
                        self.kernel_num[idx] * 2,
                        out_channels,
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 0),
                        output_padding=(1, 0),
                    ),
                    *(
                        []
                        if idx == 1
                        else [
                            (
                                nn.BatchNorm2d(self.kernel_num[idx - 1])
                                if not use_cbn
                                else ComplexBatchNorm(self.kernel_num[idx - 1])
                            ),
                            nn.PReLU(),
                        ]
                    ),
                )
            )

        self.flatten_parameters()

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        specs = input.permute(0, 2, 1)
        real, imag = specs.real, specs.imag
        cspecs = torch.stack([real, imag], 1)
        cspecs = cspecs[:, :, 1:]

        out = cspecs
        encoder_out = []
        for layer in self.encoder:
            out = layer(out)
            encoder_out.append(out)

        batch_size, channels, dims, lengths = out.size()
        residual = out

        out = out.permute(3, 0, 1, 2).reshape(
            lengths, batch_size, channels * dims
        )
        out, _ = self.time_rnn(out)
        out = self.time_projection(out)
        out = out.reshape(lengths, batch_size, channels, dims).permute(1, 2, 3, 0)

        freq_in = out.permute(2, 0, 3, 1).reshape(dims, batch_size * lengths, channels)
        freq_out, _ = self.freq_rnn(freq_in)
        freq_out = self.freq_projection(freq_out)
        freq_out = self.freq_norm(freq_out)
        freq_out = freq_out.reshape(dims, batch_size, lengths, channels).permute(
            1, 3, 0, 2
        )
        out = residual + freq_out

        for idx, layer in enumerate(self.decoder):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = layer(out)
            out = out[..., 1:]

        masks = self.create_masks(out)
        masked = self.apply_masks(masks, real, imag)
        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(self.num_spk)], masks)
        )

        if self.use_noise_mask:
            others["mask_noise1"] = masks[-1]
            others["noise1"] = masked.pop(-1)

        return masked, ilens, others

    def flatten_parameters(self):
        self.time_rnn.flatten_parameters()
        self.freq_rnn.flatten_parameters()

    def create_masks(self, mask_tensor: torch.Tensor):
        if self.use_noise_mask:
            assert mask_tensor.shape[1] == 2 * (self._num_spk + 1), mask_tensor.shape[1]
        else:
            assert mask_tensor.shape[1] == 2 * self._num_spk, mask_tensor.shape[1]

        masks = []
        for idx in range(mask_tensor.shape[1] // 2):
            mask_real = mask_tensor[:, idx * 2]
            mask_imag = mask_tensor[:, idx * 2 + 1]
            mask_real = F.pad(mask_real, [0, 0, 1, 0])
            mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

            if is_torch_1_9_plus and self.use_builtin_complex:
                complex_mask = torch.complex(
                    mask_real.permute(0, 2, 1), mask_imag.permute(0, 2, 1)
                )
            else:
                complex_mask = ComplexTensor(
                    mask_real.permute(0, 2, 1), mask_imag.permute(0, 2, 1)
                )
            masks.append(complex_mask)

        return masks

    def apply_masks(
        self,
        masks: List[Union[torch.Tensor, ComplexTensor]],
        real: torch.Tensor,
        imag: torch.Tensor,
    ):
        masked = []
        for mask in masks:
            mask_real = mask.real.permute(0, 2, 1)
            mask_imag = mask.imag.permute(0, 2, 1)
            if self.masking_mode == "E":
                spec_mags = torch.sqrt(real**2 + imag**2 + 1e-8)
                spec_phase = torch.atan2(imag, real)
                mask_mags = (mask_real**2 + mask_imag**2) ** 0.5
                real_phase = mask_real / (mask_mags + EPS)
                imag_phase = mask_imag / (mask_mags + EPS)
                mask_phase = torch.atan2(imag_phase, real_phase)
                mask_mags = torch.tanh(mask_mags)
                est_mags = mask_mags * spec_mags
                est_phase = spec_phase + mask_phase
                enhanced_real = est_mags * torch.cos(est_phase)
                enhanced_imag = est_mags * torch.sin(est_phase)
            elif self.masking_mode == "C":
                enhanced_real = real * mask_real - imag * mask_imag
                enhanced_imag = real * mask_imag + imag * mask_real
            else:
                enhanced_real = real * mask_real
                enhanced_imag = imag * mask_imag

            if is_torch_1_9_plus and self.use_builtin_complex:
                masked.append(
                    torch.complex(
                        enhanced_real.permute(0, 2, 1),
                        enhanced_imag.permute(0, 2, 1),
                    )
                )
            else:
                masked.append(
                    ComplexTensor(
                        enhanced_real.permute(0, 2, 1),
                        enhanced_imag.permute(0, 2, 1),
                    )
                )
        return masked

    @property
    def num_spk(self):
        return self._num_spk
