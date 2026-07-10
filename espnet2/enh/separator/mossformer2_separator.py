import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum

from espnet2.enh.separator.abs_separator import AbsSeparator


EPS = 1e-8


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def rotate_queries_or_keys(self, x):
        seq_len = x.shape[-2]
        freqs = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = einsum("i,j->ij", freqs, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype=x.dtype)
        while emb.dim() < x.dim():
            emb = emb.unsqueeze(0)
        rotary_dim = emb.size(-1)
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x_rot = (x_rot * emb.cos()) + (rotate_half(x_rot) * emb.sin())
        return torch.cat((x_rot, x_pass), dim=-1)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class LayerNorm(nn.Module):
    def __init__(
        self,
        input_size=None,
        input_shape=None,
        eps=1e-05,
        elementwise_affine=True,
    ):
        super().__init__()
        if input_shape is not None:
            input_size = input_shape[2:]
        self.norm = torch.nn.LayerNorm(
            input_size,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x):
        return self.norm(x)


class CLayerNorm(nn.LayerNorm):
    def forward(self, sample):
        if sample.dim() != 3:
            raise RuntimeError("CLayerNorm only accepts 3-D input")
        sample = torch.transpose(sample, 1, 2)
        sample = super().forward(sample)
        sample = torch.transpose(sample, 1, 2)
        return sample


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class DepthwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 17,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        assert expansion_factor == 2
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            DepthwiseConv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.sequential(inputs).transpose(1, 2)


class UniDeepFsmn(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(
            output_dim,
            output_dim,
            [lorder + lorder - 1, 1],
            [1, 1],
            groups=output_dim,
            bias=False,
        )

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])
        out = x_per + self.conv1(y)
        out1 = out.permute(0, 3, 2, 1)
        return input + out1.squeeze()


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.twidth = lorder * 2 - 1
        self.kernel_size = (self.twidth, 1)
        for i in range(self.depth):
            dil = 2**i
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((0, 0, pad_length, pad_length), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                    groups=self.in_channels,
                    bias=False,
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class UniDeepFsmn_dilated(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv = DilatedDenseNet(depth=2, lorder=lorder, in_channels=output_dim)

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        out = self.conv(x_per)
        out1 = out.permute(0, 3, 2, 1)
        return input + out1.squeeze()


class FFConvM(nn.Module):
    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mdl(x)


class Gated_FSMN_dilated(nn.Module):
    def __init__(self, in_channels, out_channels, lorder, hidden_size):
        super().__init__()
        self.to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        self.to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        self.fsmn = UniDeepFsmn_dilated(
            in_channels, out_channels, lorder, hidden_size
        )

    def forward(self, x):
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x)
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input
        return x


class Gated_FSMN_Block_Dilated(nn.Module):
    def __init__(self, dim, inner_channels=256, group_size=256, norm_type="scalenorm"):
        super().__init__()
        self.group_size = group_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = Gated_FSMN_dilated(
            inner_channels, inner_channels, lorder=20, hidden_size=inner_channels
        )
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        conv1 = self.conv1(input.transpose(2, 1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))
        norm2 = self.norm2(seq_out.transpose(2, 1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2, 1) + input


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = einsum("... d, h d -> ... h d", x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class FLASH_ShareA_FFConvM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=1.0,
        causal=False,
        dropout=0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=True,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens
        self.rotary_pos_emb = rotary_pos_emb
        self.dropout = nn.Dropout(dropout)
        self.to_hidden = FFConvM(
            dim_in=dim,
            dim_out=hidden_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.to_qk = FFConvM(
            dim_in=dim,
            dim_out=query_key_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)
        self.to_out = FFConvM(
            dim_in=dim * 2,
            dim_out=dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.gateActivate = nn.Sigmoid()

    def forward(self, x, *, mask=None):
        normed_x = x
        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.0)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u)
        out = (att_u * v) * self.gateActivate(att_v * u)
        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size
        if exists(mask):
            lin_mask = rearrange(mask, "... -> ... 1")
            lin_k = lin_k.masked_fill(~lin_mask, 0.0)

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(
                self.rotary_pos_emb.rotate_queries_or_keys,
                (quad_q, lin_q, quad_k, lin_k),
            )

        padding = padding_to_multiple_of(n, g)
        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(
                lambda t: F.pad(t, (0, 0, 0, padding), value=0.0),
                (quad_q, quad_k, lin_q, lin_k, v, u),
            )
            mask = default(
                mask,
                torch.ones((b, n), device=device, dtype=torch.bool),
            )
            mask = F.pad(mask, (0, padding), value=False)

        quad_q, quad_k, lin_q, lin_k, v, u = map(
            lambda t: rearrange(t, "b (g n) d -> b g n d", n=self.group_size),
            (quad_q, quad_k, lin_q, lin_k, v, u),
        )

        if exists(mask):
            mask = rearrange(mask, "b (g j) -> b g 1 j", j=g)

        sim = einsum("... i d, ... j d -> ... i j", quad_q, quad_k) / g
        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)
        if exists(mask):
            attn = attn.masked_fill(~mask, 0.0)
        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.0)

        quad_out_v = einsum("... i j, ... j d -> ... i d", attn, v)
        quad_out_u = einsum("... i j, ... j d -> ... i d", attn, u)

        if self.causal:
            lin_kv = einsum("b g n d, b g n e -> b g d e", lin_k, v) / g
            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.0)
            lin_out_v = einsum("b g d e, b g n d -> b g n e", lin_kv, lin_q)
            lin_ku = einsum("b g n d, b g n e -> b g d e", lin_k, u) / g
            lin_ku = lin_ku.cumsum(dim=1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.0)
            lin_out_u = einsum("b g d e, b g n d -> b g n e", lin_ku, lin_q)
        else:
            lin_kv = einsum("b g n d, b g n e -> b d e", lin_k, v) / n
            lin_out_v = einsum("b g n d, b d e -> b g n e", lin_q, lin_kv)
            lin_ku = einsum("b g n d, b g n e -> b d e", lin_k, u) / n
            lin_out_u = einsum("b g n d, b d e -> b g n e", lin_q, lin_ku)

        return map(
            lambda t: rearrange(t, "b g n d -> b (g n) d")[:, :n],
            (quad_out_v + lin_out_v, quad_out_u + lin_out_u),
        )


class FLASHTransformer_DualA_FSMN(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        causal=False,
        attn_dropout=0.1,
        norm_type="scalenorm",
        shift_tokens=True,
    ):
        super().__init__()
        assert norm_type in ("scalenorm", "layernorm")
        norm_klass = ScaleNorm if norm_type == "scalenorm" else nn.LayerNorm
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))
        self.fsmn = nn.ModuleList(
            [Gated_FSMN_Block_Dilated(dim) for _ in range(depth)]
        )
        self.layers = nn.ModuleList(
            [
                FLASH_ShareA_FFConvM(
                    dim=dim,
                    group_size=group_size,
                    query_key_dim=query_key_dim,
                    expansion_factor=expansion_factor,
                    causal=causal,
                    dropout=attn_dropout,
                    rotary_pos_emb=rotary_pos_emb,
                    norm_klass=norm_klass,
                    shift_tokens=shift_tokens,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, *, mask=None):
        for i, flash in enumerate(self.layers):
            x = flash(x, mask=mask)
            x = self.fsmn[i](x)
        return x


class TransformerEncoder_FLASH_DualA_FSMN(nn.Module):
    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        attention_type="regularMHA",
    ):
        super().__init__()
        self.flashT = FLASHTransformer_DualA_FSMN(dim=d_model, depth=num_layers)
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        pos_embs=None,
    ):
        output = self.flashT(src)
        output = self.norm(output)
        return output


class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device=device).type_as(self.inv_freq)
        sinu = einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        return emb * self.scale


class Linear(torch.nn.Module):
    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        bias=True,
        combine_dims=False,
    ):
        super().__init__()
        self.combine_dims = combine_dims
        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")
        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        return self.w(x)


class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim, elementwise_affine=True):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = super().forward(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            x = super().forward(x)
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        x = self.conv1d(x)
        x = F.relu(x)
        return x


class Decoder(nn.ConvTranspose1d):
    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError("Decoder accepts 2-D or 3-D input")
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if x.size(1) == 1:
            x = x.squeeze(1)
        return x


class SBFLASHBlock_DualA(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        input_shape=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
        attention_type="regularMHA",
    ):
        super().__init__()
        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")
        self.mdl = TransformerEncoder_FLASH_DualA_FSMN(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            attention_type=attention_type,
        )

    def forward(self, x):
        return self.mdl(x)


class Dual_Computation_Block(nn.Module):
    def __init__(
        self,
        intra_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
    ):
        super().__init__()
        self.intra_mdl = intra_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)
        if linear_layer_after_inter_intra:
            self.intra_linear = Linear(out_channels, input_size=out_channels)

    def forward(self, x):
        intra = x.permute(0, 2, 1).contiguous()
        intra = self.intra_mdl(intra)
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)
        intra = intra.permute(0, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)
        if self.skip_around_intra:
            intra = intra + x
        return intra


class Dual_Path_Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super().__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.dual_mdl = nn.ModuleList([])
        for _ in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.conv1d_out = nn.Conv1d(out_channels, out_channels * num_spks, 1)
        self.conv1_decoder = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1d_encoder(x)

        if self.use_global_pos_enc:
            base = x
            x = x.transpose(1, -1)
            emb = self.pos_enc(x)
            emb = emb.transpose(0, -1)
            x = base + emb

        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        x = self.prelu(x)
        x = self.conv1d_out(x)
        B, _, S = x.shape
        x = x.view(B * self.num_spks, -1, S)
        x = self.output(x) * self.output_gate(x)
        x = self.conv1_decoder(x)
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        x = x.transpose(0, 1)
        return x


class MossFormer2Separator(AbsSeparator):
    """Official MossFormer2 architecture adapted to ESPnet's separator API.

    The implementation follows the public MossFormer2 standalone model layout:
    Encoder -> Dual_Path_Model(SBFLASHBlock_DualA/FLASH+FSMN) -> Decoder.
    Width/depth can be reduced by config for size-matched experiments.
    """

    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        predict_noise: bool = False,
        encoder_kernel_size: int = 16,
        encoder_out_nchannels: int = 160,
        encoder_in_nchannels: int = 1,
        masknet_chunksize: int = 250,
        masknet_numlayers: int = 1,
        masknet_norm: str = "ln",
        masknet_useextralinearlayer: bool = False,
        masknet_extraskipconnection: bool = True,
        masknet_use_global_pos_enc: bool = True,
        intra_numlayers: int = 4,
        intra_nhead: int = 4,
        intra_dffn: int = 320,
        intra_dropout: float = 0.0,
        intra_use_positional: bool = True,
        intra_norm_before: bool = True,
    ):
        super().__init__()
        if input_dim != 1:
            raise ValueError(
                "MossFormer2Separator expects waveform input with encoder: same "
                f"(input_dim=1), but got input_dim={input_dim}."
            )
        if predict_noise:
            raise NotImplementedError("MossFormer2Separator does not support noise output.")
        self._num_spk = num_spk
        self.predict_noise = predict_noise
        self.encoder = Encoder(
            kernel_size=encoder_kernel_size,
            out_channels=encoder_out_nchannels,
            in_channels=encoder_in_nchannels,
        )
        intra_model = SBFLASHBlock_DualA(
            num_layers=intra_numlayers,
            d_model=encoder_out_nchannels,
            nhead=intra_nhead,
            d_ffn=intra_dffn,
            dropout=intra_dropout,
            use_positional_encoding=intra_use_positional,
            norm_before=intra_norm_before,
        )
        self.masknet = Dual_Path_Model(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_out_nchannels,
            intra_model=intra_model,
            num_layers=masknet_numlayers,
            norm=masknet_norm,
            K=masknet_chunksize,
            num_spks=self.num_spk,
            skip_around_intra=masknet_extraskipconnection,
            linear_layer_after_inter_intra=masknet_useextralinearlayer,
            use_global_pos_enc=masknet_use_global_pos_enc,
        )
        self.decoder = Decoder(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_in_nchannels,
            kernel_size=encoder_kernel_size,
            stride=encoder_kernel_size // 2,
            bias=False,
        )

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Dict = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Dict]:
        if input.dim() == 3:
            if input.size(-1) != 1:
                raise ValueError(
                    "MossFormer2Separator expects input shape (B, T) or (B, T, 1), "
                    f"but got {tuple(input.shape)}."
                )
            input = input.squeeze(-1)
        elif input.dim() != 2:
            raise ValueError(
                "MossFormer2Separator expects input shape (B, T) or (B, T, 1), "
                f"but got {tuple(input.shape)}."
            )

        mix_w = self.encoder(input)
        est_mask = self.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.num_spk)
        sep_h = mix_w * est_mask

        est_source = torch.cat(
            [self.decoder(sep_h[i]).unsqueeze(-1) for i in range(self.num_spk)],
            dim=-1,
        )

        t_origin = input.size(1)
        t_est = est_source.size(1)
        if t_origin > t_est:
            est_source = F.pad(est_source, (0, 0, 0, t_origin - t_est))
        else:
            est_source = est_source[:, :t_origin, :]

        separated = [est_source[..., spk] for spk in range(self.num_spk)]
        return separated, ilens, {}

    @property
    def num_spk(self):
        return self._num_spk

    def forward_streaming(self, input_frame: torch.Tensor):
        raise NotImplementedError("MossFormer2Separator does not support streaming.")
