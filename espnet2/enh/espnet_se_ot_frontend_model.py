"""SE->ASR encoder OT restoration model.

Train only a small correction frontend with frozen pretrained SE and ASR encoder.
"""

from typing import Dict, List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.diar.layers.abs_mask import AbsMask
from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetSEOTFrontendModel(ESPnetEnhancementModel):
    """SE + frozen ASR encoder feature restoration with OT/L1 losses."""

    @typechecked
    def __init__(
        self,
        encoder: AbsEncoder,
        separator: Optional[AbsSeparator],
        decoder: AbsDecoder,
        mask_module: Optional[AbsMask],
        loss_wrappers: Optional[List[AbsLossWrapper]],
        use_asr_ot_frontend: bool = False,
        se_train_config: Optional[str] = None,
        se_model_file: Optional[str] = None,
        asr_train_config: Optional[str] = None,
        asr_model_file: Optional[str] = None,
        ot_weight: float = 1.0,
        l1_weight: float = 1.0,
        mask_l1_weight: float = 1.0e-4,
        ot_epsilon: float = 0.05,
        ot_num_iters: int = 15,
        correction_hidden_dim: int = 512,
        **kwargs,
    ):
        super().__init__(
            encoder=encoder,
            separator=separator,
            decoder=decoder,
            mask_module=mask_module,
            loss_wrappers=loss_wrappers,
            **kwargs,
        )

        if not use_asr_ot_frontend:
            raise ValueError("use_asr_ot_frontend must be True for this model")
        if se_train_config is None or se_model_file is None:
            raise ValueError("se_train_config and se_model_file are required")
        if asr_train_config is None or asr_model_file is None:
            raise ValueError("asr_train_config and asr_model_file are required")

        self.ot_weight = float(ot_weight)
        self.l1_weight = float(l1_weight)
        self.mask_l1_weight = float(mask_l1_weight)
        self.ot_epsilon = float(ot_epsilon)
        self.ot_num_iters = int(ot_num_iters)

        # This model does not use base enhancement modules in forward.
        # Freeze them to avoid DDP "unused parameter" errors.
        for mod in (self.encoder, self.separator, self.decoder, self.mask_module):
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False

        # Local imports to avoid circular import with task modules.
        from espnet2.tasks.asr_transducer import ASRTransducerTask
        from espnet2.tasks.enh import EnhancementTask

        self.se_model, _ = EnhancementTask.build_model_from_file(
            config_file=se_train_config,
            model_file=se_model_file,
            device="cpu",
        )
        self.asr_model, _ = ASRTransducerTask.build_model_from_file(
            config_file=asr_train_config,
            model_file=asr_model_file,
            device="cpu",
        )
        self.se_model.eval()
        self.asr_model.eval()
        for p in self.se_model.parameters():
            p.requires_grad = False
        for p in self.asr_model.parameters():
            p.requires_grad = False

        feat_dim = self._get_asr_encoder_dim()
        hid = int(correction_hidden_dim)
        self.delta_net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, feat_dim),
        )
        self.mask_net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, feat_dim),
            torch.nn.Sigmoid(),
        )

    def _get_asr_encoder_dim(self) -> int:
        if hasattr(self.asr_model.encoder, "output_size"):
            out = self.asr_model.encoder.output_size
            return int(out() if callable(out) else out)
        if hasattr(self.asr_model.joint_network, "lin_enc"):
            return int(self.asr_model.joint_network.lin_enc.in_features)
        raise RuntimeError("Could not infer ASR encoder output feature dim")

    def _asr_encode(
        self, speech: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            h, h_len = self.asr_model.encode(speech, lengths)
        return h, h_len

    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        additional: Optional[Dict] = None,
        fs: Optional[int] = None,
    ):
        # For inference/decoding paths, always use frozen pretrained SE frontend.
        with torch.no_grad():
            return self.se_model.forward_enhance(
                speech_mix=speech_mix,
                speech_lengths=speech_lengths,
                additional=additional if additional is not None else {},
                fs=fs,
            )

    def _masked_l1(
        self, x: torch.Tensor, y: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        b, t, _ = x.shape
        frame_idx = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        mask = (frame_idx < lengths.unsqueeze(1)).unsqueeze(-1).to(x.dtype)
        denom = mask.sum().clamp_min(1.0)
        return torch.abs(x - y).mul(mask).sum() / denom

    def _sinkhorn_ot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ti = x.size(0)
        tj = y.size(0)
        if ti == 0 or tj == 0:
            return x.new_tensor(0.0)

        c = torch.cdist(x, y, p=2).pow(2)
        a = torch.full((ti,), 1.0 / float(ti), device=x.device, dtype=x.dtype)
        b = torch.full((tj,), 1.0 / float(tj), device=x.device, dtype=x.dtype)
        k = torch.exp(-c / max(self.ot_epsilon, 1.0e-6)).clamp_min(1.0e-12)
        u = torch.full_like(a, 1.0 / float(ti))
        v = torch.full_like(b, 1.0 / float(tj))

        for _ in range(self.ot_num_iters):
            kv = torch.matmul(k, v).clamp_min(1.0e-12)
            u = a / kv
            ktu = torch.matmul(k.transpose(0, 1), u).clamp_min(1.0e-12)
            v = b / ktu

        p = u.unsqueeze(1) * k * v.unsqueeze(0)
        return (p * c).sum()

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ):
        assert "speech_ref1" in kwargs, "speech_ref1 (clean) is required"

        batch_size = speech_mix.size(0)
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else speech_mix.new_full((batch_size,), speech_mix.size(1), dtype=torch.long)
        )
        speech_lengths = speech_lengths.long()
        speech_clean = kwargs["speech_ref1"][:, : speech_lengths.max()]
        speech_noisy = speech_mix[:, : speech_lengths.max()]

        with torch.no_grad():
            se_out, _, _, _ = self.se_model.forward_enhance(
                speech_noisy, speech_lengths, additional={}, fs=None
            )
            if se_out is None:
                raise RuntimeError("Frozen SE model did not produce waveform output")
            speech_enh = se_out[0]

        h_enh, h_enh_len = self._asr_encode(speech_enh, speech_lengths)
        h_clean, h_clean_len = self._asr_encode(speech_clean, speech_lengths)

        min_len = torch.minimum(h_enh_len, h_clean_len)
        max_t = int(min_len.max().item())
        h_enh = h_enh[:, :max_t]
        h_clean = h_clean[:, :max_t]

        delta = self.delta_net(h_enh)
        mask = self.mask_net(h_enh)
        h_corr = h_enh + mask * delta

        l1 = self._masked_l1(h_corr, h_clean, min_len)

        ot_vals = []
        for b in range(h_corr.size(0)):
            t = int(min_len[b].item())
            ot_vals.append(self._sinkhorn_ot(h_corr[b, :t], h_clean[b, :t]))
        ot = torch.stack(ot_vals).mean() if ot_vals else h_corr.new_tensor(0.0)

        b, t, d = mask.shape
        frame_idx = torch.arange(t, device=mask.device).unsqueeze(0).expand(b, -1)
        valid = (frame_idx < min_len.unsqueeze(1)).unsqueeze(-1).to(mask.dtype)
        mask_l1 = torch.abs(mask).mul(valid).sum() / valid.sum().clamp_min(1.0)

        loss = self.ot_weight * ot + self.l1_weight * l1 + self.mask_l1_weight * mask_l1
        stats = {
            "loss": loss.detach(),
            "loss_ot": ot.detach(),
            "loss_l1": l1.detach(),
            "loss_mask_l1": mask_l1.detach(),
        }
        weight = torch.tensor(batch_size, dtype=loss.dtype, device=loss.device)
        loss, stats, weight = force_gatherable((loss, stats, weight), loss.device)
        return loss, stats, weight
