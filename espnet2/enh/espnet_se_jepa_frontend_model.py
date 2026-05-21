"""SE->ASR encoder JEPA-style frontend model.

Train only a lightweight masked predictor in ASR encoder feature space
with frozen pretrained SE and ASR models.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.diar.layers.abs_mask import AbsMask
from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetSEJEPAFrontendModel(ESPnetEnhancementModel):
    """SE + frozen ASR encoder JEPA-style masked feature prediction."""

    @typechecked
    def __init__(
        self,
        encoder: AbsEncoder,
        separator: Optional[AbsSeparator],
        decoder: AbsDecoder,
        mask_module: Optional[AbsMask],
        loss_wrappers: Optional[List[AbsLossWrapper]],
        use_asr_jepa_frontend: bool = False,
        se_train_config: Optional[str] = None,
        se_model_file: Optional[str] = None,
        asr_train_config: Optional[str] = None,
        asr_model_file: Optional[str] = None,
        pred_l1_weight: float = 1.0,
        pred_cos_weight: float = 0.5,
        var_weight: float = 0.01,
        mask_prob: float = 0.3,
        predictor_hidden_dim: int = 512,
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

        if not use_asr_jepa_frontend:
            raise ValueError("use_asr_jepa_frontend must be True for this model")
        if se_train_config is None or se_model_file is None:
            raise ValueError("se_train_config and se_model_file are required")
        if asr_train_config is None or asr_model_file is None:
            raise ValueError("asr_train_config and asr_model_file are required")

        self.pred_l1_weight = float(pred_l1_weight)
        self.pred_cos_weight = float(pred_cos_weight)
        self.var_weight = float(var_weight)
        self.mask_prob = float(mask_prob)

        # Not used in forward path; freeze to avoid DDP unused parameter issues.
        for mod in (self.encoder, self.separator, self.decoder, self.mask_module):
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False

        # Local imports to avoid circular imports with task modules.
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
        hid = int(predictor_hidden_dim)
        self.mask_token = torch.nn.Parameter(torch.zeros(feat_dim))
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, feat_dim),
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

    def _build_valid_mask(self, lengths: torch.Tensor, max_t: int, dtype) -> torch.Tensor:
        b = lengths.size(0)
        frame_idx = torch.arange(max_t, device=lengths.device).unsqueeze(0).expand(b, -1)
        return (frame_idx < lengths.unsqueeze(1)).to(dtype)

    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        additional: Optional[Dict] = None,
        fs: Optional[int] = None,
    ):
        # Reuse pretrained SE waveform path for enhancement inference.
        with torch.no_grad():
            return self.se_model.forward_enhance(
                speech_mix=speech_mix,
                speech_lengths=speech_lengths,
                additional=additional if additional is not None else {},
                fs=fs,
            )

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
        ).long()

        max_samples = int(speech_lengths.max().item())
        speech_noisy = speech_mix[:, :max_samples]
        speech_clean = kwargs["speech_ref1"][:, :max_samples]

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

        valid = self._build_valid_mask(min_len, max_t, h_enh.dtype).unsqueeze(-1)
        bern = torch.rand((batch_size, max_t, 1), device=h_enh.device, dtype=h_enh.dtype)
        pred_mask = (bern < self.mask_prob).to(h_enh.dtype) * valid

        masked_input = h_enh * (1.0 - pred_mask) + self.mask_token.view(1, 1, -1) * pred_mask
        h_pred = self.predictor(masked_input)

        # JEPA-style prediction loss only on masked valid positions.
        denom = pred_mask.sum().clamp_min(1.0)
        l1 = torch.abs(h_pred - h_clean).mul(pred_mask).sum() / denom

        cos = 1.0 - F.cosine_similarity(h_pred, h_clean, dim=-1)
        cos = (cos.unsqueeze(-1) * pred_mask).sum() / denom

        # Anti-collapse variance penalty on predicted valid frames.
        valid_pred = h_pred * valid
        denom_valid = valid.sum().clamp_min(1.0)
        mean = valid_pred.sum(dim=(0, 1)) / denom_valid
        var = ((valid_pred - mean.view(1, 1, -1)) ** 2 * valid).sum(dim=(0, 1)) / denom_valid
        std = torch.sqrt(var + 1.0e-4)
        var_loss = torch.relu(1.0 - std).mean()

        loss = (
            self.pred_l1_weight * l1
            + self.pred_cos_weight * cos
            + self.var_weight * var_loss
        )

        stats = {
            "loss": loss.detach(),
            "loss_pred_l1": l1.detach(),
            "loss_pred_cos": cos.detach(),
            "loss_var": var_loss.detach(),
            "mask_ratio": pred_mask.mean().detach(),
        }
        weight = torch.tensor(batch_size, dtype=loss.dtype, device=loss.device)
        loss, stats, weight = force_gatherable((loss, stats, weight), loss.device)
        return loss, stats, weight
