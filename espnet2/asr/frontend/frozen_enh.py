from contextlib import nullcontext
from typing import Optional, Tuple, Union

import humanfriendly
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft


class FrozenEnhFrontend(AbsFrontend):
    """Frozen enhancement frontend for ASR transducer.

    Waveform -> frozen enhancement model -> STFT -> log-mel.
    """

    @typechecked
    def __init__(
        self,
        enh_train_config: str,
        enh_model_file: str,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: Optional[int] = None,
        fmax: Optional[int] = None,
        htk: bool = False,
        enh_no_grad: bool = True,
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
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            window=window,
            normalized=normalized,
            onesided=onesided,
        )
        # Keep explicit hop length for downstream latency utilities.
        self.hop_length = hop_length
        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels

    def output_size(self) -> int:
        return self.n_mels

    def _compute_stft(
        self, input_wav: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[ComplexTensor, torch.Tensor]:
        input_stft, feats_lens = self.stft(input_wav, input_lengths)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input.dim() == 3 and input.size(1) == 1:
            input = input.squeeze(1)

        ctx = torch.no_grad() if self.enh_no_grad else nullcontext()
        with ctx:
            feature_mix, flens = self.enh_model.encoder(input, input_lengths)
            feature_pre, _, _ = self.enh_model.separator(feature_mix, flens, {})

            if not isinstance(feature_pre, (list, tuple)):
                feature_pre = [feature_pre]

            speech_pre = [
                self.enh_model.decoder(ps, input_lengths)[0] for ps in feature_pre
            ]

        enhanced = speech_pre[0]
        if enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)
        elif enhanced.dim() == 3:
            enhanced = enhanced[..., 0]

        max_len = enhanced.size(-1)
        speech_lengths = input_lengths.to(device=enhanced.device, dtype=torch.long)
        speech_lengths = torch.clamp(speech_lengths, max=max_len)

        enhanced_stft, feats_lens = self._compute_stft(enhanced, speech_lengths)
        enhanced_power = enhanced_stft.real**2 + enhanced_stft.imag**2
        feats, _ = self.logmel(enhanced_power, feats_lens)

        return feats, feats_lens
