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
        output_waveform: bool = False,
        observation_addition: bool = False,
        observation_addition_domain: str = "waveform",
        enhanced_weight: float = 0.8,
        noisy_weight: float = 0.2,
        rms_alignment_eps: float = 1.0e-8,
        train_zero_mask_ratio: float = 0.0,
        train_zero_mask_patch_time: int = 10,
        train_zero_mask_patch_freq: int = 80,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if enhanced_weight < 0.0 or noisy_weight < 0.0:
            raise ValueError("Observation-addition weights must be non-negative")
        if observation_addition and enhanced_weight + noisy_weight <= 0.0:
            raise ValueError("At least one observation-addition weight must be positive")
        if observation_addition_domain not in ("waveform", "power"):
            raise ValueError(
                "observation_addition_domain must be 'waveform' or 'power'"
            )
        if (
            observation_addition
            and observation_addition_domain == "power"
            and output_waveform
        ):
            raise ValueError("Power-domain observation addition cannot output waveform")
        if not 0.0 <= train_zero_mask_ratio <= 1.0:
            raise ValueError("train_zero_mask_ratio must be between 0.0 and 1.0")
        if train_zero_mask_patch_time <= 0 or train_zero_mask_patch_freq <= 0:
            raise ValueError("Train zero-mask patch dimensions must be positive")
        if output_waveform and train_zero_mask_ratio > 0.0:
            raise ValueError("Train zero-mask patching requires log-mel output")

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
        self.output_waveform = output_waveform
        self.observation_addition = observation_addition
        self.observation_addition_domain = observation_addition_domain
        weight_sum = enhanced_weight + noisy_weight
        self.enhanced_weight = enhanced_weight / weight_sum
        self.noisy_weight = noisy_weight / weight_sum
        self.rms_alignment_eps = rms_alignment_eps
        self.train_zero_mask_ratio = train_zero_mask_ratio
        self.train_zero_mask_patch_time = train_zero_mask_patch_time
        self.train_zero_mask_patch_freq = train_zero_mask_patch_freq

    def output_size(self) -> int:
        if self.output_waveform:
            return 1
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

            if (
                self.observation_addition
                and self.observation_addition_domain == "power"
            ):
                enhanced_power = self._complex_power(feature_pre[0])
                noisy_power = self._complex_power(feature_mix)
                if enhanced_power.shape != noisy_power.shape:
                    raise ValueError(
                        "Enhanced and noisy spectra must match for power-domain "
                        f"observation addition: enhanced={tuple(enhanced_power.shape)}, "
                        f"noisy={tuple(noisy_power.shape)}"
                    )
                observation_added_power = (
                    self.enhanced_weight * enhanced_power
                    + self.noisy_weight * noisy_power
                )
                feats, _ = self.logmel(observation_added_power, flens)
                feats = self._apply_train_zero_patch_mask(feats, flens)
                return feats, flens

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

        if self.observation_addition:
            noisy = input[..., :max_len]
            sample_ids = torch.arange(max_len, device=enhanced.device)
            valid_mask = sample_ids.unsqueeze(0) < speech_lengths.unsqueeze(1)
            valid_mask = valid_mask.to(enhanced.dtype)
            num_samples = speech_lengths.clamp_min(1).to(enhanced.dtype).unsqueeze(1)
            enhanced_rms = torch.sqrt(
                (enhanced.square() * valid_mask).sum(dim=1, keepdim=True)
                / num_samples
                + self.rms_alignment_eps
            )
            noisy_rms = torch.sqrt(
                (noisy.square() * valid_mask).sum(dim=1, keepdim=True)
                / num_samples
                + self.rms_alignment_eps
            )
            enhanced = enhanced * (noisy_rms / enhanced_rms)
            enhanced = (
                self.enhanced_weight * enhanced + self.noisy_weight * noisy
            ) * valid_mask

        if self.output_waveform:
            return enhanced, speech_lengths

        enhanced_stft, feats_lens = self._compute_stft(enhanced, speech_lengths)
        enhanced_power = enhanced_stft.real**2 + enhanced_stft.imag**2
        feats, _ = self.logmel(enhanced_power, feats_lens)
        feats = self._apply_train_zero_patch_mask(feats, feats_lens)

        return feats, feats_lens

    def _apply_train_zero_patch_mask(
        self, feats: torch.Tensor, feats_lens: torch.Tensor
    ) -> torch.Tensor:
        if not self.training or self.train_zero_mask_ratio <= 0.0:
            return feats

        batch_size, max_frames, num_bins = feats.shape
        patch_time = self.train_zero_mask_patch_time
        patch_freq = self.train_zero_mask_patch_freq
        num_time_patches = (max_frames + patch_time - 1) // patch_time
        num_freq_patches = (num_bins + patch_freq - 1) // patch_freq
        masked_feats = feats.clone()

        for batch_index in range(batch_size):
            valid_frames = min(int(feats_lens[batch_index].item()), max_frames)
            valid_time_patches = (valid_frames + patch_time - 1) // patch_time
            num_valid_patches = valid_time_patches * num_freq_patches
            num_masked = int(num_valid_patches * self.train_zero_mask_ratio)
            if num_masked <= 0:
                continue

            selected = torch.randperm(
                num_valid_patches, device=feats.device
            )[:num_masked]
            patch_mask = torch.zeros(
                num_time_patches,
                num_freq_patches,
                dtype=torch.bool,
                device=feats.device,
            )
            selected_time = torch.div(
                selected, num_freq_patches, rounding_mode="floor"
            )
            selected_freq = selected.remainder(num_freq_patches)
            patch_mask[selected_time, selected_freq] = True
            frame_mask = patch_mask.repeat_interleave(
                patch_time, dim=0
            ).repeat_interleave(patch_freq, dim=1)
            frame_mask = frame_mask[:max_frames, :num_bins]
            masked_feats[batch_index].masked_fill_(frame_mask, 0.0)

        return masked_feats

    @staticmethod
    def _complex_power(input_spectrum: torch.Tensor) -> torch.Tensor:
        if isinstance(input_spectrum, ComplexTensor):
            return input_spectrum.real.square() + input_spectrum.imag.square()
        if torch.is_complex(input_spectrum):
            return input_spectrum.real.square() + input_spectrum.imag.square()
        raise TypeError(
            "Power-domain observation addition requires a complex spectrum, "
            f"got dtype={input_spectrum.dtype}"
        )
