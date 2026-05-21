#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import subprocess
import os
import shutil

import kaldiio
import soundfile as sf
import torch
import yaml
from torch_complex.tensor import ComplexTensor

from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_logmel_mapper import CleanMelMapper


def read_scp(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            utt, value = line.rstrip().split(maxsplit=1)
            data[utt] = value
    return data


class FeatureExtractor(torch.nn.Module):
    def __init__(self, fs=16000, n_fft=512, hop_length=128, win_length=512, n_mels=80):
        super().__init__()
        self.stft = Stft(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.logmel = LogMel(fs=fs, n_fft=n_fft, n_mels=n_mels)

    def forward(self, wav, lengths):
        spec, spec_lens = self.stft(wav, lengths)
        spec = ComplexTensor(spec[..., 0], spec[..., 1])
        power = spec.real ** 2 + spec.imag ** 2
        return self.logmel(power, spec_lens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--clean_wav_scp", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--utt2spk", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    feat_extractor = FeatureExtractor(
        fs=conf["fs"], n_fft=conf["n_fft"], hop_length=conf["hop_length"], win_length=conf["win_length"], n_mels=conf["n_mels"]
    )
    model = CleanMelMapper(
        output_dim=conf["n_mels"],
        hidden_dim=conf["hidden_dim"],
        num_blocks=conf["num_layers"],
        dropout=conf["dropout"],
        time_kernel_size=conf.get("time_kernel_size", 5),
        freq_kernel_size=conf.get("freq_kernel_size", 5),
        mapper_mode=conf.get("mapper_mode", "map"),
    )
    state_dict = ckpt["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    wavs = read_scp(args.wav_scp)
    clean_wavs = read_scp(args.clean_wav_scp) if args.clean_wav_scp else None
    ark_path = outdir / "feats.ark"
    scp_path = outdir / "feats.scp"
    tmp_ark_path = outdir / "feats.ark.tmp"
    tmp_scp_path = outdir / "feats.scp.tmp"
    if tmp_ark_path.exists():
        tmp_ark_path.unlink()
    if tmp_scp_path.exists():
        tmp_scp_path.unlink()

    clean_ark_path = outdir / "clean_speech.ark"
    clean_scp_path = outdir / "clean_speech.scp"
    tmp_clean_ark_path = outdir / "clean_speech.ark.tmp"
    tmp_clean_scp_path = outdir / "clean_speech.scp.tmp"
    if tmp_clean_ark_path.exists():
        tmp_clean_ark_path.unlink()
    if tmp_clean_scp_path.exists():
        tmp_clean_scp_path.unlink()

    with kaldiio.WriteHelper(f"ark,scp:{tmp_ark_path},{tmp_scp_path}") as writer:
        clean_writer = None
        if clean_wavs is not None:
            clean_writer = kaldiio.WriteHelper(
                f"ark,scp:{tmp_clean_ark_path},{tmp_clean_scp_path}"
            )
            clean_writer.__enter__()

        try:
            for utt, wav_path in wavs.items():
                wav, _ = sf.read(wav_path, dtype="float32")
                wav = torch.from_numpy(wav).float().unsqueeze(0)
                lengths = torch.tensor([wav.size(1)], dtype=torch.long)
                with torch.no_grad():
                    noisy_mel, noisy_mel_lens = feat_extractor(wav, lengths)
                    pred, pred_lens = model(noisy_mel, noisy_mel_lens)
                    feat_len = int(torch.minimum(pred_lens, noisy_mel_lens)[0].item())
                    pred = pred[0, :feat_len].cpu().numpy()
                writer(utt, pred)

                if clean_writer is not None and utt in clean_wavs:
                    clean_wav, _ = sf.read(clean_wavs[utt], dtype="float32")
                    clean_wav = torch.from_numpy(clean_wav).float().unsqueeze(0)
                    clean_lengths = torch.tensor([clean_wav.size(1)], dtype=torch.long)
                    with torch.no_grad():
                        clean_feat, clean_feat_lens = feat_extractor(clean_wav, clean_lengths)
                        clean_len = int(clean_feat_lens[0].item())
                        clean_feat = clean_feat[0, :clean_len].cpu().numpy()
                    clean_writer(utt, clean_feat)
        finally:
            if clean_writer is not None:
                clean_writer.__exit__(None, None, None)

    os.replace(tmp_ark_path, ark_path)
    os.replace(tmp_scp_path, scp_path)
    scp_text = scp_path.read_text(encoding="utf-8")
    scp_text = scp_text.replace(f"{tmp_ark_path.name}:", f"{ark_path.name}:")
    scp_path.write_text(scp_text, encoding="utf-8")

    if clean_wavs is not None and tmp_clean_scp_path.exists():
        os.replace(tmp_clean_ark_path, clean_ark_path)
        os.replace(tmp_clean_scp_path, clean_scp_path)
        clean_scp_text = clean_scp_path.read_text(encoding="utf-8")
        clean_scp_text = clean_scp_text.replace(
            f"{tmp_clean_ark_path.name}:", f"{clean_ark_path.name}:"
        )
        clean_scp_path.write_text(clean_scp_text, encoding="utf-8")

    for name in ("text", "utt2spk"):
        src = Path(getattr(args, name))
        dst = outdir / name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    spk2utt_tool = Path(__file__).resolve().parents[2] / "asr1" / "utils" / "utt2spk_to_spk2utt.pl"
    spk2utt = subprocess.check_output(
        [str(spk2utt_tool), str(outdir / "utt2spk")],
        text=True,
    )
    (outdir / "spk2utt").write_text(spk2utt, encoding="utf-8")


if __name__ == "__main__":
    main()
