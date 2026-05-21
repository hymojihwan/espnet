#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    try:
        from scipy.signal import resample_poly  # type: ignore

        g = np.gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
        return resample_poly(audio, up, down).astype(np.float32)
    except Exception:
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        n_new = int(round(len(audio) * float(dst_sr) / float(src_sr)))
        x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def pick_noise_segment(noise: np.ndarray, target_len: int, rng: random.Random) -> np.ndarray:
    if len(noise) < target_len:
        rep = int(np.ceil(target_len / max(1, len(noise))))
        noise = np.tile(noise, rep)
    start_max = len(noise) - target_len
    start = rng.randint(0, start_max) if start_max > 0 else 0
    return noise[start : start + target_len]


def mix_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / max(noise_rms, 1e-12)
    mixed = clean + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.999:
        mixed = mixed / peak * 0.999
    return mixed.astype(np.float32)


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), int(sr)


def copy_transcripts(clean_split_dir: Path, noisy_split_dir: Path) -> None:
    for trans in clean_split_dir.rglob("*.trans.txt"):
        rel = trans.relative_to(clean_split_dir)
        dst = noisy_split_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trans, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_root", type=Path, required=True)
    parser.add_argument("--noise_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--mode", choices=["random", "fixed"], required=True)
    parser.add_argument("--snr_min", type=float, default=-5.0)
    parser.add_argument("--snr_max", type=float, default=15.0)
    parser.add_argument("--snr_value", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    clean_split_dir = args.clean_root / args.split
    noisy_split_dir = args.out_root / args.split
    noisy_split_dir.mkdir(parents=True, exist_ok=True)

    noise_files = sorted(
        [p for p in args.noise_root.rglob("*") if p.suffix.lower() in {".wav", ".flac"}]
    )
    if not noise_files:
        raise RuntimeError(f"No noise files found: {args.noise_root}")

    flac_files = sorted(clean_split_dir.rglob("*.flac"))
    if not flac_files:
        raise RuntimeError(f"No flac files found: {clean_split_dir}")

    copy_transcripts(clean_split_dir, noisy_split_dir)

    for idx, clean_path in enumerate(flac_files, 1):
        clean, sr = read_mono(clean_path)
        noise_path = noise_files[rng.randrange(len(noise_files))]
        noise, noise_sr = read_mono(noise_path)
        noise = resample_if_needed(noise, noise_sr, sr)
        noise = pick_noise_segment(noise, len(clean), rng)

        if args.mode == "random":
            snr_db = rng.uniform(args.snr_min, args.snr_max)
        else:
            snr_db = args.snr_value

        mixed = mix_with_snr(clean, noise, snr_db)
        dst = noisy_split_dir / clean_path.relative_to(clean_split_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), mixed, sr, format="FLAC", subtype="PCM_16")

        if idx % 1000 == 0:
            print(f"[{args.split}] {idx}/{len(flac_files)} done")

    print(f"[{args.split}] completed: {len(flac_files)} utterances")


if __name__ == "__main__":
    main()
