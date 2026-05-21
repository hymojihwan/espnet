#!/usr/bin/env python3
import argparse
from pathlib import Path


def read_kaldi_text(path: Path):
    data = {}
    if not path.is_file():
        return data
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                data[parts[0]] = []
            else:
                data[parts[0]] = parts[1:]
    return data


def read_utt2num_samples(path: Path):
    data = {}
    if not path.is_file():
        return data
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, num = line.split()[:2]
            data[utt] = int(num)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare token_int/token_time_ms from ASR decode for SE training"
    )
    parser.add_argument(
        "--asr_decode_dir",
        type=str,
        required=True,
        help="ASR decode dir containing 1best_recog/token_int and token_time_ms",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Target data dir to write token_int/token_time_ms",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=16000,
        help="Sampling rate for fallback time generation",
    )
    args = parser.parse_args()

    asr_decode_dir = Path(args.asr_decode_dir)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    token_int_path = asr_decode_dir / "1best_recog" / "token_int"
    token_time_path = asr_decode_dir / "1best_recog" / "token_time_ms"
    token_chunk_path = asr_decode_dir / "1best_recog" / "token_chunk_int"

    token_int = read_kaldi_text(token_int_path)
    token_time = read_kaldi_text(token_time_path)
    token_chunk = read_kaldi_text(token_chunk_path)
    utt2num_samples = read_utt2num_samples(data_dir / "utt2num_samples")

    out_token_int = data_dir / "token_int"
    out_token_time = data_dir / "token_time_ms"
    out_token_chunk = data_dir / "token_chunk_int"

    with out_token_int.open("w", encoding="utf-8") as f_int, out_token_time.open(
        "w", encoding="utf-8"
    ) as f_time, out_token_chunk.open("w", encoding="utf-8") as f_chunk:
        for utt, toks in token_int.items():
            f_int.write(f"{utt} {' '.join(toks)}\n")

            if utt in token_chunk:
                chunks = token_chunk[utt]
                f_chunk.write(f"{utt} {' '.join(chunks)}\n")
            else:
                f_chunk.write(f"{utt}\n")

            if utt in token_time:
                times = token_time[utt]
                f_time.write(f"{utt} {' '.join(times)}\n")
                continue

            # Fallback: uniform timing across utterance duration
            if utt in utt2num_samples and len(toks) > 0:
                total_ms = (utt2num_samples[utt] / float(args.fs)) * 1000.0
                step = total_ms / float(len(toks))
                times = [f"{(i + 1) * step:.1f}" for i in range(len(toks))]
                f_time.write(f"{utt} {' '.join(times)}\n")
            else:
                f_time.write(f"{utt}\n")


if __name__ == "__main__":
    main()
