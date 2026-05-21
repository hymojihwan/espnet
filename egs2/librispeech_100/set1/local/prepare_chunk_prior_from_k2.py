#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from espnet2.fileio.read_text import read_2columns_text
from espnet2.fileio.sound_scp import SoundScp
from espnet2.tasks.asr import ASRTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.asr_transducer.utils import get_transducer_task_io


def _get_chunk_frames(model, chunk_samples: int) -> int:
    hop_length = getattr(getattr(model, "frontend", None), "hop_length", 160)
    subsampling = getattr(getattr(getattr(model, "encoder", None), "embed", None), "subsampling_factor", 1)
    denom = hop_length * max(subsampling, 1)
    return max(1, int(round(chunk_samples / float(denom))))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare chunk-level prior from k2 RNNT forward-backward"
    )
    parser.add_argument("--asr_train_config", type=str, required=True)
    parser.add_argument("--asr_model_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--chunk_length", type=int, default=10240)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--blank_id", type=int, default=0)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    device = args.device
    model, asr_train_args = ASRTask.build_model_from_file(
        config_file=args.asr_train_config, model_file=args.asr_model_file, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if not getattr(model, "use_k2_pruned_loss", False):
        logging.warning(
            "ASR model is not configured with use_k2_pruned_loss=True. "
            "This script requires k2 components (am_proj/lm_proj)."
        )

    token_list = model.token_list
    token_type = getattr(asr_train_args, "token_type", None)
    bpemodel = getattr(asr_train_args, "bpemodel", None)
    tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
    converter = TokenIDConverter(token_list=token_list)

    data_dir = Path(args.data_dir)
    wav_scp = data_dir / "wav.scp"
    text_path = data_dir / "text"
    if not wav_scp.is_file() or not text_path.is_file():
        raise FileNotFoundError("wav.scp/text not found in data_dir")

    wav_reader = SoundScp(wav_scp)
    text_map = read_2columns_text(text_path)

    chunk_frames = _get_chunk_frames(model, args.chunk_length)
    logging.info(f"Chunk frames: {chunk_frames}")

    utts = list(text_map.keys())
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f_out, torch.no_grad():
        for utt in utts:
            if utt not in wav_reader:
                continue
            speech, rate = wav_reader[utt]
            if isinstance(speech, np.ndarray):
                speech = torch.from_numpy(speech)
            speech = speech.to(device=device, dtype=torch.float32)
            speech = speech.unsqueeze(0)
            speech_lengths = torch.tensor([speech.shape[1]], device=device)

            text = text_map[utt]
            tokens = tokenizer.text2tokens(text)
            token_ids = converter.tokens2ids(tokens)
            if len(token_ids) == 0:
                f_out.write(f"{utt}\n")
                continue
            text_tensor = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0)
            text_lengths = torch.tensor([len(token_ids)], device=device)

            # Encoder
            encoder_out, encoder_out_lens = model.encode(speech, speech_lengths)

            # Prepare transducer IO
            decoder_in, target, t_len, u_len = get_transducer_task_io(
                text_tensor, encoder_out_lens, ignore_id=model.ignore_id
            )
            model.decoder.set_device(encoder_out.device)
            decoder_out = model.decoder(decoder_in)

            # k2 forward-backward (px_grad)
            loss_out = model._calc_k2_transducer_pruned_loss(
                encoder_out,
                decoder_out,
                text_tensor,
                t_len,
                u_len,
                reduction="none",
                return_px_grad=True,
            )
            _, _, _, px_grad = loss_out

            # px_grad: (B, T, V), use -px_grad as posterior proxy
            posterior = -px_grad
            top1 = posterior.argmax(dim=-1)  # (B, T)
            top1 = top1[0][: int(t_len[0].item())].tolist()

            # Chunk-level token: if all blank => blank_id, else majority non-blank
            num_frames = len(top1)
            num_chunks = int(np.ceil(num_frames / float(chunk_frames)))
            chunk_tokens = []
            for c in range(num_chunks):
                s = c * chunk_frames
                e = min((c + 1) * chunk_frames, num_frames)
                chunk = top1[s:e]
                non_blank = [t for t in chunk if t != args.blank_id]
                if len(non_blank) == 0:
                    chunk_tokens.append(args.blank_id)
                else:
                    # majority non-blank
                    values, counts = np.unique(non_blank, return_counts=True)
                    chunk_tokens.append(int(values[counts.argmax()]))

            f_out.write(f"{utt} {' '.join(map(str, chunk_tokens))}\n")


if __name__ == "__main__":
    main()
