#!/usr/bin/env python3
import argparse
from pathlib import Path

from espnet2.fileio.read_text import read_2columns_text
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter


def main():
    parser = argparse.ArgumentParser(
        description="Prepare token_int from text using BPE model"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bpemodel", type=str, required=True)
    parser.add_argument("--token_list", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    text_path = data_dir / "text"
    if not text_path.is_file():
        raise FileNotFoundError(f"{text_path} not found")

    token_list = Path(args.token_list).read_text(encoding="utf-8").splitlines()
    tokenizer = build_tokenizer(token_type="bpe", bpemodel=args.bpemodel)
    converter = TokenIDConverter(token_list=token_list)

    text_map = read_2columns_text(text_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for utt, text in text_map.items():
            tokens = tokenizer.text2tokens(text)
            ids = converter.tokens2ids(tokens)
            f.write(f"{utt} {' '.join(map(str, ids))}\n")


if __name__ == "__main__":
    main()
