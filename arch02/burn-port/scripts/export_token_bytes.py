#!/usr/bin/env python3
"""Export SentencePiece byte-count LUTs for the Rust BPB evaluator.

Replicates the LUT logic in train_gpt_sprint_native.py (lines ~294-318):
  base_bytes[id]        = utf-8 byte length of the piece with leading '▁'
                          (U+2581) stripped; byte tokens (<0xNN>) count as 1
  has_leading_space[id] = piece starts with '▁'
  is_boundary_token[id] = control / unknown / unused pieces
                          (True for ids >= sp vocab size)

Usage:
  python export_token_bytes.py fineweb_1024_bpe.model token_bytes.json [vocab_size]
"""
import json
import sys

import sentencepiece as spm


def main() -> None:
    model_path = sys.argv[1]
    out_path = sys.argv[2]
    vocab_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

    sp = spm.SentencePieceProcessor(model_file=model_path)
    n = sp.get_piece_size()
    assert n <= vocab_size, f"sp vocab {n} > requested {vocab_size}"

    base_bytes, has_space, is_boundary = [], [], []
    for i in range(vocab_size):
        if i >= n:
            base_bytes.append(0)
            has_space.append(False)
            is_boundary.append(True)  # python default for out-of-range ids
            continue
        piece = sp.id_to_piece(i)
        if sp.is_byte(i):
            base_bytes.append(1)
            has_space.append(False)
            is_boundary.append(False)
            continue
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i):
            base_bytes.append(0)
            has_space.append(False)
            is_boundary.append(True)
            continue
        leading = piece.startswith("▁")
        stripped = piece.removeprefix("▁")  # strip ONE leading space marker
        base_bytes.append(len(stripped.encode("utf-8")))
        has_space.append(leading)
        is_boundary.append(False)

    with open(out_path, "w") as f:
        json.dump(
            {
                "base_bytes": base_bytes,
                "has_leading_space": has_space,
                "is_boundary_token": is_boundary,
            },
            f,
        )
    print(f"wrote {out_path}: {vocab_size} entries ({n} real pieces)")


if __name__ == "__main__":
    main()
