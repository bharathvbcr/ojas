# GPT-2 rank table

`vocab.json` and `merges.txt` are GPT-2's byte-level BPE vocabulary, the files
`ojas-data`'s real-vocabulary tests load (`bpe.rs`, `rank_table_dir`).

- Source: Hugging Face `openai-community/gpt2`, license MIT (its model card).
- Content matches that repository's `vocab.json` and `merges.txt` exactly
  (50,257 pieces, 50,000 merges, checked 2026-10-10). The bytes differ: this
  `vocab.json` is compact JSON, and the `merges.txt` header reads
  `#version: 0.2 - Trained by huggingface/tokenizers`.
- SHA-256:
  - `vocab.json` `3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7`
  - `merges.txt` `fe36cab26d4f4421ed725e10a2e9ddb7f799449c603a96e7f29b5a3c82a95862`
