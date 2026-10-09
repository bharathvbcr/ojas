# GPT-2 tokenizer throughput (2026-10-08)

Task `gp-tokenizer-throughput-and-decode`. The benchmark is
`bench/bpe_throughput.rs`, compiled into
`ojas-data/examples/bpe_throughput.rs`. It times `Bpe::encode_ordinary` and
`Bpe::decode_ordinary` with the GPT-2 rank table (Hugging Face `vocab.json`
and `merges.txt`, 50,257 pieces, 50,000 merges) over the complete documents
of nanolab's FineWeb-Edu `val.bin`: 2,434 documents, 11,412,017 bytes of
English text, 2,496,959 tokens. Before timing, every document's re-encode is
compared with the ids tiktoken 0.12 wrote into the bin: 2434 of 2434 match on
the base tree, so this is a byte-identity check over 11.4 MB, not twenty
strings. `digest` is FNV-1a over the ids of the one-call (`encode_joined`)
encode; lanes with the same digest produced the same 2,501,825 ids.

Apple M5 Pro, macOS 27.0.1, rustc 1.99.0, release, one thread. File hashes
and the binaries of each lane are in each run's `env.txt`; `uptime` before
and after every process is in `load.txt`. The machine was shared with other
sessions' builds; read every ratio against that, and ratios between 0.9 and
1.1 are not changes.

## Reproduce

```bash
cargo build -p ojas-data --release --example bpe_throughput
# one lane per binary; lanes alternate order every round
bash run.sh OUT_DIR 3 base=BASE_BIN after=AFTER_BIN
python3 summarize.py OUT_DIR base     # analysis only
```

`run.sh` defaults `VOCAB_DIR` and `CORPUS_BIN` to the two local paths in
`env.txt`. Each process runs one untimed warm-up and 5 timed rounds of every
row and prints min and median; `summarize.py` takes the minimum over
processes and the median of per-round ratios against the base lane.

## Baseline (`baseline/`, tree `c5aeee1`)

| row | min s | MB/s | Mtok/s |
| :-- | --: | --: | --: |
| encode_docs | 2.1230 | 5.38 | 1.176 |
| encode_joined | 2.1295 | 5.36 | 1.175 |
| decode_docs | 0.0423 | 270.05 | 59.086 |
| decode_joined | 0.0393 | 290.63 | 63.687 |

3 processes, 1-minute load 9.5 to 13.5. Before this benchmark the only
performance check was a 2 s wall-clock assertion on the 6-token fixture
(`long_input_encodes_in_near_linear_time`).
