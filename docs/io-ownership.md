# Decision: ojas owns safetensors IO and checkpoints

Owner decision, 2026-11-01. The tessl side of this record is
[`tessl/docs/io-ownership.md`](https://github.com/bharathvbcr/tessl/blob/main/docs/io-ownership.md);
the two say the same thing from each repository's side. This is the IO slice of
the ownership split in [`unification-plan.md`](unification-plan.md) (phase P8),
and it is the first phase of that plan to land.

## The decision

**ojas owns safetensors IO and checkpoints.** `ojas_io::safetensors` is the
canonical reader. tessl's `safetensors` module is a thin adapter over it and
holds no parser. This follows the binding direction of 2026-10-31 — ojas is the
training framework and owns "checkpoints, IO" — and it reverses, in code, the
direction an earlier card had taken.

Before: two parsers for one format, 2,200 lines in ojas-io and 950 in tessl,
kept in step by hand. After: one parser, and an adapter that states what is
tessl's.

## What lives where

| Concern | Owner |
| :--- | :--- |
| Header grammar, schema walk, tiling of the data buffer, dtype widths | `ojas-io` |
| Positioned reads, file-identity re-check, index parsing and its byte cap | `ojas-io` |
| Shard-name validation, containment, `weight_map` agreement | `ojas-io` |
| The payload budget a caller passes, and the per-read allocation check | `ojas-io` |
| tessl's `Dtype` (19 variants) and its refusal of the sub-byte floats | `tessl` |
| tessl's `String` errors and `Vec<usize>` shapes | `tessl` |
| tessl's physical-memory default when no limit is given | `tessl` |
| Mac inference loaders built on the reader (Qwen3.5, MLX Q4, BERT, EmbeddingGemma 2) | `tessl` |

Three things are deliberately not shared, and each is a decision rather than an
oversight:

- **`None` means different things.** In ojas-io a `None` payload limit is
  unbounded: it validates a header without loading the file and reads tensor
  bytes positioned from disk, so opening a checkpoint larger than memory is a
  legitimate thing to do, and the read path bounds its allocations fallibly.
  In tessl `None` means this machine's physical memory, because its loaders
  hand whole tensors to host code. That default is tessl policy and lives in
  tessl's adapter.
- **Sub-byte floats.** ojas-io supports `F4`, `F6_E2M3` and `F6_E3M2` on a
  checked path: it sizes them from their bit width and refuses any span that is
  not bit-exact. tessl declines them, because every path below its reader is
  written in whole bytes. The refusal is `TryFrom<StDtype> for Dtype`, so it is
  enforced by the type at open rather than documented. `pub type Dtype =
  StDtype` was rejected: it would put three variants in tessl's namespace and
  break four exhaustive matches in `tessl/src/qwen35.rs`.
- **Header amplification.** ojas-io budgets the JSON tree as it builds it
  (`8 * input + 64 KiB`) and refuses a header that would exceed it; tessl's
  reader had no such bound. ojas-io is the stricter of the two and stays that
  way, so tessl can no longer open a header made almost entirely of shape
  dimensions — above about 8 KiB of dimension text, measured at 4,099
  dimensions in `ojas-io/tests/header_amplification.rs`. No model writes such a
  header; real tensors have rank 5 at most and the 2B's whole header is about
  90 KB.

## What ojas-io absorbed first, and why the order mattered

tessl's reader was **stricter** than ojas-io's in several places. Deduplicating
in the other order would have dropped each of those bounds silently, so
ojas-io absorbed them first, each with a test that fails against the
pre-change code. Four of the eight were anticipated; four were found by
reading tessl's suite and by running ojas-qwen35's adversarial cases against
the migrated path.

| # | What was missing in ojas-io | Found by |
| :--- | :--- | :--- |
| 1 | `SafeTensors::open_with_limit`: the only bounded way in was `CheckpointView` | card |
| 2 | A per-read budget check — widening BF16 to f32 asks for twice the declared bytes | card |
| 3 | The limit reaching each shard's reader, so reads through a view are bounded | card |
| 4 | The limit honoured on a single-file checkpoint (fixed earlier, in 9c5e784) | card |
| 5 | File-identity re-check on every read, so a file rewritten in place is refused instead of read through stale offsets | reading tessl's suite |
| 6 | A byte cap and a regular-file check on the checkpoint index | diffing the two `open_index` bodies |
| 7 | Order-independent shape overflow: `[2^32, 2^32, 0]` was refused while `[0, 2^32, 2^32]` — the same empty tensor — opened | reading tessl's `checked_numel` |
| 8 | A FIFO named as a shard **hung** the reader instead of being refused | running the 26 adversarial cases |

Item 8 is the one that matters most: it is a hang, not a missing bound.
`SafeTensors::open` called `File::open` and left the regular-file check until
after it returned, and opening a FIFO for reading blocks until a writer
appears. Migrating the loader onto this reader without fixing it would have
turned a refusal into a denial of service on a hostile checkpoint.
`open::open_regular_file` is the one owner of that refusal now.

## What became of ft-02256073

**Re-pointed and kept, not superseded.** That card routed ojas-qwen35's sharded
loading through tessl's `CheckpointView` and verified it properly. Its work is
merged (`4affd09` is an ancestor of `main`), and its 26-case
`ojas-qwen35/tests/sharded_checkpoint_adversarial.rs` is the acceptance bar
this change inherited and passes **unchanged, 26/26** — symlink-followed,
escaping-shard-name, symlink-loop, dangling-link, oversize-header and FIFO
cases included. Not one assertion was edited. What changed is only which reader
sits underneath.

Two runs before that did not pass, and both found real defects rather than test
drift: the first hung on the FIFO case (item 8), the second failed two cases
that assert a refused index names the index file, which ojas-io's `open_index`
had stopped doing.

## The ojas/tessl path cycle survives this

Removing the safetensors strand does **not** break the cycle.
`ojas-qwen35/src/config.rs` still uses `tessl::qwen35_model::Qwen35Config`,
`tessl::gdn_train::{GDN_TRAIN_DK, GDN_TRAIN_BV}`, `tessl::qwen35::GDN_KEY_DIM`
and `tessl::attn_train::ATTN_TRAIN_HEAD_DIM` to validate a config against the
kernels that will run it, and `ojas-metal` depends on tessl by design. So
`ojas-qwen35` still path-depends on tessl, and an ojas worktree at a second
path still cannot build (tessl declares `links = "tessl"`, so every crate in
one build must point at one tessl path; a worktree gets a lockfile collision
and exits 101). What this change removes is one strand of the cycle, not the
cycle.

A note for whoever takes the MSRV decision (D2 in the plan): tessl already
declared `ojas-io` as a path dependency before this change and used it nowhere,
so no Cargo change and no MSRV ask was needed here. tessl declares
`rust-version = "1.82"` while ojas's workspace floor is `1.97`, which is a
standing inconsistency this change neither created nor fixed.

## Verification

Counts from running, not inferred. The ojas-io baseline at `9c5e784` was 161
passed over 12 binaries.

| Suite | Result |
| :--- | :--- |
| `ojas-io` | 181 passed, 0 failed, 14 binaries |
| `ojas-qwen35` `sharded_checkpoint_adversarial` | 26 passed, 0 failed |
| tessl `--test safetensors` | 12 passed, 0 failed, 1 ignored |
| tessl `--test reader_bounds` | 12 passed, 0 failed, no edit to the file |
| tessl `--lib parser_mutation::` | 16 passed, 0 failed |

`tessl/tests/safetensors.rs` kept its rejection semantics: of the 52 malformed
inputs it covers, 42 keep their exact needle and 10 are the same refusal in
different words, each annotated in place with the needle it had. No test was
deleted or weakened. One contract was changed deliberately — the header
amplification bound above — and `ojas-io/tests/tessl_rejection_corpus.rs` now
holds tessl's whole rejection corpus against the reader underneath, so a future
ojas-io change cannot quietly widen what a checkpoint may be.

The ignored tessl test is `opens_the_qwen35_2b_checkpoint`, which needs the
4.5 GB file from the Hugging Face cache; it was ignored before this change too.
