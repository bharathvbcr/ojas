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

**The tree did not get smaller, and that is the honest number.** tessl lost 368
lines of source net (295 insertions against 663 deletions over three files),
and its reader went from 950 lines to a 517-line adapter. ojas-io gained 1,396
insertions against 82 deletions over eight files — `src/safetensors.rs` is
3,070 lines now and `src/open.rs` 300 — of which about 385 lines are new tests:
`tests/tessl_rejection_corpus.rs` (246) holds tessl's whole rejection corpus
against the reader underneath, `tests/header_amplification.rs` (101) measures
where the budget refuses, and the `widen_in_place` test (38) moved here with
the code it covers. The rest is the eight bounds in the next section. One
duplicate parser is gone; what replaced it is the discipline the surviving
reader did not have, with tests that fail without it. Trading 950 duplicated
lines for a bound reader is the point, not the line count.

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

## What this leaves for ft-b5fb19e6

That card wants more safetensors dtypes and a sharded index in ojas-io, and the
decision above puts it downstream of this change rather than beside it. Of its
two items:

- **The sharded index is there and is now bounded.** `CheckpointView::open_index`
  already read `model.safetensors.index.json`; what this change added is the
  per-shard budget beside the cumulative one, the 16 MiB cap on the index text
  before it is read, the regular-file check on the index itself, and the index
  file named in its own refusals. A sharded checkpoint is read through ojas-io
  by ojas-qwen35 today, over the 26 adversarial cases.
- **The dtype list is untouched.** `StDtype` still has the same 22 variants, so
  whatever ft-b5fb19e6 wants beyond them — and the question of whether tessl's
  `Dtype` should follow — is still its own work. This change deliberately did
  not widen either enum: it moved who owns the parser, not what the format
  admits.

## ojas-oracle keeps its wrapper, and is not a third parser

The card asked whether `ojas-oracle/src/safetensors.rs` should re-export this
reader instead. It should not, and it was never the third parser the question
assumed: at 115 lines it already reads the format through
`ojas_io::SafeTensors` — the header grammar, schema, tiling, dtype widths and
positioned reads are all this crate's — and adds only the oracle's fixture
policy on top.

That policy is deliberately narrower than the reader's, which is why it is not
a re-export:

- a header cap of 1 MiB, where the reader's is 100 MB, because a fixture's
  header is a few KiB and the 124M export's is about 25 KiB;
- its own two-variant `StDtype` of `F32` and `I64` only, because that is what
  the generators write, so a fixture naming any other dtype is a generator bug
  and is refused by the type rather than read;
- `OjasError::OutOfRange { op: "oracle_fixture", .. }` rather than `IoError`,
  because the oracle's callers are ojas tests.

A re-export would widen all three, and the fixture generators would lose the
check that catches them writing something they did not mean to. So the
disposition is **keep**, for the same reason tessl's adapter is kept: the
policy on top is the crate's, the format underneath is not.

## The ojas/tessl path cycle survives this

Removing the safetensors strand does **not** break the cycle.
`ojas-qwen35/src/config.rs` still uses `tessl::qwen35_model::Qwen35Config`,
`tessl::gdn_train::{GDN_TRAIN_DK, GDN_TRAIN_BV}`, `tessl::qwen35::GDN_KEY_DIM`
and `tessl::attn_train::ATTN_TRAIN_HEAD_DIM` to validate a config against the
kernels that will run it, and `ojas-metal` depends on tessl by design. So
`ojas-qwen35` still path-depends on tessl. Those eleven call sites in
`config.rs` and `links = "tessl"` in `tessl/Cargo.toml` were read this pass;
the consequence — that an ojas worktree at a second path cannot build, because
every crate in one build must point at one tessl path and a worktree gets a
lockfile collision and exits 101 — is carried over from the card and from an
earlier session's finding, and was **not** re-run here. What this change
removes is one strand of the cycle, not the cycle.

A note for whoever takes the MSRV decision (D2 in the plan): tessl already
declared `ojas-io` as a path dependency before this change and used it nowhere,
so no Cargo change and no MSRV ask was needed here. tessl declares
`rust-version = "1.82"` while ojas's workspace floor is `1.97`, which is a
standing inconsistency this change neither created nor fixed.

## Verification

Every row below is a run with its real exit code captured, not a count read out
of a pipe. The ojas-io baseline at `9c5e784` was 161 passed over 12 binaries.

| Check | Result |
| :--- | :--- |
| `cargo test -p ojas-io --release` | **181 passed, 0 failed**, 0 ignored, 14 binaries |
| `cargo test -p ojas-oracle --release` | **255 passed, 0 failed**, 14 ignored, 21 binaries |
| `cargo test -p ojas-qwen35 --release` | **54 passed, 0 failed**, 9 ignored, 13 binaries |
| of which `sharded_checkpoint_adversarial.rs` | **26 passed, 0 failed**, no assertion edited |
| `cargo check --workspace --all-targets` | clean |
| `cargo clippy -p ojas-io -p ojas-qwen35 -p ojas-oracle --all-targets -- -D warnings` | clean, no error or warning line |
| `./scripts/fmt_check.sh` | clean |
| `go run scripts/sitegen/main.go -check` | generated files up to date, 57 files |
| tessl, `cargo test --release` (full suite, on the GPU) | **not re-run — see below**; the last complete run was 1,435 passed, 1 failed, 36 ignored, 128 binaries, and that one failure is fixed and green |
| tessl, the three binaries that failure touched | **30 passed, 0 failed**, 1 ignored (`host_access_ban` 6, `safetensors` 12, `reader_bounds` 12) |
| tessl, `cargo test --release --lib parser_mutation::` | **16 passed, 0 failed** |
| tessl, `cargo clippy --all-targets -- -D warnings` | clean |
| tessl, `rustfmt --check` on the three files this change touched | clean |

Each of the eight absorbed bounds has a test that fails against the pre-change
code. Three of them were found by a run rather than by reading: the FIFO hang,
and the two adversarial cases that assert a refused index names the index file,
which this reader's `open_index` had stopped doing.

`tessl/tests/safetensors.rs` kept its rejection semantics: of the 52 malformed
inputs it covers, 42 keep their exact needle and 10 are the same refusal in
different words, each annotated in place with the needle it had, and the
sub-byte block gained two inputs rather than losing any. No test was deleted or
weakened. One contract was changed deliberately — the header amplification
bound — and `ojas-io/tests/tessl_rejection_corpus.rs` now holds tessl's whole
rejection corpus against the reader underneath, so a future ojas-io change
cannot quietly widen what a checkpoint may be.

The ignored tessl test worth naming is `opens_the_qwen35_2b_checkpoint`, which
needs the 4.5 GB file from the Hugging Face cache; it was ignored before this
change too.

**tessl's full suite could not be re-run, and is not reported as passing.** It
ran complete once on this work at 1,435 passed / 1 failed / 36 ignored over 128
binaries. The one failure was
`host_access_ban::library_code_never_calls_a_panicking_host_accessor`, which
caught the adapter calling `.read_u32(` in method form — a real defect the guard
exists to catch, fixed and now green on its own (6 passed, 0 failed), along with
`safetensors` (12) and `reader_bounds` (12). Two attempts to re-run the whole
suite after that fix did not produce a result: the first was `SIGKILL`ed
mid-compile with the machine at load 38, and the second failed to build in six
seconds with no test run, because another session holds this checkout and has
`src/config_json.rs` and `src/qwen35_adamw.rs` part-way through a
`Json` → `JsonValue` rename that does not compile yet. Neither file belongs to
this work and neither was touched. So the outstanding check is one clean
end-to-end run of tessl's suite once that refactor lands.

**What is not verified here, and is not claimed to be.** `cargo fmt --all
--check` in tessl fails on ten files this change never touched — all wrapped at
100 columns where that repo's `rustfmt.toml` says 120 — so tessl's `fmt` CI job
is red on `main` independently of this work. `README.md`'s per-crate test
counts come from `docs/data/test-runs.json`, a single recorded run at
`51825fd`, so `ojas-io` still reads 76 passed there; refreshing that means
re-running the whole workspace with the command that file records, which this
change did not do. An ojas worktree at a second path still cannot build, for
the reason in the section above.
