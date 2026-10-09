//! GPT-2 tokenizer throughput (`bench/bpe_throughput.rs`).
//!
//! `cargo run -p ojas-data --release --example bpe_throughput -- VOCAB_DIR
//! CORPUS_BIN [ROUNDS]`

#[path = "../../bench/bpe_throughput.rs"]
mod bpe_throughput;

fn main() {
    bpe_throughput::main();
}
