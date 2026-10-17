//! SHA-256 throughput (`bench/sha256_throughput.rs`).
//!
//! `cargo run -p ojas-train --release --example sha256_throughput -- [ROUNDS]`

#[path = "../../bench/sha256_throughput.rs"]
mod sha256_throughput;

fn main() {
    sha256_throughput::main();
}
