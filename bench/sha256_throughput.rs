//! `ojas_train::sha256` throughput on a 64 MiB buffer.
//!
//! Compiled by `#[path]` into `ojas-train/examples/sha256_throughput.rs`.
//!
//! Rows, each timed once per round, min and median over rounds:
//! - `one_shot`: `sha256` over the whole buffer.
//! - `streamed_64k`: `Sha256::update` in 64 KiB pieces, then `finalize`.
//!
//! The buffer is a fixed xorshift64 stream (seed `0x9e3779b97f4a7c15`).
//! Before timing, both rows must produce the same digest; the run refuses to
//! time otherwise. The digest is printed so two trees can be compared.
//!
//! Usage: `sha256_throughput [ROUNDS]` (default 7).

use ojas_train::sha256::{sha256, to_hex, Sha256};
use std::hint::black_box;
use std::time::{Duration, Instant};

const BYTES: usize = 64 << 20;
const PIECE: usize = 64 << 10;

fn buffer() -> Vec<u8> {
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut out = Vec::with_capacity(BYTES);
    while out.len() < BYTES {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.truncate(BYTES);
    out
}

fn streamed(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    for piece in data.chunks(PIECE) {
        h.update(piece);
    }
    h.finalize()
}

fn report(name: &str, mut times: Vec<Duration>) {
    times.sort();
    let min = times[0];
    let median = times[times.len() / 2];
    let mib = BYTES as f64 / (1 << 20) as f64;
    println!(
        "{name:<14} min {:>9.3} ms ({:>8.1} MiB/s)  median {:>9.3} ms ({:>8.1} MiB/s)",
        min.as_secs_f64() * 1e3,
        mib / min.as_secs_f64(),
        median.as_secs_f64() * 1e3,
        mib / median.as_secs_f64(),
    );
}

pub fn main() {
    let rounds: usize = match std::env::args().nth(1) {
        Some(s) => match s.parse() {
            Ok(n) if n >= 1 => n,
            _ => {
                eprintln!("ROUNDS must be an integer >= 1, got {s:?}");
                std::process::exit(2);
            }
        },
        None => 7,
    };
    let data = buffer();
    let want = sha256(&data);
    if streamed(&data) != want {
        eprintln!("one-shot and streamed digests differ; refusing to time");
        std::process::exit(1);
    }
    println!("bytes {BYTES} rounds {rounds} digest {}", to_hex(&want));

    let mut one_shot = Vec::with_capacity(rounds);
    let mut piecewise = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t = Instant::now();
        black_box(sha256(black_box(&data)));
        one_shot.push(t.elapsed());
        let t = Instant::now();
        black_box(streamed(black_box(&data)));
        piecewise.push(t.elapsed());
    }
    report("one_shot", one_shot);
    report("streamed_64k", piecewise);
}
