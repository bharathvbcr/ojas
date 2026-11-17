//! `BatchSampler::next_batch` reuses its scratch buffers: after a warm-up call, the number of
//! allocations one call makes does not grow with the batch size `B`. The returned `x` and `y`
//! (and the window-start list) are the only per-call allocations, so the count is the same
//! at `B = 1` and `B = 64`. Before the scratch was reused, every row allocated a raw byte
//! buffer inside the read, so the count grew with `B`.

use ojas_data::{BatchSampler, SamplerConfig, TokenBin};
use ojas_test_support::{thread_allocs as allocs, ThreadAllocCount};
use std::path::PathBuf;

#[global_allocator]
static ALLOC: ThreadAllocCount = ThreadAllocCount;

struct Tmp(PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

const SEQ_LEN: usize = 16;

/// Allocations made by one `next_batch` call at batch size `batch`, after one warm-up call.
fn allocs_per_batch(bin: &TokenBin, batch: usize) -> usize {
    let cfg = SamplerConfig {
        seq_len: SEQ_LEN,
        batch,
        seed: 7,
    };
    let mut sampler = BatchSampler::new(bin, cfg).expect("sampler");
    sampler.next_batch().expect("warm-up batch");
    let before = allocs();
    let batch_out = sampler.next_batch().expect("measured batch");
    let used = allocs() - before;
    assert_eq!(batch_out.x.len(), batch * SEQ_LEN, "batch shape");
    used
}

#[test]
fn next_batch_allocation_count_does_not_grow_with_batch_size() {
    let file =
        Tmp(std::env::temp_dir().join(format!("ojas-sampler-alloc-{}.bin", std::process::id())));
    // 8192 u32 tokens = 512 windows of 16, so B = 64 fits in one epoch without wrapping.
    let mut bytes = Vec::with_capacity(8192 * 4);
    for t in 0..8192u32 {
        bytes.extend_from_slice(&(t.wrapping_mul(2_654_435_761) | 0x1_0000).to_le_bytes());
    }
    std::fs::write(&file.0, &bytes).expect("write token bin");
    let bin = TokenBin::open_headerless_u32(&file.0).expect("open token bin");

    let one = allocs_per_batch(&bin, 1);
    let many = allocs_per_batch(&bin, 64);
    assert_eq!(
        one, many,
        "next_batch allocated {one} times at B=1 but {many} times at B=64: \
         a per-row buffer is allocated again"
    );
}
