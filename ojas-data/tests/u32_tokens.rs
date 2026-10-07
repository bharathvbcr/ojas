use ojas_data::{
    BatchSampler, CounterRng, SamplerConfig, TokenBin, TokenWidth, FINEWEB_HEADER_BYTES,
    FINEWEB_U32_MAGIC, FINEWEB_U32_VERSION,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

struct Tmp(PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn tmp(tag: &str) -> Tmp {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    Tmp(std::env::temp_dir().join(format!("ojas-u32-{tag}-{}-{n}.bin", std::process::id())))
}

#[test]
fn headerless_u32_qwen_and_llama3_values() {
    let file = tmp("qwen-llama3");
    // Tokens exceeding u16::MAX: Llama 3 (128,256) and Qwen (248,320)
    let tokens: Vec<u32> = vec![
        0,
        1,
        65_535,
        65_536,
        128_256,
        200_000,
        248_320,
        u32::MAX - 5,
        u32::MAX,
    ];
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for &t in &tokens {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    std::fs::write(&file.0, &bytes).unwrap();

    let bin = TokenBin::open_headerless_u32(&file.0).unwrap();
    assert_eq!(bin.len(), tokens.len() as u64);
    assert_eq!(bin.token_width(), TokenWidth::U32);
    assert_eq!(bin.bytes_per_token(), 4);
    assert!(!bin.is_empty());

    let mut got = vec![0u32; tokens.len()];
    bin.read_into_u32(0, &mut got).unwrap();
    assert_eq!(got, tokens);

    // Sub-window reads
    let mut sub = [0u32; 3];
    bin.read_into_u32(4, &mut sub).unwrap();
    assert_eq!(sub, [128_256, 200_000, 248_320]);
}

#[test]
fn headerless_u32_alignment_rejections() {
    let file = tmp("bad-lengths");
    for bad_len in [1, 2, 3, 5, 6, 7, 9, 13, 101] {
        let bytes = vec![0u8; bad_len];
        std::fs::write(&file.0, &bytes).unwrap();
        let err = TokenBin::open_headerless_u32(&file.0).unwrap_err();
        assert!(
            err.detail().contains("multiple of 4") || err.detail().contains("truncated"),
            "bad_len {bad_len}: {err}"
        );
    }
}

#[test]
fn fineweb_u32_roundtrip_and_version_support() {
    let file = tmp("fw-u32-roundtrip");
    let tokens: Vec<u32> = (0..500)
        .map(|i| 100_000 + i * 297 % 148_320) // all > 65535, up to 248,320
        .collect();

    for version in [FINEWEB_U32_VERSION, 1] {
        let mut bytes = vec![0u8; FINEWEB_HEADER_BYTES as usize];
        bytes[0..4].copy_from_slice(&FINEWEB_U32_MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        bytes[8..12].copy_from_slice(&(tokens.len() as i32).to_le_bytes());
        for &t in &tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        std::fs::write(&file.0, &bytes).unwrap();

        // open_fineweb auto-detects u32
        let bin = TokenBin::open_fineweb(&file.0).unwrap();
        assert_eq!(bin.token_width(), TokenWidth::U32);
        assert_eq!(bin.len(), tokens.len() as u64);

        let mut readback = vec![0u32; tokens.len()];
        bin.read_into_u32(0, &mut readback).unwrap();
        assert_eq!(readback, tokens);

        // explicit constructor
        let bin_explicit = TokenBin::open_fineweb_u32(&file.0).unwrap();
        assert_eq!(bin_explicit.len(), tokens.len() as u64);
    }
}

#[test]
fn read_into_u16_on_u32_bin_guards_against_silent_truncation() {
    let file = tmp("no-silent-trunc");
    // Tokens: [100, 65535, 65536, 128256]
    let tokens: [u32; 4] = [100, 65_535, 65_536, 128_256];
    let mut bytes = Vec::new();
    for &t in &tokens {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    std::fs::write(&file.0, &bytes).unwrap();

    let bin = TokenBin::open_headerless_u32(&file.0).unwrap();

    // Reading the first 2 tokens (which fit in u16) succeeds:
    let mut fits = [0u16; 2];
    bin.read_into(0, &mut fits).unwrap();
    assert_eq!(fits, [100, 65_535]);

    // Reading across token 65536 (at index 2) fails loud:
    let mut overflows = [0u16; 2];
    let err = bin.read_into(1, &mut overflows).unwrap_err();
    assert!(err.detail().contains("65536"), "{err}");
    assert!(err.detail().contains("exceeds u16::MAX"), "{err}");

    // Reading Qwen/Llama3 token (at index 3) fails loud:
    let mut one = [0u16; 1];
    let err_qwen = bin.read_into(3, &mut one).unwrap_err();
    assert!(err_qwen.detail().contains("128256"), "{err_qwen}");
}

#[test]
fn batch_sampler_end_to_end_with_qwen_u32_tokens() {
    let file = tmp("sampler-qwen");
    let seq_len = 16;
    let batch_size = 4;
    let total_tokens = 1000;
    // Generate distinct tokens with values in Qwen vocabulary range [100_000..248_320]
    let tokens: Vec<u32> = (0..total_tokens).map(|i| 100_000 + i as u32).collect();

    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for &t in &tokens {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    std::fs::write(&file.0, &bytes).unwrap();

    let bin = TokenBin::open_headerless_u32(&file.0).unwrap();
    let cfg = SamplerConfig {
        seq_len,
        batch: batch_size,
        seed: 42,
    };
    let mut sampler = BatchSampler::new(&bin, cfg.clone()).unwrap();
    let windows_per_epoch = sampler.windows_per_epoch();
    assert_eq!(
        windows_per_epoch,
        (total_tokens as u64 - 1) / seq_len as u64
    );

    // Row start tokens in the order the sampler produced them.
    let mut visited_starts = Vec::new();
    let batches_per_epoch = windows_per_epoch.div_ceil(batch_size as u64);

    for _ in 0..batches_per_epoch {
        let batch = sampler.next_batch().unwrap();
        assert_eq!(batch.batch, batch_size);
        assert_eq!(batch.seq_len, seq_len);
        assert_eq!(batch.x.len(), batch_size * seq_len);
        assert_eq!(batch.y.len(), batch_size * seq_len);

        for row in 0..batch_size {
            let row_x = &batch.x[row * seq_len..(row + 1) * seq_len];
            let row_y = &batch.y[row * seq_len..(row + 1) * seq_len];
            let start_tok = row_x[0];
            visited_starts.push(start_tok);

            // Verify contiguous sequence and y is x shifted by 1
            for i in 0..seq_len {
                assert_eq!(row_x[i], start_tok + i as u32, "x row sequence intact");
                assert_eq!(row_y[i], start_tok + i as u32 + 1, "y is x shifted by 1");
                // Verify all token values stayed in u32 Qwen range:
                assert!(row_x[i] >= 100_000, "token {} preserved", row_x[i]);
            }
        }
    }

    // Window k starts at token k * seq_len, so its first token is
    // 100_000 + k * seq_len. Epoch 0 is the first W rows: each window once.
    let w = usize::try_from(windows_per_epoch).unwrap();
    let every_window: BTreeSet<u32> = (0..w).map(|k| 100_000 + (k * seq_len) as u32).collect();
    let epoch0: BTreeSet<u32> = visited_starts[..w].iter().copied().collect();
    assert_eq!(epoch0.len(), w, "a window repeated within epoch 0");
    assert_eq!(epoch0, every_window, "epoch 0 visits every window");
    // The last batch runs past the epoch end into epoch 1; those rows are
    // windows too, and the cursor says where epoch 1 stands.
    let spill = visited_starts.len() - w;
    assert_eq!(spill, batches_per_epoch as usize * batch_size - w);
    let all: BTreeSet<u32> = visited_starts.iter().copied().collect();
    assert_eq!(all, every_window, "no start outside the window grid");
    assert_eq!(sampler.cursor().shard, 1);
    assert_eq!(sampler.cursor().token_index, spill as u64);
}

#[test]
fn batch_sampler_resumes_cleanly_on_u32_tokens() {
    let file = tmp("sampler-resume-u32");
    let seq_len = 8;
    let batch_size = 2;
    let tokens: Vec<u32> = (0..200).map(|i| 200_000 + i as u32).collect();
    let mut bytes = Vec::new();
    for &t in &tokens {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    std::fs::write(&file.0, &bytes).unwrap();

    let bin = TokenBin::open_headerless_u32(&file.0).unwrap();
    let cfg = SamplerConfig {
        seq_len,
        batch: batch_size,
        seed: 999,
    };

    let mut s1 = BatchSampler::new(&bin, cfg.clone()).unwrap();
    let b1 = s1.next_batch().unwrap();
    let cursor = s1.cursor();

    let mut s2 = BatchSampler::resume(&bin, cfg, cursor).unwrap();
    let b2 = s2.next_batch().unwrap();
    let b1_second = s1.next_batch().unwrap();

    assert_eq!(b2.x, b1_second.x);
    assert_eq!(b2.y, b1_second.y);
    assert_ne!(b1.x, b2.x);
}

#[test]
fn boundary_and_adversarial_stress_testing() {
    let file = tmp("adversarial");
    std::fs::write(&file.0, []).unwrap();
    let empty_bin = TokenBin::open_headerless_u32(&file.0).unwrap();
    assert!(empty_bin.is_empty());
    assert_eq!(empty_bin.len(), 0);

    // Reading 0 tokens into empty slice is Ok(())
    assert!(empty_bin.read_into_u32(0, &mut []).is_ok());
    assert!(empty_bin.read_into(0, &mut []).is_ok());

    // Reading 1 token from empty bin errors
    assert!(empty_bin.read_into_u32(0, &mut [0]).is_err());
    assert!(empty_bin.read_into(0, &mut [0]).is_err());

    // Out-of-bounds start
    assert!(empty_bin.read_into_u32(100, &mut [0]).is_err());

    // Overflow index
    assert!(empty_bin.read_into_u32(u64::MAX, &mut [0; 2]).is_err());
    assert!(empty_bin.read_into_u32(u64::MAX / 2, &mut [0; 10]).is_err());

    // Random walk stress test with CounterRng
    let mut rng = CounterRng::new(0xABCD_EF01);
    let stress_file = tmp("random-walk");
    let count = 2500;
    let stress_tokens: Vec<u32> = (0..count).map(|_| rng.next_u64() as u32).collect();
    let mut stress_bytes = Vec::with_capacity(count * 4);
    for &t in &stress_tokens {
        stress_bytes.extend_from_slice(&t.to_le_bytes());
    }
    std::fs::write(&stress_file.0, &stress_bytes).unwrap();
    let stress_bin = TokenBin::open_headerless_u32(&stress_file.0).unwrap();

    for _ in 0..1000 {
        let max_len = (rng.next_u64() % 32) as usize;
        let start = rng.next_u64() % (count as u64 + 10);
        let mut buf = vec![0u32; max_len];
        let fits = start + max_len as u64 <= count as u64;
        let res = stress_bin.read_into_u32(start, &mut buf);
        assert_eq!(res.is_ok(), fits);
        if fits {
            let s = start as usize;
            assert_eq!(buf, stress_tokens[s..s + max_len]);
        }
    }
}
