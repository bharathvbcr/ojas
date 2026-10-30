//! Host-only micro-bench of `CacheKey::new` on the published GDN module.
//!
//! No device, no NVRTC compile, no launch. Ignored so a normal `cargo test`
//! does not spend the timing loop. The key check still runs when this test
//! is selected.
//!
//! ```text
//! cargo test -p ojas-cuda --test nvrtc_cache_key_bench -- --ignored --nocapture --test-threads=1
//! ```

use std::hint::black_box;
use std::time::Instant;

use ojas_cuda::gdn_kernels::GDN_PUBLISHED;
use ojas_cuda::kernels::STRICT_SM90;
use ojas_cuda::nvrtc_cache::{fnv1a64, CacheKey};

const N: usize = 200;
const NVRTC: (i32, i32) = (12, 8);

#[test]
#[ignore = "host-only timing benchmark; run with --ignored --nocapture --test-threads=1"]
fn cache_key_new_hashes_gdn_published_per_call() {
    let once = CacheKey::new(&GDN_PUBLISHED, &STRICT_SM90, NVRTC);
    assert_eq!(once.module, GDN_PUBLISHED.name);
    assert_eq!(once.source_len, GDN_PUBLISHED.source.len());
    assert_eq!(
        once.source_hash,
        fnv1a64(GDN_PUBLISHED.source.as_bytes()),
        "CacheKey::new did not hash the published GDN source"
    );
    assert!(once.source_len > 0);

    for _ in 0..2 {
        black_box(CacheKey::new(
            black_box(&GDN_PUBLISHED),
            black_box(&STRICT_SM90),
            black_box(NVRTC),
        ));
    }
    let mut ns = Vec::with_capacity(N);
    let mut acc = 0u64;
    for _ in 0..N {
        let t0 = Instant::now();
        let key = CacheKey::new(
            black_box(&GDN_PUBLISHED),
            black_box(&STRICT_SM90),
            black_box(NVRTC),
        );
        let dt = t0.elapsed().as_nanos();
        acc ^= key.source_hash;
        acc = acc.wrapping_add(key.source_len as u64);
        ns.push(dt);
    }
    black_box(acc);
    ns.sort_unstable();
    let min_ns = ns[0];
    let max_ns = ns[ns.len() - 1];
    println!(
        "OJAS_META op=cache_key_new module={} source_len={} nvrtc={:?} host_only=1",
        GDN_PUBLISHED.name, once.source_len, NVRTC
    );
    println!(
        "OJAS_OP op=cache_key_new dir=new shape=GDN_PUBLISHED+STRICT_SM90 threads=1 n={N} min_ms={:.6} med_ms={:.6} max_ms={:.6} spread_ms={:.6} numerics=na source_len={}",
        min_ns as f64 / 1e6,
        ns[ns.len() / 2] as f64 / 1e6,
        max_ns as f64 / 1e6,
        (max_ns - min_ns) as f64 / 1e6,
        once.source_len
    );
}
