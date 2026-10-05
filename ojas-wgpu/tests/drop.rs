//! Dropping the last handle on a context must not wait without a bound for
//! the GPU (a session freed through the C ABI drops its backend on the
//! caller's thread). wgpu's own `Queue` drop waits for the queue to go
//! idle with no timeout (wgpu-core-30.0.1 `device/queue.rs:281`, Metal
//! `wgpu-hal-30.0.1/src/metal/mod.rs:837`), so with work still queued the
//! drop lasts as long as that work does.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use ojas_core::{Backend, Budget, Tensor};
use ojas_wgpu::WgpuBackend;

const N: usize = 4096;

fn square(seed: u32, b: &WgpuBackend) -> Tensor {
    let host: Vec<f32> = (0..N * N)
        .map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32 * 1e-3)
        .collect();
    b.upload(&Tensor::from_f32(&host, &[N, N], b.budget()).unwrap())
        .unwrap()
}

#[test]
fn dropping_a_busy_backend_returns_within_its_bound() {
    let g = WgpuBackend::open(Budget::new(4 << 30)).expect("wgpu adapter");
    let bound = Duration::from_millis(300);
    g.context().set_drop_wait(bound);
    let x = square(1, &g);
    let w = square(2, &g);
    // One GEMM's time, so the queued work is known to outlast the bound.
    let mut one = Duration::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        drop(g.linear_forward(&x, &w).unwrap());
        g.sync().unwrap();
        one = one.min(t.elapsed());
    }
    let queued = Duration::from_millis(1500);
    // The recorder submits every `FLUSH_AT` dispatches without waiting, so
    // a whole number of those leaves all of this work in flight.
    let flush = ojas_wgpu::FLUSH_AT;
    let n = (queued.as_secs_f64() / one.as_secs_f64())
        .ceil()
        .min(4096.0) as usize;
    let n = n.div_ceil(flush) * flush;
    for _ in 0..n {
        drop(g.linear_forward(&x, &w).unwrap());
    }
    // Each stage reports when it is done, so a failure names the drop that
    // blocked: an input tensor, or the backend and its context.
    let (tx, rx) = mpsc::channel::<(&str, Duration)>();
    let started = Instant::now();
    std::thread::spawn(move || {
        drop(x);
        let _ = tx.send(("x", started.elapsed()));
        drop(w);
        let _ = tx.send(("w", started.elapsed()));
        drop(g);
        let _ = tx.send(("g", started.elapsed()));
    });
    let limit = bound + Duration::from_millis(700);
    let mut stages = Vec::new();
    while let Some(left) = limit.checked_sub(started.elapsed()) {
        match rx.recv_timeout(left) {
            Ok(stage) => stages.push(stage),
            Err(_) => break,
        }
        if stages.len() == 3 {
            break;
        }
    }
    let returned = stages.len() == 3;
    let took = started.elapsed();
    let done: Vec<String> = stages
        .iter()
        .map(|(name, at)| format!("{name} at {:.3} s", at.as_secs_f64()))
        .collect();
    eprintln!(
        "one GEMM {:.1} ms, {n} queued (~{:.2} s), drop returned {returned} after {:.3} s; done: [{}]",
        one.as_secs_f64() * 1e3,
        one.as_secs_f64() * n as f64,
        took.as_secs_f64(),
        done.join(", ")
    );
    assert!(
        one.as_secs_f64() * n as f64 > limit.as_secs_f64(),
        "the queued work does not outlast the limit, so this proves nothing"
    );
    assert!(returned, "the drop was still waiting after {limit:?}");
}
