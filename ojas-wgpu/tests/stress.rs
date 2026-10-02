//! Many OS threads on one `WgpuBackend`: mixed ops recorded concurrently
//! into the shared encoder must give exactly the bits a serial run gives; a
//! caller thread that panics must not disturb the others; a device lost in
//! the middle must fail every thread with an error, never a hang or a panic.
//! Every test runs under a watchdog, so a deadlock fails instead of hanging.

use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use ojas_core::{AdamWConfig, Backend, Budget, OjasError, Tensor, RMS_NORM_EPS};
use ojas_wgpu::WgpuBackend;

const WATCHDOG: Duration = Duration::from_secs(300);

/// Run `f` on its own thread; fail if it panics or outlives the watchdog.
fn watchdog<T: Send + 'static>(name: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(WATCHDOG) {
        Ok(v) => v,
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("{name}: hung past {WATCHDOG:?}"),
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{name}: the body panicked"),
    }
}

fn data(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn t(g: &WgpuBackend, seed: u64, shape: &[usize]) -> Result<Tensor, OjasError> {
    let n = shape.iter().product();
    g.upload(&Tensor::from_f32(&data(seed, n), shape, g.budget())?)
}

fn bits(g: &WgpuBackend, x: &Tensor) -> Result<Vec<u32>, OjasError> {
    Ok(g.download(x)?
        .to_f32_vec()?
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

/// One caller's mixed sequence. Every op is recorded before anything is
/// read back, so the recordings of different threads interleave in the
/// shared encoder.
fn workload(g: &WgpuBackend, seed: u64) -> Result<Vec<Vec<u32>>, OjasError> {
    let s = seed * 100;
    let x = t(g, s, &[37, 48])?;
    let w = t(g, s + 1, &[29, 48])?;
    let gy = t(g, s + 2, &[37, 29])?;
    let nw = t(g, s + 3, &[48])?;
    let shape = [1usize, 2, 33, 16];
    let (q, k, v) = (
        t(g, s + 4, &shape)?,
        t(g, s + 5, &shape)?,
        t(g, s + 6, &shape)?,
    );
    let ids: Vec<u32> = (0..37)
        .map(|i| ((i * 7 + seed as usize) % 29) as u32)
        .collect();
    let tgt = g.upload(&Tensor::from_u32(&ids, &[37], g.budget())?)?;
    let mut p = t(g, s + 7, &[300])?;
    let grad = t(g, s + 8, &[300])?;
    let zeros = Tensor::from_f32(&[0.0; 300], &[300], g.budget())?;
    let (mut m1, mut m2) = (g.upload(&zeros)?, g.upload(&zeros)?);

    let y = g.linear_forward(&x, &w)?;
    let (gx, gw) = g.linear_backward(&x, &w, &gy)?;
    let sy = g.silu_forward(&y)?;
    let r = g.rms_norm_forward(&x, &nw, RMS_NORM_EPS)?;
    let a = g.causal_sdpa_forward(&q, &k, &v)?;
    let (dq, dk, dv) = g.causal_sdpa_backward(&q, &k, &v, &q)?;
    let pt = g.permute(&x, &[1, 0])?;
    let ce = g.cross_entropy_mean_forward(&y, &tgt, None)?;
    g.adamw_step(
        &mut p,
        &grad,
        &mut m1,
        &mut m2,
        0,
        AdamWConfig::nanolab(1e-2, 0.1),
    )?;
    [
        &y, &gx, &gw, &sy, &r, &a, &dq, &dk, &dv, &pt, &ce, &p, &m1, &m2,
    ]
    .into_iter()
    .map(|x| bits(g, x))
    .collect()
}

fn open() -> WgpuBackend {
    WgpuBackend::open(Budget::new(4 << 30)).expect("wgpu adapter; a missing GPU fails the test")
}

#[test]
fn the_backend_is_send_and_sync() {
    fn shared<T: Send + Sync>() {}
    shared::<WgpuBackend>();
    shared::<ojas_wgpu::WgpuContext>();
}

#[test]
fn threads_sharing_one_backend_get_the_serial_bits() {
    watchdog("shared backend", || {
        const THREADS: usize = 8;
        const ROUNDS: usize = 6;
        let g = Arc::new(open());
        let want: Vec<Vec<Vec<u32>>> = (0..THREADS)
            .map(|i| workload(&g, i as u64).unwrap())
            .collect();
        let want = Arc::new(want);
        let start = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|i| {
                let (g, want, start) = (Arc::clone(&g), Arc::clone(&want), Arc::clone(&start));
                std::thread::spawn(move || {
                    start.wait();
                    for round in 0..ROUNDS {
                        let got = workload(&g, i as u64).unwrap();
                        assert!(
                            got == want[i],
                            "thread {i} round {round} differs from serial"
                        );
                    }
                })
            })
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            h.join().unwrap_or_else(|_| panic!("thread {i} panicked"));
        }
        g.sync().expect("no fault was raised");
    });
}

/// `clip_grad_norm` reads its norm back after its own commit. Any thread's
/// submit returns committed scratch to the pool, so a norm kept in scratch
/// could be recycled and overwritten by another caller before the read: it
/// came back as 0.0 (seen as a 1-in-10 failure of a parity test run in
/// parallel). Every returned norm must equal the host norm while other
/// threads churn the pool.
#[test]
fn clip_norms_stay_correct_while_other_threads_submit() {
    watchdog("clip norms", || {
        const THREADS: usize = 8;
        const ROUNDS: usize = 150;
        let g = Arc::new(open());
        let start = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|i| {
                let (g, start) = (Arc::clone(&g), Arc::clone(&start));
                std::thread::spawn(move || {
                    start.wait();
                    for round in 0..ROUNDS {
                        let seed = (i * ROUNDS + round) as u64;
                        if i % 2 == 0 {
                            let v = data(seed, 64);
                            let want = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                            let mut grads = vec![t(&g, seed, &[64]).unwrap()];
                            let got = g.clip_grad_norm(&mut grads, 1.0e30).unwrap();
                            let err = (f64::from(got) - want).abs() / want;
                            assert!(
                                err < 1e-5,
                                "thread {i} round {round}: norm {got}, host {want}"
                            );
                        } else {
                            let x = t(&g, seed, &[16]).unwrap();
                            let y = g.silu_forward(&x).unwrap();
                            bits(&g, &y).unwrap();
                        }
                    }
                })
            })
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            h.join().unwrap_or_else(|_| panic!("thread {i} panicked"));
        }
    });
}

#[test]
fn a_caller_that_panics_does_not_disturb_the_others() {
    watchdog("panicking caller", || {
        let g = Arc::new(open());
        let want = workload(&g, 3).unwrap();
        // A thread records ops, then panics before reading anything back.
        let bad = {
            let g = Arc::clone(&g);
            std::thread::spawn(move || {
                let x = t(&g, 900, &[64, 64]).unwrap();
                let _y = g.linear_forward(&x, &x).unwrap();
                panic!("caller panics with work recorded");
            })
        };
        assert!(bad.join().is_err());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let g = Arc::clone(&g);
                std::thread::spawn(move || workload(&g, 3).unwrap())
            })
            .collect();
        for h in handles {
            assert!(h.join().expect("thread panicked") == want);
        }
        g.sync().unwrap();
    });
}

#[test]
fn a_device_lost_mid_flight_fails_every_thread_cleanly() {
    watchdog("device lost", || {
        const THREADS: usize = 6;
        let g = Arc::new(open());
        let start = Arc::new(Barrier::new(THREADS + 1));
        let handles: Vec<_> = (0..THREADS)
            .map(|i| {
                let (g, start) = (Arc::clone(&g), Arc::clone(&start));
                std::thread::spawn(move || {
                    start.wait();
                    let deadline = Instant::now() + Duration::from_secs(120);
                    let mut ok = 0usize;
                    while Instant::now() < deadline {
                        match workload(&g, i as u64) {
                            Ok(_) => ok += 1,
                            Err(e) => return (ok, Some(e)),
                        }
                    }
                    (ok, None)
                })
            })
            .collect();
        start.wait();
        std::thread::sleep(Duration::from_millis(300));
        g.context().device().destroy();
        let mut errors = 0;
        for (i, h) in handles.into_iter().enumerate() {
            let (ok, err) = h.join().unwrap_or_else(|_| panic!("thread {i} panicked"));
            match err {
                Some(OjasError::Backend { .. }) => errors += 1,
                Some(other) => panic!("thread {i}: unexpected error kind {other:?}"),
                None => panic!("thread {i}: ran {ok} rounds past the loss without an error"),
            }
        }
        assert_eq!(errors, THREADS);
        match g.sync() {
            Err(OjasError::Backend { detail, .. }) => {
                assert!(detail.contains("lost"), "{detail}")
            }
            other => panic!("sync after the loss: {other:?}"),
        }
    });
}
