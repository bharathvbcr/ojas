//! `MetalBackend::wait_counts`: every waited commit is counted once, under
//! the trigger that made it wait, and an upload while work is recorded
//! waits only when it cannot ride inline in the command.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::Backend;
use ojas_metal::{MetalBackend, WaitCounts};

/// Open `m` with nothing recorded, so the next wait is the one under test.
fn settled(m: &MetalBackend) -> WaitCounts {
    ok("settle", m.sync());
    m.wait_counts()
}

#[test]
fn each_wait_is_counted_under_its_trigger() {
    let m = metal();
    let x = up(&m, &rand(&[64], 1, 1.0));

    let c0 = settled(&m);
    let _y = ok("silu", m.silu_forward(&x));
    ok("sync", m.sync());
    let sync = m.wait_counts().since(&c0);
    assert_eq!(
        sync,
        WaitCounts {
            sync: 1,
            ..WaitCounts::default()
        }
    );

    let c0 = settled(&m);
    let y = ok("silu", m.silu_forward(&x));
    down(&y);
    let read = m.wait_counts().since(&c0);
    assert_eq!(
        read,
        WaitCounts {
            read: 1,
            ..WaitCounts::default()
        }
    );

    let c0 = settled(&m);
    let mut grads = vec![ok("silu", m.silu_forward(&x))];
    ok("clip", m.clip_grad_norm(&mut grads, 1.0));
    let clip = m.wait_counts().since(&c0);
    assert_eq!(
        clip,
        WaitCounts {
            clip_norm: 1,
            ..WaitCounts::default()
        }
    );

    // Nothing recorded: a sync and a read do not wait, and count nothing.
    let c0 = settled(&m);
    ok("sync", m.sync());
    down(&x);
    assert_eq!(m.wait_counts().since(&c0), WaitCounts::default());
    // The total is the sum of the triggers.
    let all = m.wait_counts();
    assert_eq!(m.waits(), all.total());
}

#[test]
fn a_small_upload_while_work_is_recorded_rides_inline_and_does_not_wait() {
    // Pre-fix, every upload made while work was recorded waited for the
    // GPU, so each micro-batch's ids, targets and backward seed cost a wait.
    let m = metal();
    let x = up(&m, &rand(&[64], 2, 1.0));
    let c0 = settled(&m);
    let _pending = ok("silu", m.silu_forward(&x));
    let floats = [
        1.5f32,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::MIN_POSITIVE / 2.0,
        f32::MAX,
        -3.25,
        0.0,
    ];
    let ids: Vec<u32> = (0..64).map(|i| i * 3 + 1).collect();
    let finite = [1.5f32, -3.25, 0.0, 7.0];
    let f = up(&m, &host(&floats, &[floats.len()]));
    let u = up(&m, &host_u32(&ids, &[ids.len()]));
    let g = up(&m, &host(&finite, &[finite.len()]));
    // An op recorded after the upload reads the uploaded values.
    let doubled = ok("add", m.residual_add_forward(&g, &g));
    assert_eq!(
        m.wait_counts().since(&c0),
        WaitCounts::default(),
        "a small upload behind recorded work waited"
    );
    let twice = ok("download", m.download(&doubled));
    assert_eq!(ok("f32", twice.to_f32_vec()), vec![3.0, -6.5, 0.0, 14.0]);
    // Bit for bit, a NaN included: an upload is data, not a fault.
    let got = ok("download", m.download(&f));
    let got_bits: Vec<u32> = ok("f32", got.to_f32_vec())
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let want_bits: Vec<u32> = floats.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got_bits, want_bits);
    let got_ids = ok("download", m.download(&u));
    assert_eq!(ok("u32", got_ids.u32_slice()), ids.as_slice());
}

#[test]
fn a_large_upload_while_work_is_recorded_waits_and_is_counted_as_an_upload() {
    let m = metal();
    let x = up(&m, &rand(&[64], 3, 1.0));
    let c0 = settled(&m);
    let _pending = ok("silu", m.silu_forward(&x));
    // One word past the inline limit (64 KiB).
    let big = values((64 << 10) / 4 + 1, 4, 1.0);
    let d = up(&m, &host(&big, &[big.len()]));
    assert_eq!(
        m.wait_counts().since(&c0),
        WaitCounts {
            upload: 1,
            ..WaitCounts::default()
        }
    );
    assert_eq!(down(&d), big);
}

#[test]
fn inline_uploads_past_their_share_of_the_constant_arena_fall_back_to_a_wait() {
    // tessl resets its constant arena only at a waited commit, and a full
    // arena poisons the runtime: inline uploads stop at 1 MiB per window.
    let m = metal();
    let x = up(&m, &rand(&[64], 5, 1.0));
    let c0 = settled(&m);
    let _pending = ok("silu", m.silu_forward(&x));
    let chunk = (64 << 10) / 4;
    let mut sent = Vec::new();
    for i in 0..17u64 {
        let v = values(chunk, 10 + i, 1.0);
        let d = up(&m, &host(&v, &[chunk]));
        // Keep the window open: the next upload sees recorded work.
        let _keep = ok("silu", m.silu_forward(&x));
        sent.push((v, d));
    }
    // Sixteen 64 KiB uploads fill the 1 MiB share; the seventeenth waits,
    // and the wait opens a new window.
    assert_eq!(
        m.wait_counts().since(&c0),
        WaitCounts {
            upload: 1,
            ..WaitCounts::default()
        }
    );
    for (i, (v, d)) in sent.iter().enumerate() {
        assert_eq!(&down(d), v, "upload {i}");
    }
}
