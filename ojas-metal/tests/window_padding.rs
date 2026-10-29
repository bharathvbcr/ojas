//! `window_partition` and `window_merge` on heights and widths that the window
//! does not divide, on Metal (ft-8d3aa337, criterion 2): the same checks as
//! `ojas-cpu/tests/window_padding.rs` (one body, included by `#[path]`), and
//! the dispatch counts that prove the divides path is untouched: it encodes
//! the permute alone, no `ojas_fill_words`, `ojas_concat_part` or `ojas_slice`.
//!
//! Run with `--test-threads=1`: the dispatch counts are per backend, but a
//! test that counts should not share the device with another.

#![cfg(target_os = "macos")]

mod common;
#[path = "../../ojas-cpu/tests/window_padding_suite/mod.rs"]
mod suite;

use common::*;
use ojas_core::{window_merge, window_partition, Backend, Tensor};
use ojas_metal::MetalBackend;

struct MetalRig(MetalBackend);

impl suite::Rig for MetalRig {
    type B = MetalBackend;
    fn backend(&self) -> &MetalBackend {
        &self.0
    }
    fn put(&self, data: &[f32], shape: &[usize]) -> Tensor {
        up(&self.0, &host(data, shape))
    }
    fn get(&self, t: &Tensor) -> Vec<f32> {
        ok("sync", self.0.sync());
        down(t)
    }
}

fn rig() -> MetalRig {
    MetalRig(metal())
}

#[test]
fn partition_matches_the_reference_and_merge_inverts_it() {
    suite::partition_matches_the_reference_and_merge_inverts_it(&rig());
}

#[test]
fn the_padding_is_positive_zero() {
    suite::the_padding_is_positive_zero(&rig());
}

#[test]
fn an_empty_extent_is_refused() {
    suite::an_empty_extent_is_refused(&rig());
}

#[test]
fn the_remaining_refusals_stand() {
    suite::the_remaining_refusals_stand(&rig());
}

#[test]
fn merge_refuses_sizes_the_windows_cannot_make() {
    suite::merge_refuses_sizes_the_windows_cannot_make(&rig());
}

const PAD_KERNELS: [&str; 3] = ["ojas_fill_words", "ojas_concat_part", "ojas_slice"];

/// Dispatches `f` encodes, and the dispatches of each pad kernel among them.
fn counted<T>(m: &MetalBackend, f: impl FnOnce() -> T) -> (T, u64, [u64; 3]) {
    ok("sync", m.sync());
    let all = m.dispatches();
    let kernels = PAD_KERNELS.map(|k| m.kernel_dispatches(k));
    let out = f();
    ok("sync", m.sync());
    let after = PAD_KERNELS.map(|k| m.kernel_dispatches(k));
    (
        out,
        m.dispatches() - all,
        [
            after[0] - kernels[0],
            after[1] - kernels[1],
            after[2] - kernels[2],
        ],
    )
}

/// The divides path encodes the permute and nothing else: the same dispatch
/// count as the permute alone, and no pad kernel. SpikeTrack's cut is 72 / 24.
/// The padded path adds, per padded axis, the `fill` (1 dispatch) and the
/// two-input `concat` (2); per cropped axis the `slice` (2: a scan and a copy,
/// as `concat_slice_fill.rs` pins).
#[test]
fn the_divides_path_dispatches_the_permute_alone_and_padding_adds_its_kernels() {
    let m = metal();
    let c = 2usize;
    let dims = [0usize, 1, 3, 2, 4, 5];
    let permute_alone = |h: usize, w: usize, win: usize| -> u64 {
        let x = up(&m, &rand(&[1, h, w, c], 900, 1.0));
        let split = ok("view", x.reshape(&[1, h / win, win, w / win, win, c]));
        counted(&m, || ok("permute", m.permute(&split, &dims))).1
    };
    // Divides: 72 x 72 in windows of 24, and a small cut.
    for (h, w, win) in [(72usize, 72usize, 24usize), (6, 9, 3)] {
        let base = permute_alone(h, w, win);
        assert!(base >= 1, "the permute is at least one dispatch");
        let x = up(&m, &rand(&[1, h, w, c], 901, 1.0));
        let (windows, n, pad) = counted(&m, || ok("partition", window_partition(&m, &x, win)));
        assert_eq!((n, pad), (base, [0, 0, 0]), "partition {h}x{w}/{win}");
        let (_, n, pad) = counted(&m, || ok("merge", window_merge(&m, &windows, h, w)));
        assert_eq!((n, pad), (base, [0, 0, 0]), "merge {h}x{w}/{win}");
    }
    // Padded: height only, width only, both. The permute is that of the padded
    // grid (8 x 8 in windows of 4).
    let base = permute_alone(8, 8, 4);
    for (h, w, axes) in [(5usize, 8usize, 1u64), (8, 5, 1), (5, 5, 2)] {
        let x = up(&m, &rand(&[1, h, w, c], 902, 1.0));
        let (windows, n, pad) = counted(&m, || ok("partition", window_partition(&m, &x, 4)));
        assert_eq!(
            (n, pad),
            (base + 3 * axes, [axes, 2 * axes, 0]),
            "partition {h}x{w}/4"
        );
        let (_, n, pad) = counted(&m, || ok("merge", window_merge(&m, &windows, h, w)));
        assert_eq!((n, pad), (base + 2 * axes, [0, 0, axes]), "merge {h}x{w}/4");
    }
}

#[test]
fn windows_and_merge_match_torch() {
    suite::windows_and_merge_match_torch(&rig());
}
