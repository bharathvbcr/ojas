//! `window_partition` and `window_merge` on heights and widths that the window
//! does not divide (ft-8d3aa337, criterion 2): zero padding at the bottom and
//! right, and the crop back, as SAM's `window_partition` and
//! `window_unpartition` do. The checks that hold on any backend that runs
//! `permute`, `fill`, `concat` and `slice` are in `window_padding_suite` (the
//! Metal run of the same body is `ojas-metal/tests/window_padding.rs`); this
//! file adds what a counting, refusing backend can pin: the divides path never
//! calls `fill`, `concat` or `slice`, and a backend without them refuses the
//! padded path under the window op's name.

mod window_padding_suite;

use std::sync::atomic::{AtomicUsize, Ordering};

use ojas_core::{
    forward_backend_methods, window_merge, window_partition, Backend, BackendId, Budget, DType,
    Numerics, OjasError, Tensor,
};
use ojas_cpu::CpuBackend;
use window_padding_suite as suite;
use window_padding_suite::Rig;

struct CpuRig(CpuBackend);

impl suite::Rig for CpuRig {
    type B = CpuBackend;
    fn backend(&self) -> &CpuBackend {
        &self.0
    }
    fn put(&self, data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_f32(data, shape, self.0.budget()).unwrap()
    }
    fn get(&self, t: &Tensor) -> Vec<f32> {
        t.to_f32_vec().unwrap()
    }
}

fn rig() -> CpuRig {
    CpuRig(
        CpuBackend::with_threads(Budget::new(1 << 28), 1)
            .unwrap()
            .with_numerics(Numerics::Exact),
    )
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

/// A CPU backend that counts its `fill`, `concat` and `slice` calls and, per
/// flag, refuses them the way a backend without the op does
/// ([`OjasError::Unsupported`] naming the op).
struct Gate {
    inner: CpuBackend,
    refuse_fill: bool,
    refuse_concat: bool,
    refuse_slice: bool,
    fills: AtomicUsize,
    concats: AtomicUsize,
    slices: AtomicUsize,
}

macro_rules! all_but_pad_ops {
    (fill, { $($item:tt)* }) => {};
    (concat, { $($item:tt)* }) => {};
    (slice, { $($item:tt)* }) => {};
    ($name:ident, { $($item:tt)* }) => {
        $($item)*
    };
}

fn unsupported(op: &'static str) -> OjasError {
    OjasError::Unsupported {
        op,
        detail: format!("{:?} backend does not implement {op}", BackendId::Cpu),
    }
}

impl Backend for Gate {
    ojas_core::with_backend_methods!(
        forward_backend_methods,
        { self, self.inner, none, all_but_pad_ops }
    );

    fn fill(&self, shape: &[usize], dtype: DType, value: f32) -> Result<Tensor, OjasError> {
        self.fills.fetch_add(1, Ordering::SeqCst);
        if self.refuse_fill {
            return Err(unsupported("fill"));
        }
        self.inner.fill(shape, dtype, value)
    }

    fn concat(&self, inputs: &[&Tensor], axis: usize) -> Result<Tensor, OjasError> {
        self.concats.fetch_add(1, Ordering::SeqCst);
        if self.refuse_concat {
            return Err(unsupported("concat"));
        }
        self.inner.concat(inputs, axis)
    }

    fn slice(
        &self,
        input: &Tensor,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Result<Tensor, OjasError> {
        self.slices.fetch_add(1, Ordering::SeqCst);
        if self.refuse_slice {
            return Err(unsupported("slice"));
        }
        self.inner.slice(input, axis, start, len)
    }
}

impl Gate {
    fn new(refuse_fill: bool, refuse_concat: bool, refuse_slice: bool) -> Self {
        Gate {
            inner: CpuBackend::with_threads(Budget::new(1 << 28), 1)
                .unwrap()
                .with_numerics(Numerics::Exact),
            refuse_fill,
            refuse_concat,
            refuse_slice,
            fills: AtomicUsize::new(0),
            concats: AtomicUsize::new(0),
            slices: AtomicUsize::new(0),
        }
    }

    fn counts(&self) -> (usize, usize, usize) {
        (
            self.fills.load(Ordering::SeqCst),
            self.concats.load(Ordering::SeqCst),
            self.slices.load(Ordering::SeqCst),
        )
    }

    fn put(&self, data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_f32(data, shape, self.inner.budget()).unwrap()
    }
}

/// Condition on the divides path: when the window divides both extents there
/// is no `fill`, `concat` or `slice` call, so a backend that refuses all
/// three runs it, with the bits of the plain CPU. SpikeTrack's cut is 72 / 24;
/// the other shapes cover a window of 1 and a window the size of the image.
#[test]
fn the_divides_path_never_calls_fill_concat_or_slice() {
    let gate = Gate::new(true, true, true);
    let plain = rig();
    for (b, h, w, c, win) in [
        (1usize, 72usize, 72usize, 2usize, 24usize),
        (2, 6, 9, 3, 3),
        (1, 5, 5, 1, 1),
        (1, 4, 4, 1, 4),
    ] {
        let what = format!("{b}x{h}x{w}x{c} window {win}");
        let data = suite::pattern(b * h * w * c);
        let x = gate.put(&data, &[b, h, w, c]);
        let windows = window_partition(&gate, &x, win).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let want =
            window_partition(plain.backend(), &plain.put(&data, &[b, h, w, c]), win).unwrap();
        assert_eq!(
            suite::bits(&windows.to_f32_vec().unwrap()),
            suite::bits(&want.to_f32_vec().unwrap()),
            "{what}: windows"
        );
        let merged =
            window_merge(&gate, &windows, h, w).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert_eq!(
            suite::bits(&merged.to_f32_vec().unwrap()),
            suite::bits(&data),
            "{what}: round trip"
        );
        assert_eq!(gate.counts(), (0, 0, 0), "{what}: no pad op was called");
    }
}

/// One `fill` and one `concat` per padded axis on the partition, one `slice`
/// per cropped axis on the merge, and nothing for an axis that divides.
#[test]
fn padding_costs_a_fill_and_a_concat_per_padded_axis_and_a_slice_per_cropped_axis() {
    // (h, w, padded axes) at window 4, batch 1, one channel.
    for (h, w, axes) in [(5usize, 8usize, 1usize), (8, 5, 1), (5, 5, 2), (3, 3, 2)] {
        let gate = Gate::new(false, false, false);
        let data = suite::pattern(h * w);
        let x = gate.put(&data, &[1, h, w, 1]);
        let windows = window_partition(&gate, &x, 4).unwrap();
        assert_eq!(gate.counts(), (axes, axes, 0), "partition {h}x{w}");
        let merged = window_merge(&gate, &windows, h, w).unwrap();
        assert_eq!(gate.counts(), (axes, axes, axes), "merge {h}x{w}");
        assert_eq!(
            suite::bits(&merged.to_f32_vec().unwrap()),
            suite::bits(&data),
            "round trip {h}x{w}"
        );
    }
}

/// A backend without `fill`, `concat` or `slice` refuses the padded path under
/// the window op's name (the way wgpu does), not with a bare `fill` or
/// `concat` refusal; the detail still names the op that is missing.
#[test]
fn a_backend_without_the_pad_ops_refuses_under_the_window_ops_name() {
    let data = suite::pattern(5 * 7);
    for (fill, concat, missing) in [(true, false, "fill"), (false, true, "concat")] {
        let gate = Gate::new(fill, concat, false);
        let x = gate.put(&data, &[1, 5, 7, 1]);
        match window_partition(&gate, &x, 4) {
            Err(OjasError::Unsupported { op, detail }) => {
                assert_eq!(op, "window_partition", "refusal without {missing}");
                assert!(detail.contains(missing), "{detail}");
                assert!(detail.contains("window_partition"), "{detail}");
            }
            other => panic!("without {missing}: expected Unsupported, got {other:?}"),
        }
    }
    // The merge's crop needs slice. Windows from a padded partition on a
    // backend that has the pad ops, then a merge on one that lacks slice.
    let good = Gate::new(false, false, false);
    let windows = window_partition(&good, &good.put(&data, &[1, 5, 7, 1]), 4).unwrap();
    let gate = Gate::new(false, false, true);
    match window_merge(&gate, &windows, 5, 7) {
        Err(OjasError::Unsupported { op, detail }) => {
            assert_eq!(op, "window_merge");
            assert!(detail.contains("slice"), "{detail}");
        }
        other => panic!("without slice: expected Unsupported, got {other:?}"),
    }
}

#[test]
fn windows_and_merge_match_torch() {
    suite::windows_and_merge_match_torch(&rig());
}
