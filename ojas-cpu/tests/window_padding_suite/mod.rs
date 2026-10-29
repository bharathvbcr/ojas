//! The padded `window_partition` / `window_merge` checks that hold on every
//! backend that runs `permute`, `fill`, `concat` and `slice`: one body, run on
//! the CPU by `ojas-cpu/tests/window_padding.rs` and on Metal by
//! `ojas-metal/tests/window_padding.rs` (which includes this file by `#[path]`).
//! A [`Rig`] says how a backend makes a tensor from host values and reads one
//! back; nothing here is specific to a backend.

#![allow(dead_code)]

use ojas_core::{window_merge, window_partition, Backend, OjasError, Tensor};

/// A backend, and how to move host values on and off it.
pub trait Rig {
    type B: Backend;
    fn backend(&self) -> &Self::B;
    /// A tensor of `shape` holding `data`, on the backend.
    fn put(&self, data: &[f32], shape: &[usize]) -> Tensor;
    /// The values of a tensor made by the backend, row-major.
    fn get(&self, t: &Tensor) -> Vec<f32>;
}

/// `(batch, height, width, channels, window)`.
pub type Case = (usize, usize, usize, usize, usize);

/// Every way the cut can fall: both axes padded, one axis padded, a window
/// larger than the height, the width or both, an extent of 1, a window of 1,
/// and the divides path.
pub const CASES: &[Case] = &[
    (2, 5, 7, 3, 4),
    (1, 7, 7, 2, 3),
    (2, 5, 8, 3, 4),
    (1, 8, 6, 2, 4),
    (1, 3, 9, 2, 5),
    (1, 9, 3, 2, 5),
    (2, 3, 2, 2, 5),
    (1, 1, 9, 2, 4),
    (2, 6, 1, 2, 3),
    (1, 1, 1, 1, 3),
    (2, 4, 6, 3, 2),
    (1, 5, 5, 1, 1),
    (1, 4, 4, 1, 4),
];

/// Distinct, finite values, none of them 0 except a `-0.0` every 11th, so a
/// padded zero cannot be mistaken for a value and a sign survives a move.
pub fn pattern(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            if i % 11 == 0 {
                -0.0
            } else {
                (i as f32 + 1.0) * 0.37 - 5.0
            }
        })
        .collect()
}

/// `window_partition` as SAM writes it, from scratch: window `(gy, gx)` of
/// image `bi` holds `x[bi, gy * win + i, gx * win + j, :]` where that is inside
/// the image and `+0.0` where it is not.
pub fn reference_windows(x: &[f32], case: Case) -> Vec<f32> {
    let (b, h, w, c, win) = case;
    let (nh, nw) = (h.div_ceil(win), w.div_ceil(win));
    let mut out = vec![0.0f32; b * nh * nw * win * win * c];
    for bi in 0..b {
        for gy in 0..nh {
            for gx in 0..nw {
                for i in 0..win {
                    for j in 0..win {
                        let (y, xx) = (gy * win + i, gx * win + j);
                        if y >= h || xx >= w {
                            continue;
                        }
                        for ch in 0..c {
                            let from = ((bi * h + y) * w + xx) * c + ch;
                            let to = ((((bi * nh + gy) * nw + gx) * win + i) * win + j) * c + ch;
                            out[to] = x[from];
                        }
                    }
                }
            }
        }
    }
    out
}

pub fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn shape_op(r: Result<Tensor, OjasError>, op: &str, what: &str) -> String {
    match r {
        Err(OjasError::Shape { op: got, detail }) => {
            assert_eq!(got, op, "{what}: op name");
            detail
        }
        other => panic!("{what}: expected a Shape refusal, got {other:?}"),
    }
}

fn range_op(r: Result<Tensor, OjasError>, op: &str, what: &str) {
    match r {
        Err(OjasError::OutOfRange { op: got, .. }) => assert_eq!(got, op, "{what}: op name"),
        other => panic!("{what}: expected an OutOfRange refusal, got {other:?}"),
    }
}

/// The windows are torch's, bit for bit (a from-scratch reference of SAM's
/// pad, view, permute, reshape), the shape is `[B * ceil * ceil, w, w, C]`,
/// and `window_merge` of them is the input, bit for bit.
pub fn partition_matches_the_reference_and_merge_inverts_it<R: Rig>(rig: &R) {
    let be = rig.backend();
    for &case in CASES {
        let (b, h, w, c, win) = case;
        let what = format!("{case:?}");
        let data = pattern(b * h * w * c);
        let x = rig.put(&data, &[b, h, w, c]);
        let windows = window_partition(be, &x, win).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let count = b * h.div_ceil(win) * w.div_ceil(win);
        assert_eq!(windows.shape(), &[count, win, win, c], "{what}: shape");
        let got = rig.get(&windows);
        assert_eq!(
            bits(&got),
            bits(&reference_windows(&data, case)),
            "{what}: windows"
        );
        let merged = window_merge(be, &windows, h, w).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert_eq!(merged.shape(), &[b, h, w, c], "{what}: merged shape");
        assert_eq!(bits(&rig.get(&merged)), bits(&data), "{what}: round trip");
    }
}

/// Every element a window holds beyond the image is exactly `+0.0`, never
/// `-0.0` and never a leftover value: the input here is `-1.5` throughout, so
/// a padded cell that kept any input bit would show.
pub fn the_padding_is_positive_zero<R: Rig>(rig: &R) {
    let be = rig.backend();
    for &case in CASES {
        let (b, h, w, c, win) = case;
        let what = format!("{case:?}");
        let x = rig.put(&vec![-1.5f32; b * h * w * c], &[b, h, w, c]);
        let windows = window_partition(be, &x, win).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let got = bits(&rig.get(&windows));
        let image = b * h * w * c;
        let kept = got.iter().filter(|&&v| v == (-1.5f32).to_bits()).count();
        let zeros = got.iter().filter(|&&v| v == 0).count();
        assert_eq!(kept, image, "{what}: every input element lands once");
        assert_eq!(
            zeros,
            got.len() - image,
            "{what}: every other element is +0.0 (bits 0)"
        );
    }
}

/// An empty extent is refused under the window op's name, as `concat`,
/// `slice` and `fill` refuse an empty tensor, rather than returned empty.
pub fn an_empty_extent_is_refused<R: Rig>(rig: &R) {
    let be = rig.backend();
    for shape in [[1usize, 0, 4, 2], [1, 4, 0, 2], [0, 4, 4, 2]] {
        let empty = Tensor::from_f32(&[], &shape, be.budget()).unwrap();
        let d = shape_op(
            window_partition(be, &empty, 2),
            "window_partition",
            &format!("empty {shape:?}"),
        );
        assert!(d.contains("empty"), "{d}");
    }
    let windows = rig.put(&pattern(4 * 2 * 2 * 3), &[4, 2, 2, 3]);
    for (h, w) in [(0usize, 4usize), (4, 0), (0, 0)] {
        let d = shape_op(
            window_merge(be, &windows, h, w),
            "window_merge",
            &format!("merge to {h} x {w}"),
        );
        assert!(d.contains("empty"), "{d}");
    }
}

/// A window of 0 is refused by the partition; windows that are not square
/// are refused by the merge.
pub fn the_remaining_refusals_stand<R: Rig>(rig: &R) {
    let be = rig.backend();
    let x = rig.put(&pattern(4 * 6 * 3), &[1, 4, 6, 3]);
    let d = shape_op(window_partition(be, &x, 0), "window_partition", "window 0");
    assert!(d.contains("window is 0"), "{d}");
    let windows = rig.put(&pattern(3 * 2 * 4 * 3), &[3, 2, 4, 3]);
    let d = shape_op(
        window_merge(be, &windows, 4, 4),
        "window_merge",
        "non-square windows",
    );
    assert!(d.contains("not square"), "{d}");
}

/// `window_merge` derives the grid from the size and the window, so a size the
/// windows cannot make is a refusal, never a wrong or out-of-bounds read: a
/// height or width past what `count` windows cover, a count the grid does not
/// divide, and sizes whose grid overflows `usize`.
pub fn merge_refuses_sizes_the_windows_cannot_make<R: Rig>(rig: &R) {
    let be = rig.backend();
    // Four 2 x 2 windows of 3 channels: a 1-image grid of 2 x 2 windows.
    let windows = rig.put(&pattern(4 * 2 * 2 * 3), &[4, 2, 2, 3]);
    // The grid of 2 x 2 windows covers heights 3..=4 and widths 3..=4, and
    // any of those merges (the padding is cropped).
    for (h, w) in [(3usize, 3usize), (3, 4), (4, 3), (4, 4)] {
        let merged = window_merge(be, &windows, h, w).unwrap_or_else(|e| panic!("{h}x{w}: {e:?}"));
        assert_eq!(merged.shape(), &[1, h, w, 3], "{h}x{w}");
    }
    // Past the grid: more rows or columns than 4 windows can hold.
    for (h, w) in [(5usize, 4usize), (4, 5), (100, 2), (2, 100), (6, 6)] {
        let d = shape_op(
            window_merge(be, &windows, h, w),
            "window_merge",
            &format!("{h} x {w} past the grid"),
        );
        assert!(d.contains("window count"), "{h}x{w}: {d}");
    }
    // A count the grid of the size does not divide.
    let five = rig.put(&pattern(5 * 2 * 2 * 3), &[5, 2, 2, 3]);
    for (h, w) in [(3usize, 3usize), (4, 4), (2, 4), (6, 2)] {
        let d = shape_op(
            window_merge(be, &five, h, w),
            "window_merge",
            &format!("5 windows into {h} x {w}"),
        );
        assert!(d.contains("window count"), "{h}x{w}: {d}");
    }
    // Sizes whose rounded-up cover or window grid overflows `usize`.
    range_op(
        window_merge(be, &windows, usize::MAX, 2),
        "window_merge",
        "height usize::MAX",
    );
    range_op(
        window_merge(be, &windows, 2, usize::MAX),
        "window_merge",
        "width usize::MAX",
    );
    range_op(
        window_merge(be, &windows, 1 << 40, 1 << 40),
        "window_merge",
        "grid of 2^78 windows",
    );
    // An even extent just under the maximum rounds up fine; its window count
    // cannot divide four, which is a refusal and not a wrapped product.
    shape_op(
        window_merge(be, &windows, usize::MAX - 1, 2),
        "window_merge",
        "height usize::MAX - 1",
    );
    // A window of usize::MAX covers a 3 x 3 image in one window whose padding
    // cannot be sized: refused as out of range (by `fill`, whose product
    // overflows), before anything is allocated.
    let x = rig.put(&pattern(3 * 3), &[1, 3, 3, 1]);
    assert!(
        matches!(
            window_partition(be, &x, usize::MAX),
            Err(OjasError::OutOfRange { .. })
        ),
        "window usize::MAX"
    );
}

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../ojas-oracle/fixtures/window_pad/golden.txt"
);

fn hex_values(line: &str, key: &str) -> Vec<f32> {
    let rest = line
        .strip_prefix(key)
        .unwrap_or_else(|| panic!("expected `{key}`, got `{}`", &line[..line.len().min(20)]));
    rest.split_whitespace()
        .map(|h| f32::from_bits(u32::from_str_radix(h, 16).expect("hex bits")))
        .collect()
}

/// Torch's windows and merge, bit for bit, on every case of
/// `ojas-oracle/fixtures/window_pad/golden.txt` (SAM 3's `window_partition`
/// and `window_unpartition`, generated by `ojas-oracle/python/
/// window_pad_fixture.py`). A missing or short fixture fails: an unrun check
/// is not a pass.
pub fn windows_and_merge_match_torch<R: Rig>(rig: &R) {
    let be = rig.backend();
    let text = std::fs::read_to_string(GOLDEN).unwrap_or_else(|e| panic!("{GOLDEN}: {e}"));
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("ojas-window-pad-golden-v1"));
    assert!(lines.next().is_some_and(|l| l.starts_with("torch ")));
    let (mut cases, mut padded) = (0, 0);
    while let Some(head) = lines.next() {
        let f: Vec<&str> = head.split_whitespace().collect();
        assert_eq!(f[0], "case", "{head}");
        let n: Vec<usize> = f[2..].iter().map(|v| v.parse().expect("size")).collect();
        let (b, h, w, c, win) = (n[0], n[1], n[2], n[3], n[4]);
        let x = hex_values(lines.next().expect("x line"), "x ");
        let want_windows = hex_values(lines.next().expect("windows line"), "windows ");
        let want_merged = hex_values(lines.next().expect("merged line"), "merged ");
        assert_eq!(lines.next(), Some("end"));
        let what = f[1];
        let windows = window_partition(be, &rig.put(&x, &[b, h, w, c]), win)
            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert_eq!(
            bits(&rig.get(&windows)),
            bits(&want_windows),
            "{what}: windows against torch"
        );
        // Merge torch's own windows, not ours, so the two directions are
        // checked against torch independently.
        let torch_windows = rig.put(&want_windows, windows.shape());
        let merged =
            window_merge(be, &torch_windows, h, w).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert_eq!(
            bits(&rig.get(&merged)),
            bits(&want_merged),
            "{what}: merge against torch"
        );
        cases += 1;
        padded += usize::from(h % win != 0 || w % win != 0);
    }
    assert_eq!((cases, padded), (13, 10), "the fixture's cases");
}
