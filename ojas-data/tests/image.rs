//! MNIST idx and CIFAR-10 binary readers, and the seeded `f32` image
//! sampler. Fixtures are generated here; nothing is downloaded.

use ojas_core::{Budget, DataCursor};
use ojas_data::{
    augment_draw, Augment, ImageBatchSampler, ImageError, ImageSamplerConfig, ImageSet,
    Normalization, CIFAR10_RECORD_BYTES,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::{Cursor, Read, Seek, SeekFrom};

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

/// Counts allocations per thread, so parallel tests do not see each other.
struct Counting;

// SAFETY: every method forwards to `System` unchanged and only bumps a
// thread-local counter.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(Cell::get)
}

const GIB: u64 = 1 << 30;

fn pixel(i: usize, c: usize, y: usize, x: usize) -> u8 {
    (i.wrapping_mul(31) ^ c.wrapping_mul(97) ^ y.wrapping_mul(13) ^ x.wrapping_mul(5)) as u8
}

fn label(i: usize) -> u8 {
    (i % 10) as u8
}

fn idx_images(n: u32, rows: u32, cols: u32) -> Vec<u8> {
    let mut v = vec![0, 0, 8, 3];
    for d in [n, rows, cols] {
        v.extend_from_slice(&d.to_be_bytes());
    }
    for i in 0..n as usize {
        for y in 0..rows as usize {
            for x in 0..cols as usize {
                v.push(pixel(i, 0, y, x));
            }
        }
    }
    v
}

fn idx_labels(n: u32) -> Vec<u8> {
    let mut v = vec![0, 0, 8, 1];
    v.extend_from_slice(&n.to_be_bytes());
    v.extend((0..n as usize).map(label));
    v
}

fn cifar(n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * CIFAR10_RECORD_BYTES);
    for i in 0..n {
        v.push(label(i));
        for c in 0..3 {
            for y in 0..32 {
                for x in 0..32 {
                    v.push(pixel(i, c, y, x));
                }
            }
        }
    }
    v
}

fn mnist(img: Vec<u8>, lab: Vec<u8>, budget: &Budget) -> Result<ImageSet, ImageError> {
    ImageSet::from_mnist(Cursor::new(img), Cursor::new(lab), budget)
}

fn cifar_set(n: usize) -> ImageSet {
    ImageSet::from_cifar10(Cursor::new(cifar(n)), &Budget::new(GIB)).unwrap()
}

#[test]
fn mnist_idx_parses_the_real_layout() {
    let budget = Budget::new(GIB);
    let set = mnist(idx_images(5, 28, 28), idx_labels(5), &budget).unwrap();
    assert_eq!(
        (set.len(), set.channels(), set.height(), set.width()),
        (5, 1, 28, 28)
    );
    assert_eq!(set.image_len(), 784);
    assert_eq!(set.image(3)[2 * 28 + 5], pixel(3, 0, 2, 5));
    assert_eq!(set.label(4), 4);
    assert_eq!(budget.live_bytes().unwrap(), 5 * 784 + 5);
    drop(set);
    assert_eq!(
        budget.live_bytes().unwrap(),
        0,
        "the charge outlived the set"
    );
}

#[test]
fn cifar10_parses_3073_byte_records() {
    let set = cifar_set(12);
    assert_eq!(
        (set.len(), set.channels(), set.height(), set.width()),
        (12, 3, 32, 32)
    );
    assert_eq!(set.image(7)[2 * 1024 + 9 * 32 + 4], pixel(7, 2, 9, 4));
    assert_eq!(set.label(11), 1);
}

fn magic_err(img: Vec<u8>, lab: Vec<u8>) -> ImageError {
    mnist(img, lab, &Budget::new(GIB)).unwrap_err()
}

#[test]
fn wrong_magic_and_dtype_are_refused() {
    let mut img = idx_images(2, 4, 4);
    img[0] = 1;
    assert!(matches!(
        magic_err(img, idx_labels(2)),
        ImageError::BadMagic {
            got: 0x0100_0803,
            ..
        }
    ));
    let mut img = idx_images(2, 4, 4);
    img[3] = 2;
    assert!(matches!(
        magic_err(img, idx_labels(2)),
        ImageError::BadMagic { .. }
    ));
    let mut img = idx_images(2, 4, 4);
    img[2] = 0x0D;
    assert_eq!(
        magic_err(img, idx_labels(2)),
        ImageError::BadDtype { got: 0x0D }
    );
    let mut lab = idx_labels(2);
    lab[2] = 0x0B;
    assert_eq!(
        magic_err(idx_images(2, 4, 4), lab),
        ImageError::BadDtype { got: 0x0B }
    );
    // A label file in the image slot.
    assert!(matches!(
        magic_err(idx_labels(20), idx_labels(20)),
        ImageError::BadMagic { .. }
    ));
}

/// A reader that counts payload bytes handed out past the headers.
struct Probe {
    inner: Cursor<Vec<u8>>,
    read: usize,
}

impl Read for Probe {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.read += n;
        Ok(n)
    }
}

impl Seek for Probe {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(to)
    }
}

fn header_only(n: u32, rows: u32, cols: u32) -> Vec<u8> {
    let mut v = vec![0, 0, 8, 3];
    for d in [n, rows, cols] {
        v.extend_from_slice(&d.to_be_bytes());
    }
    v
}

#[test]
fn overflowing_dims_are_refused_before_any_allocation() {
    let before = allocs();
    let e = mnist(
        header_only(u32::MAX, u32::MAX, u32::MAX),
        Vec::new(),
        &Budget::new(GIB),
    )
    .unwrap_err();
    assert!(matches!(e, ImageError::Truncated { .. }), "{e:?}");
    let mut lab = vec![0, 0, 8, 1];
    lab.extend_from_slice(&u32::MAX.to_be_bytes());
    let e = mnist(
        header_only(u32::MAX, u32::MAX, u32::MAX),
        lab,
        &Budget::new(GIB),
    )
    .unwrap_err();
    assert_eq!(e, ImageError::DimensionOverflow);
    assert!(
        allocs() - before < 64,
        "a header-only refusal allocated a payload"
    );
}

#[test]
fn a_header_claiming_more_than_the_file_holds_is_truncated() {
    // 2 images declared, 1 present.
    let mut img = idx_images(2, 4, 4);
    img.truncate(16 + 16);
    let e = mnist(img, idx_labels(2), &Budget::new(GIB)).unwrap_err();
    assert_eq!(
        e,
        ImageError::Truncated {
            needed: 16 + 32,
            available: 16 + 16
        }
    );
    let mut lab = idx_labels(4);
    lab.pop();
    assert!(matches!(
        mnist(idx_images(4, 4, 4), lab, &Budget::new(GIB)).unwrap_err(),
        ImageError::Truncated { .. }
    ));
    // Shorter than a header.
    assert!(matches!(
        mnist(vec![0, 0, 8], idx_labels(1), &Budget::new(GIB)).unwrap_err(),
        ImageError::Truncated { .. }
    ));
    assert!(matches!(
        mnist(idx_images(1, 2, 2), vec![0, 0, 8, 1, 0], &Budget::new(GIB)).unwrap_err(),
        ImageError::Truncated { .. }
    ));
}

#[test]
fn trailing_bytes_and_a_cifar_remainder_are_refused() {
    let mut img = idx_images(2, 4, 4);
    img.push(0);
    assert_eq!(
        mnist(img, idx_labels(2), &Budget::new(GIB)).unwrap_err(),
        ImageError::TrailingBytes { extra: 1 }
    );
    let mut c = cifar(3);
    c.truncate(2 * CIFAR10_RECORD_BYTES + 100);
    assert_eq!(
        ImageSet::from_cifar10(Cursor::new(c), &Budget::new(GIB)).unwrap_err(),
        ImageError::TrailingBytes { extra: 100 }
    );
    assert_eq!(
        ImageSet::from_cifar10(Cursor::new(Vec::new()), &Budget::new(GIB)).unwrap_err(),
        ImageError::Empty
    );
}

#[test]
fn image_and_label_counts_must_agree_and_be_nonzero() {
    assert_eq!(
        magic_err(idx_images(3, 4, 4), idx_labels(2)),
        ImageError::CountMismatch {
            images: 3,
            labels: 2
        }
    );
    assert_eq!(
        magic_err(idx_images(0, 4, 4), idx_labels(0)),
        ImageError::Empty
    );
    assert_eq!(
        magic_err(idx_images(2, 0, 4), idx_labels(2)),
        ImageError::Empty
    );
}

#[test]
fn labels_of_ten_or_more_are_refused() {
    let mut lab = idx_labels(4);
    lab[8 + 2] = 10;
    assert_eq!(
        magic_err(idx_images(4, 4, 4), lab),
        ImageError::LabelOutOfRange {
            index: 2,
            label: 10
        }
    );
    let mut c = cifar(4);
    c[3 * CIFAR10_RECORD_BYTES] = 200;
    assert_eq!(
        ImageSet::from_cifar10(Cursor::new(c), &Budget::new(GIB)).unwrap_err(),
        ImageError::LabelOutOfRange {
            index: 3,
            label: 200
        }
    );
}

#[test]
fn a_payload_over_the_budget_is_refused_before_it_is_read() {
    let img = idx_images(8, 28, 28);
    let need = 8 * 784 + 8;
    let budget = Budget::new(need - 1);
    let mut probe = Probe {
        inner: Cursor::new(img.clone()),
        read: 0,
    };
    let e = ImageSet::from_mnist(&mut probe, Cursor::new(idx_labels(8)), &budget).unwrap_err();
    assert_eq!(
        e,
        ImageError::OverBudget {
            requested: need,
            cap: need - 1
        }
    );
    assert_eq!(
        probe.read, 16,
        "payload bytes were read before the budget check"
    );
    assert_eq!(budget.live_bytes().unwrap(), 0);
    // One byte more fits, exactly.
    let ok = mnist(img, idx_labels(8), &Budget::new(need)).unwrap();
    assert_eq!(ok.len(), 8);

    let c = cifar(4);
    let need = 4 * 3072 + 4;
    let mut probe = Probe {
        inner: Cursor::new(c),
        read: 0,
    };
    let e = ImageSet::from_cifar10(&mut probe, &Budget::new(need - 1)).unwrap_err();
    assert!(matches!(e, ImageError::OverBudget { .. }));
    assert_eq!(probe.read, 0);
}

/// Seeded xorshift for the mutation loop.
struct Xs(u64);

impl Xs {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn random_truncations_and_header_mutations_never_panic() {
    let img = idx_images(6, 5, 5);
    let lab = idx_labels(6);
    let cif = cifar(3);
    let small = Budget::new(1 << 20);
    let mut rng = Xs(0x5EED_1234_ABCD_9876);
    let (mut refused, mut accepted) = (0, 0);
    for round in 0..4000 {
        let mutate = |src: &[u8], rng: &mut Xs| {
            let mut v = src.to_vec();
            match rng.next() % 3 {
                0 => v.truncate((rng.next() % (v.len() as u64 + 1)) as usize),
                1 => {
                    let at = (rng.next() % 20.min(v.len() as u64)) as usize;
                    v[at] = rng.next() as u8;
                }
                _ => {
                    let at = (rng.next() % v.len() as u64) as usize;
                    v[at] ^= 1 << (rng.next() % 8);
                }
            }
            v
        };
        let (i, l) = (
            mutate(&img, &mut Xs(rng.next())),
            mutate(&lab, &mut Xs(rng.next())),
        );
        let c = mutate(&cif, &mut Xs(rng.next()));
        for r in [
            mnist(i, l, &small).map(|s| s.len()),
            ImageSet::from_cifar10(Cursor::new(c), &small).map(|s| s.len()),
        ] {
            match r {
                Ok(_) => accepted += 1,
                Err(_) => refused += 1,
            }
        }
        assert_eq!(
            small.live_bytes().unwrap(),
            0,
            "round {round} leaked a charge"
        );
    }
    assert!(
        refused > 1000 && accepted > 100,
        "{refused} refused, {accepted} accepted"
    );
}

fn cfg(batch: usize, seed: u64, aug: Augment, drop_last: bool) -> ImageSamplerConfig {
    ImageSamplerConfig {
        batch,
        seed,
        drop_last,
        normalization: Normalization::CIFAR10,
        augment: aug,
    }
}

fn run(
    set: &ImageSet,
    c: ImageSamplerConfig,
    from: DataCursor,
    batches: usize,
) -> Vec<(Vec<u32>, Vec<u32>, usize)> {
    let mut s = ImageBatchSampler::resume(set, c, from).unwrap();
    let mut x = vec![0.0f32; s.batch_floats()];
    let mut y = vec![0u32; c.batch];
    (0..batches)
        .map(|_| {
            let n = s.next_batch(&mut x, &mut y).unwrap();
            let floats = n * set.image_len();
            (
                x[..floats].iter().map(|v| v.to_bits()).collect(),
                y[..n].to_vec(),
                n,
            )
        })
        .collect()
}

#[test]
fn normalization_matches_the_pinned_constants() {
    // One image whose first pixels are 0, 128, 255 in channel 0.
    let mut rec = vec![3u8];
    rec.extend(std::iter::repeat_n(0u8, 3072));
    rec[1] = 0;
    rec[2] = 128;
    rec[3] = 255;
    let set = ImageSet::from_cifar10(Cursor::new(rec), &Budget::new(GIB)).unwrap();
    let mut s = ImageBatchSampler::new(&set, cfg(1, 1, Augment::NONE, false)).unwrap();
    let (mut x, mut y) = (vec![0.0f32; 3072], [0u32; 1]);
    assert_eq!(s.next_batch(&mut x, &mut y).unwrap(), 1);
    assert_eq!(y[0], 3);
    // (p / 255 - 0.4914) / 0.2470, recorded as f32 bit patterns.
    assert_eq!(x[0].to_bits(), PIN_NORM_0);
    assert_eq!(x[1].to_bits(), PIN_NORM_128);
    assert_eq!(x[2].to_bits(), PIN_NORM_255);
    // A zero pixel in channels 1 and 2 uses their own mean and std.
    assert_eq!(x[1024].to_bits(), PIN_NORM_CH1_0);
    assert_eq!(x[2048].to_bits(), PIN_NORM_CH2_0);
}

#[test]
fn augmentation_off_is_the_raw_decode_normalised() {
    let set = cifar_set(40);
    let mut s = ImageBatchSampler::new(&set, cfg(16, 9, Augment::NONE, false)).unwrap();
    let (mut x, mut y) = (vec![0.0f32; s.batch_floats()], vec![0u32; 16]);
    let n = s.next_batch(&mut x, &mut y).unwrap();
    assert_eq!(n, 16);
    let norm = Normalization::CIFAR10;
    let (mean, std) = ([0.4914f32, 0.4822, 0.4465], [0.2470f32, 0.2435, 0.2616]);
    assert_eq!(norm.channels(), 3);
    for k in 0..n {
        let idx = s.index_at(0, k).unwrap();
        assert_eq!(y[k], u32::from(set.label(idx)));
        for (j, &p) in set.image(idx).iter().enumerate() {
            let want = (f32::from(p) / 255.0 - mean[j / 1024]) / std[j / 1024];
            assert_eq!(
                x[k * 3072 + j].to_bits(),
                want.to_bits(),
                "image {k} elt {j}"
            );
        }
    }
}

#[test]
fn same_seed_same_order_and_augmentation_and_other_seeds_differ() {
    let set = cifar_set(70);
    let c = cfg(16, 42, Augment::CIFAR10, false);
    let a = run(&set, c, DataCursor::default(), 12);
    assert_eq!(a, run(&set, c, DataCursor::default(), 12));
    assert_ne!(
        a,
        run(
            &set,
            cfg(16, 43, Augment::CIFAR10, false),
            DataCursor::default(),
            12
        )
    );
    let off = run(
        &set,
        cfg(16, 42, Augment::NONE, false),
        DataCursor::default(),
        1,
    );
    assert_ne!(a[0].0, off[0].0, "augmentation changed nothing");
    assert_eq!(a[0].1, off[0].1, "augmentation moved the order");
}

#[test]
fn a_cursor_resumes_mid_epoch_and_across_epochs_bit_identically() {
    let set = cifar_set(70);
    let c = cfg(16, 5, Augment::CIFAR10, false);
    let whole = run(&set, c, DataCursor::default(), 14);
    for k in [1, 2, 4, 5, 9] {
        let mut s = ImageBatchSampler::new(&set, c).unwrap();
        let (mut x, mut y) = (vec![0.0f32; s.batch_floats()], vec![0u32; 16]);
        for _ in 0..k {
            s.next_batch(&mut x, &mut y).unwrap();
        }
        let cur = s.cursor();
        let tail = run(&set, c, cur, 14 - k);
        assert_eq!(
            tail,
            whole[k..],
            "resume after {k} batches (cursor {cur:?})"
        );
    }
}

#[test]
fn an_epoch_visits_every_image_once_and_the_tail_is_explicit() {
    let set = cifar_set(300);
    // drop_last off: 128, 128, 44, then the next epoch starts.
    let mut s = ImageBatchSampler::new(&set, cfg(128, 3, Augment::NONE, false)).unwrap();
    let (mut x, mut y) = (vec![0.0f32; s.batch_floats()], vec![0u32; 128]);
    let counts: Vec<usize> = (0..4)
        .map(|_| s.next_batch(&mut x, &mut y).unwrap())
        .collect();
    assert_eq!(counts, [128, 128, 44, 128]);
    assert_eq!(
        s.cursor(),
        DataCursor {
            shard: 1,
            token_index: 128
        }
    );
    let mut seen = vec![0u8; 300];
    for ord in 0..300 {
        seen[s.index_at(0, ord).unwrap()] += 1;
    }
    assert!(seen.iter().all(|&v| v == 1), "epoch 0 is not a permutation");
    let e1: Vec<usize> = (0..300).map(|o| s.index_at(1, o).unwrap()).collect();
    let e0: Vec<usize> = (0..300).map(|o| s.index_at(0, o).unwrap()).collect();
    assert_ne!(e0, e1, "epochs share one order");
    // drop_last on: 128, 128, then epoch 1.
    let mut d = ImageBatchSampler::new(&set, cfg(128, 3, Augment::NONE, true)).unwrap();
    let counts: Vec<usize> = (0..3)
        .map(|_| d.next_batch(&mut x, &mut y).unwrap())
        .collect();
    assert_eq!(counts, [128, 128, 128]);
    assert_eq!(
        d.cursor(),
        DataCursor {
            shard: 1,
            token_index: 128
        }
    );
}

/// Independent oracle: pad with zero pixels, crop, flip, normalise.
fn naive(set: &ImageSet, idx: usize, pad: usize, dy: usize, dx: usize, flip: bool) -> Vec<f32> {
    let (mean, std) = ([0.4914f32, 0.4822, 0.4465], [0.2470f32, 0.2435, 0.2616]);
    let side = 32 + 2 * pad;
    let mut out = Vec::new();
    for c in 0..3 {
        let mut padded = vec![0u8; side * side];
        for y in 0..32 {
            for x in 0..32 {
                padded[(y + pad) * side + x + pad] = set.image(idx)[c * 1024 + y * 32 + x];
            }
        }
        for y in 0..32 {
            for x in 0..32 {
                let fx = if flip { 31 - x } else { x };
                let p = padded[(y + dy) * side + fx + dx];
                out.push((f32::from(p) / 255.0 - mean[c]) / std[c]);
            }
        }
    }
    out
}

#[test]
fn crop_and_flip_match_an_independent_pad_crop_flip() {
    let set = cifar_set(64);
    let aug = Augment::CIFAR10;
    let mut s = ImageBatchSampler::new(&set, cfg(64, 77, aug, false)).unwrap();
    let (mut x, mut y) = (vec![0.0f32; s.batch_floats()], vec![0u32; 64]);
    assert_eq!(s.next_batch(&mut x, &mut y).unwrap(), 64);
    let (mut shifted, mut flipped) = (0, 0);
    for k in 0..64 {
        let idx = s.index_at(0, k).unwrap();
        let d = s.draw_for(0, idx);
        assert!(d.dy <= 8 && d.dx <= 8);
        shifted += usize::from(d.dy != 4 || d.dx != 4);
        flipped += usize::from(d.flip);
        let want = naive(&set, idx, 4, d.dy, d.dx, d.flip);
        let got: Vec<u32> = x[k * 3072..(k + 1) * 3072]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let want: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want, "image {idx} draw {d:?}");
    }
    assert!(
        shifted > 40 && (10..54).contains(&flipped),
        "{shifted} shifted, {flipped} flipped"
    );
}

#[test]
fn crop_and_flip_masks_are_pinned_for_a_seed() {
    let draws: Vec<(usize, usize, bool)> = (0..8)
        .map(|i| {
            let d = augment_draw(2024, 0, i, Augment::CIFAR10);
            (d.dy, d.dx, d.flip)
        })
        .collect();
    assert_eq!(draws, PIN_DRAWS);
    // The epoch and the image index both address the stream.
    assert_ne!(
        augment_draw(2024, 1, 0, Augment::CIFAR10),
        augment_draw(2024, 0, 0, Augment::CIFAR10)
    );
    // Turning the flip off leaves the crop mask where it was.
    for i in 0..64 {
        let both = augment_draw(5, 2, i, Augment::CIFAR10);
        let crop = augment_draw(
            5,
            2,
            i,
            Augment {
                crop_pad: 4,
                flip: false,
            },
        );
        assert_eq!((both.dy, both.dx), (crop.dy, crop.dx));
        assert!(!crop.flip);
    }
    // Every offset in 0..=8 occurs.
    let mut hit = [false; 9];
    for i in 0..400 {
        let d = augment_draw(1, 0, i, Augment::CIFAR10);
        hit[d.dy] = true;
        hit[d.dx] = true;
    }
    assert!(hit.iter().all(|&h| h));
}

#[test]
fn the_shuffle_is_pinned_for_a_seed() {
    let set = cifar_set(32);
    let mut s = ImageBatchSampler::new(&set, cfg(8, 2024, Augment::NONE, false)).unwrap();
    let order: Vec<usize> = (0..8).map(|o| s.index_at(0, o).unwrap()).collect();
    assert_eq!(order, PIN_ORDER);
}

#[test]
fn a_batch_of_128_cifar_images_allocates_nothing_after_warm_up() {
    let set = cifar_set(1024);
    let mut s = ImageBatchSampler::new(&set, cfg(128, 11, Augment::CIFAR10, false)).unwrap();
    let (mut x, mut y) = (vec![0.0f32; s.batch_floats()], vec![0u32; 128]);
    s.next_batch(&mut x, &mut y).unwrap();
    let before = allocs();
    // 20 batches of an 8-batch epoch: two epoch reshuffles are inside.
    for _ in 0..20 {
        assert_eq!(s.next_batch(&mut x, &mut y).unwrap(), 128);
    }
    assert_eq!(allocs() - before, 0, "next_batch allocated");
    // The counter sees allocations at all.
    let probe = allocs();
    let v = std::hint::black_box(vec![1u8; 64]);
    assert!(allocs() > probe);
    drop(v);
}

#[test]
fn bad_sampler_configurations_are_refused() {
    let set = cifar_set(10);
    let cfg0 = cfg(4, 1, Augment::NONE, false);
    let config = |c: ImageSamplerConfig, from| {
        matches!(
            ImageBatchSampler::resume(&set, c, from),
            Err(ImageError::Config(_))
        )
    };
    assert!(config(
        ImageSamplerConfig { batch: 0, ..cfg0 },
        DataCursor::default()
    ));
    assert!(config(
        ImageSamplerConfig {
            normalization: Normalization::MNIST,
            ..cfg0
        },
        DataCursor::default()
    ));
    assert!(config(
        cfg(11, 1, Augment::NONE, true),
        DataCursor::default()
    ));
    assert!(config(
        cfg0,
        DataCursor {
            shard: 0,
            token_index: 10
        }
    ));
    let mut s = ImageBatchSampler::new(&set, cfg0).unwrap();
    let before = s.cursor();
    let e = s
        .next_batch(&mut [0.0; 4 * 3072 - 1], &mut [0u32; 4])
        .unwrap_err();
    assert!(matches!(e, ImageError::Config(_)));
    assert!(matches!(
        s.next_batch(&mut [0.0; 4 * 3072], &mut [0u32; 3]),
        Err(ImageError::Config(_))
    ));
    assert_eq!(s.cursor(), before, "a refused call moved the cursor");
    assert!(Normalization::new(&[0.5], &[0.0]).is_err());
    assert!(Normalization::new(&[0.5, 0.5], &[1.0]).is_err());
    assert!(Normalization::new(&[f32::NAN], &[1.0]).is_err());
    assert!(Normalization::new(&[0.5; 4], &[1.0; 4]).is_err());
    assert!(Normalization::new(&[0.1307], &[0.3081]).is_ok());
}

#[test]
fn mnist_batches_are_single_channel() {
    let set = mnist(idx_images(20, 28, 28), idx_labels(20), &Budget::new(GIB)).unwrap();
    let c = ImageSamplerConfig {
        batch: 8,
        seed: 1,
        drop_last: true,
        normalization: Normalization::MNIST,
        augment: Augment::NONE,
    };
    let mut s = ImageBatchSampler::new(&set, c).unwrap();
    assert_eq!(s.batch_floats(), 8 * 784);
    let (mut x, mut y) = (vec![0.0f32; 8 * 784], [0u32; 8]);
    assert_eq!(s.next_batch(&mut x, &mut y).unwrap(), 8);
    let idx = s.index_at(0, 0).unwrap();
    let want = (f32::from(set.image(idx)[30]) / 255.0 - 0.1307) / 0.3081;
    assert_eq!(x[30].to_bits(), want.to_bits());
    assert_eq!(y[0], u32::from(set.label(idx)));
}

/// Time one 128-image CIFAR batch, min of N. Run with
/// `cargo test -p ojas-data --release --test image -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark"]
fn bench_cifar_batch_128() {
    let set = cifar_set(2048);
    for (name, aug) in [
        ("augmentation off", Augment::NONE),
        ("crop+flip", Augment::CIFAR10),
    ] {
        let mut s = ImageBatchSampler::new(&set, cfg(128, 1, aug, false)).unwrap();
        let (mut x, mut y) = (vec![0.0f32; s.batch_floats()], vec![0u32; 128]);
        s.next_batch(&mut x, &mut y).unwrap();
        let mut best = u128::MAX;
        for _ in 0..2000 {
            let t = std::time::Instant::now();
            s.next_batch(&mut x, &mut y).unwrap();
            best = best.min(t.elapsed().as_nanos());
        }
        println!("cifar batch 128, {name}: min of 2000 = {best} ns");
    }
}

const PIN_NORM_0: u32 = 3221137171;
const PIN_NORM_128: u32 = 1026498880;
const PIN_NORM_255: u32 = 1073989747;
const PIN_NORM_CH1_0: u32 = 3221060112;
const PIN_NORM_CH2_0: u32 = 3218765969;
/// `(dy, dx, flip)` of images 0..8, seed 2024, epoch 0, `Augment::CIFAR10`.
const PIN_DRAWS: [(usize, usize, bool); 8] = [
    (2, 8, true),
    (5, 1, true),
    (8, 3, true),
    (8, 2, false),
    (5, 3, false),
    (3, 4, true),
    (0, 8, true),
    (7, 4, false),
];
/// The first 8 dataset indices of seed 2024's epoch-0 shuffle of 32 images.
const PIN_ORDER: [usize; 8] = [29, 1, 13, 10, 30, 0, 26, 5];
