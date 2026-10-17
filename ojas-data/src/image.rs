//! MNIST idx and CIFAR-10 binary readers, and a seeded, shuffled, resumable
//! `f32` batch sampler over them.
//!
//! The readers are std-only (no decoder crate). Pixels stay `u8` on the host
//! and are widened to `f32` per batch; there is no `U8` `DType`. Every size
//! in a header is checked against the bytes the stream really has, and the
//! total against the caller's [`Budget`], before any payload is read or
//! allocated. See `docs/data-formats.md`.
//!
//! [`ImageBatchSampler`] visits every image once per epoch in an order keyed
//! by `(seed, epoch)`. The augmentation of an image is keyed by
//! `(seed, epoch, image index)`, so it does not depend on the batch size or
//! on where a run was resumed. A batch never straddles an epoch.

use crate::error::DataError;
use crate::rng::counter_stream;
use ojas_core::{Budget, DataCursor, Reservation};
use std::fmt;
use std::io::{Read, Seek, SeekFrom};

/// idx magic of an `u8`, 3-dimensional image file.
pub const MNIST_IMAGES_MAGIC: u32 = 0x0000_0803;
/// idx magic of an `u8`, 1-dimensional label file.
pub const MNIST_LABELS_MAGIC: u32 = 0x0000_0801;
/// One CIFAR-10 record: a label byte, then 3 x 32 x 32 `u8` in CHW order.
pub const CIFAR10_RECORD_BYTES: usize = 3073;
pub const CIFAR10_SIDE: usize = 32;
pub const CIFAR10_CHANNELS: usize = 3;
/// Labels are `0..NUM_CLASSES` in both formats.
pub const NUM_CLASSES: u8 = 10;
/// Most channels a [`Normalization`] carries.
pub const MAX_CHANNELS: usize = 3;
const IDX_IMAGE_HEADER_BYTES: u64 = 16;
const IDX_LABEL_HEADER_BYTES: u64 = 8;
const CIFAR_CHUNK_RECORDS: usize = 8;
const READ_CHUNK_BYTES: usize = 64 * 1024;
/// Domain tag of the per-epoch shuffle stream; image indices are below
/// `2^32`, so it never equals an image's augmentation stream.
const SHUFFLE_STREAM: u64 = u64::MAX;

/// Why an image file or a sampler configuration was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    /// The idx magic's two leading zero bytes or its dimension count is wrong.
    BadMagic { expected: u32, got: u32 },
    /// The idx dtype byte is not `0x08` (`u8`).
    BadDtype { got: u8 },
    /// A header's dimensions multiply past `u64` or `usize`.
    DimensionOverflow,
    /// The header declares more bytes than the stream has.
    Truncated { needed: u64, available: u64 },
    /// The stream is longer than its header or record size accounts for.
    TrailingBytes { extra: u64 },
    /// Image and label files declare different counts.
    CountMismatch { images: u64, labels: u64 },
    /// A label is `>= NUM_CLASSES`.
    LabelOutOfRange { index: u64, label: u8 },
    /// The payload does not fit the budget; nothing was read or allocated.
    OverBudget { requested: u64, cap: u64 },
    /// The allocator refused a buffer.
    Allocation { bytes: u64 },
    /// A zero count or zero-sized image.
    Empty,
    /// The sampler configuration or cursor is unusable.
    Config(String),
    /// An I/O error other than a short read.
    Io(String),
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic { expected, got } => {
                write!(f, "image: magic {got:#010x}, expected {expected:#010x}")
            }
            Self::BadDtype { got } => write!(f, "image: idx dtype byte {got:#04x}, expected 0x08"),
            Self::DimensionOverflow => f.write_str("image: header dimensions overflow"),
            Self::Truncated { needed, available } => write!(
                f,
                "image: header needs {needed} bytes, the stream has {available}"
            ),
            Self::TrailingBytes { extra } => {
                write!(f, "image: {extra} bytes past the declared end")
            }
            Self::CountMismatch { images, labels } => {
                write!(f, "image: {images} images but {labels} labels")
            }
            Self::LabelOutOfRange { index, label } => {
                write!(
                    f,
                    "image: label {label} at index {index} is not below {NUM_CLASSES}"
                )
            }
            Self::OverBudget { requested, cap } => {
                write!(f, "image: {requested} bytes exceed the {cap} byte budget")
            }
            Self::Allocation { bytes } => write!(f, "image: allocation of {bytes} bytes refused"),
            Self::Empty => f.write_str("image: no images, or a zero-sized image"),
            Self::Config(detail) => write!(f, "image sampler: {detail}"),
            Self::Io(detail) => write!(f, "image: {detail}"),
        }
    }
}

impl std::error::Error for ImageError {}

impl From<ImageError> for DataError {
    fn from(e: ImageError) -> Self {
        DataError::new(e.to_string())
    }
}

/// `u8` images with `u8` labels, held on the host. The bytes are charged to
/// the [`Budget`] they were read under until this value drops.
#[derive(Debug)]
pub struct ImageSet {
    // Declared before the reservation: the buffers free, then the charge releases.
    pixels: Vec<u8>,
    labels: Vec<u8>,
    len: usize,
    channels: usize,
    height: usize,
    width: usize,
    _charge: Reservation,
}

/// Fill `buf` (capacity already reserved) with exactly `want` bytes.
fn read_payload<R: Read>(r: &mut R, buf: &mut Vec<u8>, want: usize) -> Result<(), ImageError> {
    let mut chunk = [0u8; READ_CHUNK_BYTES];
    while buf.len() < want {
        let take = (want - buf.len()).min(READ_CHUNK_BYTES);
        r.read_exact(&mut chunk[..take]).map_err(io_error)?;
        buf.extend_from_slice(&chunk[..take]);
    }
    Ok(())
}

fn io_error(e: std::io::Error) -> ImageError {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        ImageError::Truncated {
            needed: 0,
            available: 0,
        }
    } else {
        ImageError::Io(e.to_string())
    }
}

fn stream_len<R: Seek>(r: &mut R) -> Result<u64, ImageError> {
    let len = r
        .seek(SeekFrom::End(0))
        .map_err(|e| ImageError::Io(e.to_string()))?;
    r.seek(SeekFrom::Start(0))
        .map_err(|e| ImageError::Io(e.to_string()))?;
    Ok(len)
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

fn buffer<T>(len: usize) -> Result<Vec<T>, ImageError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| ImageError::Allocation {
            bytes: (len as u64).saturating_mul(std::mem::size_of::<T>() as u64),
        })?;
    Ok(v)
}

/// Validate an idx magic: two zero bytes, dtype `0x08`, `dims` dimensions.
fn check_idx_magic(magic: u32, expected: u32) -> Result<(), ImageError> {
    let [z0, z1, dtype, dims] = magic.to_be_bytes();
    if z0 != 0 || z1 != 0 || dims != expected.to_be_bytes()[3] {
        return Err(ImageError::BadMagic {
            expected,
            got: magic,
        });
    }
    if dtype != 0x08 {
        return Err(ImageError::BadDtype { got: dtype });
    }
    Ok(())
}

fn read_header<R: Read>(r: &mut R, bytes: usize) -> Result<[u8; 16], ImageError> {
    let mut h = [0u8; 16];
    r.read_exact(&mut h[..bytes]).map_err(io_error)?;
    Ok(h)
}

impl ImageSet {
    /// Read an MNIST idx image file and its label file.
    ///
    /// Refuses, before reading any payload: a wrong magic or dtype, a
    /// dimension product that overflows, a stream shorter or longer than its
    /// header says, differing image and label counts, and a payload over
    /// `budget`. Labels `>= 10` are refused after the label bytes are read.
    pub fn from_mnist<I: Read + Seek, L: Read + Seek>(
        mut images: I,
        mut labels: L,
        budget: &Budget,
    ) -> Result<Self, ImageError> {
        let (img_len, lab_len) = (stream_len(&mut images)?, stream_len(&mut labels)?);
        let short = |needed, available| ImageError::Truncated { needed, available };
        if img_len < IDX_IMAGE_HEADER_BYTES {
            return Err(short(IDX_IMAGE_HEADER_BYTES, img_len));
        }
        if lab_len < IDX_LABEL_HEADER_BYTES {
            return Err(short(IDX_LABEL_HEADER_BYTES, lab_len));
        }
        let ih = read_header(&mut images, 16)?;
        check_idx_magic(be32(&ih), MNIST_IMAGES_MAGIC)?;
        let lh = read_header(&mut labels, 8)?;
        check_idx_magic(be32(&lh), MNIST_LABELS_MAGIC)?;
        let (n, rows, cols) = (
            u64::from(be32(&ih[4..])),
            u64::from(be32(&ih[8..])),
            u64::from(be32(&ih[12..])),
        );
        let n_labels = u64::from(be32(&lh[4..]));
        if n != n_labels {
            return Err(ImageError::CountMismatch {
                images: n,
                labels: n_labels,
            });
        }
        if n == 0 || rows == 0 || cols == 0 {
            return Err(ImageError::Empty);
        }
        let px = n
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(cols))
            .ok_or(ImageError::DimensionOverflow)?;
        check_exact(IDX_IMAGE_HEADER_BYTES, px, img_len)?;
        check_exact(IDX_LABEL_HEADER_BYTES, n_labels, lab_len)?;
        let charge = reserve(budget, px, n_labels)?;
        let (px_us, n_us) = (to_usize(px)?, to_usize(n)?);
        let mut pixels = buffer::<u8>(px_us)?;
        read_payload(&mut images, &mut pixels, px_us).map_err(|e| truncated(e, px, img_len))?;
        let mut label_bytes = buffer::<u8>(n_us)?;
        read_payload(&mut labels, &mut label_bytes, n_us).map_err(|e| truncated(e, n, lab_len))?;
        check_labels(&label_bytes)?;
        Ok(Self {
            pixels,
            labels: label_bytes,
            len: n_us,
            channels: 1,
            height: to_usize(rows)?,
            width: to_usize(cols)?,
            _charge: charge,
        })
    }

    /// Read a CIFAR-10 binary file (3073-byte records, no header). A length
    /// that is not a whole number of records is refused, as is a payload
    /// over `budget` or a label `>= 10`.
    pub fn from_cifar10<R: Read + Seek>(mut file: R, budget: &Budget) -> Result<Self, ImageError> {
        let total = stream_len(&mut file)?;
        let record = CIFAR10_RECORD_BYTES as u64;
        if total % record != 0 {
            return Err(ImageError::TrailingBytes {
                extra: total % record,
            });
        }
        let n = total / record;
        if n == 0 {
            return Err(ImageError::Empty);
        }
        let image_bytes = (CIFAR10_RECORD_BYTES - 1) as u64;
        let px = n
            .checked_mul(image_bytes)
            .ok_or(ImageError::DimensionOverflow)?;
        let charge = reserve(budget, px, n)?;
        let (px_us, n_us) = (to_usize(px)?, to_usize(n)?);
        let mut pixels: Vec<u8> = buffer(px_us)?;
        let mut labels: Vec<u8> = buffer(n_us)?;
        let mut chunk = [0u8; CIFAR_CHUNK_RECORDS * CIFAR10_RECORD_BYTES];
        let mut left = n_us;
        while left > 0 {
            let take = left.min(CIFAR_CHUNK_RECORDS);
            file.read_exact(&mut chunk[..take * CIFAR10_RECORD_BYTES])
                .map_err(|e| truncated(io_error(e), total, total))?;
            let (records, _) =
                chunk[..take * CIFAR10_RECORD_BYTES].as_chunks::<CIFAR10_RECORD_BYTES>();
            for rec in records {
                labels.push(rec[0]);
                pixels.extend_from_slice(&rec[1..]);
            }
            left -= take;
        }
        check_labels(&labels)?;
        Ok(Self {
            pixels,
            labels,
            len: n_us,
            channels: CIFAR10_CHANNELS,
            height: CIFAR10_SIDE,
            width: CIFAR10_SIDE,
            _charge: charge,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// `channels * height * width`, the `u8` and `f32` element count of one image.
    pub fn image_len(&self) -> usize {
        self.channels * self.height * self.width
    }

    /// Image `i` as raw `u8`, CHW. Panics if `i >= len()`.
    pub fn image(&self, i: usize) -> &[u8] {
        let n = self.image_len();
        &self.pixels[i * n..(i + 1) * n]
    }

    /// Label of image `i`. Panics if `i >= len()`.
    pub fn label(&self, i: usize) -> u8 {
        self.labels[i]
    }
}

fn to_usize(v: u64) -> Result<usize, ImageError> {
    usize::try_from(v).map_err(|_| ImageError::DimensionOverflow)
}

/// The stream must be exactly `header + payload` bytes.
fn check_exact(header: u64, payload: u64, available: u64) -> Result<(), ImageError> {
    let needed = header
        .checked_add(payload)
        .ok_or(ImageError::DimensionOverflow)?;
    match needed.cmp(&available) {
        std::cmp::Ordering::Greater => Err(ImageError::Truncated { needed, available }),
        std::cmp::Ordering::Less => Err(ImageError::TrailingBytes {
            extra: available - needed,
        }),
        std::cmp::Ordering::Equal => Ok(()),
    }
}

fn truncated(e: ImageError, needed: u64, available: u64) -> ImageError {
    match e {
        ImageError::Truncated { .. } => ImageError::Truncated { needed, available },
        other => other,
    }
}

fn reserve(budget: &Budget, pixels: u64, labels: u64) -> Result<Reservation, ImageError> {
    let requested = pixels
        .checked_add(labels)
        .ok_or(ImageError::DimensionOverflow)?;
    budget
        .try_reserve(requested)
        .map_err(|_| ImageError::OverBudget {
            requested,
            cap: budget.cap_bytes(),
        })
}

fn check_labels(labels: &[u8]) -> Result<(), ImageError> {
    match labels.iter().position(|&l| l >= NUM_CLASSES) {
        Some(i) => Err(ImageError::LabelOutOfRange {
            index: i as u64,
            label: labels[i],
        }),
        None => Ok(()),
    }
}

/// Per-channel `(p / 255 - mean) / std`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normalization {
    mean: [f32; MAX_CHANNELS],
    std: [f32; MAX_CHANNELS],
    channels: usize,
}

impl Normalization {
    /// MNIST's usual single-channel constants.
    pub const MNIST: Self = Self {
        mean: [0.1307, 0.0, 0.0],
        std: [0.3081, 1.0, 1.0],
        channels: 1,
    };
    /// CIFAR-10's usual training-set mean and standard deviation.
    pub const CIFAR10: Self = Self {
        mean: [0.4914, 0.4822, 0.4465],
        std: [0.2470, 0.2435, 0.2616],
        channels: 3,
    };

    /// Refuses a length mismatch, more than [`MAX_CHANNELS`], and a
    /// non-finite mean or a non-finite or non-positive std.
    pub fn new(mean: &[f32], std: &[f32]) -> Result<Self, ImageError> {
        let c = mean.len();
        if c == 0 || c > MAX_CHANNELS || std.len() != c {
            return Err(ImageError::Config(format!(
                "normalization: {} means and {} stds for 1..={MAX_CHANNELS} channels",
                mean.len(),
                std.len()
            )));
        }
        if !mean.iter().all(|m| m.is_finite()) || !std.iter().all(|s| s.is_finite() && *s > 0.0) {
            return Err(ImageError::Config(
                "normalization: mean must be finite and std finite and positive".into(),
            ));
        }
        let mut n = Self {
            mean: [0.0; MAX_CHANNELS],
            std: [1.0; MAX_CHANNELS],
            channels: c,
        };
        n.mean[..c].copy_from_slice(mean);
        n.std[..c].copy_from_slice(std);
        Ok(n)
    }

    pub fn channels(&self) -> usize {
        self.channels
    }
}

/// Random crop and horizontal flip. Both draw from the counter RNG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Augment {
    /// Zero padding on every side before a random crop back to the original
    /// size; `0` disables the crop.
    pub crop_pad: usize,
    /// Mirror each image left-right with probability one half.
    pub flip: bool,
}

impl Augment {
    pub const NONE: Self = Self {
        crop_pad: 0,
        flip: false,
    };
    /// The CIFAR-10 recipe: pad 4, random crop, random flip.
    pub const CIFAR10: Self = Self {
        crop_pad: 4,
        flip: true,
    };

    fn is_off(&self) -> bool {
        self.crop_pad == 0 && !self.flip
    }
}

/// What one image's augmentation drew: the crop window's offset into the
/// padded image (`0..=2*pad`) and the flip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AugmentDraw {
    pub dy: usize,
    pub dx: usize,
    pub flip: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageSamplerConfig {
    pub batch: usize,
    pub seed: u64,
    /// `true` skips the last, short batch of an epoch; `false` returns it
    /// with its smaller count from [`ImageBatchSampler::next_batch`].
    pub drop_last: bool,
    pub normalization: Normalization,
    pub augment: Augment,
}

/// Multiply-shift reduction of a `u64` draw into `0..m`.
fn reduce(r: u64, m: u64) -> usize {
    ((u128::from(r) * u128::from(m)) >> 64) as usize
}

/// The augmentation of image `index` in `epoch` under `seed`. Three draws
/// are always consumed, so enabling one feature never moves the other's mask.
pub fn augment_draw(seed: u64, epoch: u64, index: u64, aug: Augment) -> AugmentDraw {
    let mut rng = counter_stream(seed, epoch, index);
    let span = 2 * aug.crop_pad as u64 + 1;
    let dy = reduce(rng.next_u64(), span);
    let dx = reduce(rng.next_u64(), span);
    let flip = aug.flip && rng.next_u64() >> 63 == 1;
    AugmentDraw { dy, dx, flip }
}

/// Seeded epoch batches of normalised `f32` NCHW images and `u32` labels.
///
/// The cursor is a [`DataCursor`]: `shard` is the epoch and `token_index` the
/// position, within that epoch's shuffled order, of the next image. The seed,
/// batch size and `drop_last` are configuration, not cursor state.
#[derive(Debug)]
pub struct ImageBatchSampler<'a> {
    set: &'a ImageSet,
    cfg: ImageSamplerConfig,
    lut: [[f32; 256]; MAX_CHANNELS],
    order: Vec<u32>,
    order_epoch: Option<u64>,
    epoch: u64,
    pos: usize,
}

impl<'a> ImageBatchSampler<'a> {
    pub fn new(set: &'a ImageSet, cfg: ImageSamplerConfig) -> Result<Self, ImageError> {
        Self::resume(set, cfg, DataCursor::default())
    }

    /// Continue from `cursor`. A cursor at the end of an epoch's usable
    /// images (past the last full batch under `drop_last`) moves to the next
    /// epoch. `cursor.token_index` must be below the set's length.
    pub fn resume(
        set: &'a ImageSet,
        cfg: ImageSamplerConfig,
        cursor: DataCursor,
    ) -> Result<Self, ImageError> {
        let bad = |d: String| Err(ImageError::Config(d));
        if cfg.batch == 0 {
            return bad("batch must be non-zero".into());
        }
        if cfg.normalization.channels != set.channels {
            return bad(format!(
                "normalization has {} channels, the images {}",
                cfg.normalization.channels, set.channels
            ));
        }
        if u32::try_from(set.len).is_err() {
            return bad(format!("{} images exceed the u32 index", set.len));
        }
        if cfg.drop_last && cfg.batch > set.len {
            return bad(format!(
                "drop_last with batch {} over {} images yields no batch",
                cfg.batch, set.len
            ));
        }
        if cursor.token_index >= set.len as u64 {
            return bad(format!(
                "cursor position {} is past the {} images of an epoch",
                cursor.token_index, set.len
            ));
        }
        cfg.batch
            .checked_mul(set.image_len())
            .ok_or_else(|| ImageError::Config("batch * image size overflows".into()))?;
        let mut lut = [[0.0f32; 256]; MAX_CHANNELS];
        for (c, table) in lut.iter_mut().enumerate().take(set.channels) {
            let (mean, std) = (cfg.normalization.mean[c], cfg.normalization.std[c]);
            for (p, v) in table.iter_mut().enumerate() {
                *v = (p as f32 / 255.0 - mean) / std;
            }
        }
        let order = buffer::<u32>(set.len)?;
        let mut s = Self {
            set,
            cfg,
            lut,
            order,
            order_epoch: None,
            epoch: cursor.shard,
            pos: cursor.token_index as usize,
        };
        s.normalize_cursor()?;
        Ok(s)
    }

    /// Images in one epoch.
    pub fn images_per_epoch(&self) -> usize {
        self.set.len
    }

    /// Where the next batch starts. Store it in the checkpoint.
    pub fn cursor(&self) -> DataCursor {
        DataCursor {
            shard: self.epoch,
            token_index: self.pos as u64,
        }
    }

    /// Floats one batch needs in the caller's `x`: `batch * C * H * W`.
    pub fn batch_floats(&self) -> usize {
        self.cfg.batch * self.set.image_len()
    }

    /// The augmentation image `index` gets in `epoch`.
    pub fn draw_for(&self, epoch: u64, index: usize) -> AugmentDraw {
        augment_draw(self.cfg.seed, epoch, index as u64, self.cfg.augment)
    }

    /// Dataset index of the `ordinal`-th image of `epoch`'s shuffle.
    pub fn index_at(&mut self, epoch: u64, ordinal: usize) -> Option<usize> {
        self.ensure_order(epoch);
        self.order.get(ordinal).map(|&i| i as usize)
    }

    /// Write the next batch into `x` (`[count, C, H, W]` `f32`, NCHW) and
    /// `y` (`[count]` `u32`) and return `count`. `count == batch` except for
    /// an epoch's last batch when `drop_last` is `false`. Entries past
    /// `count` are not touched. Allocates nothing. On error the cursor does
    /// not move.
    pub fn next_batch(&mut self, x: &mut [f32], y: &mut [u32]) -> Result<usize, ImageError> {
        if x.len() < self.batch_floats() || y.len() < self.cfg.batch {
            return Err(ImageError::Config(format!(
                "output buffers hold {} floats and {} labels, a batch needs {} and {}",
                x.len(),
                y.len(),
                self.batch_floats(),
                self.cfg.batch
            )));
        }
        let epoch = self.epoch;
        self.ensure_order(epoch);
        let count = self.cfg.batch.min(self.set.len - self.pos);
        let n = self.set.image_len();
        for k in 0..count {
            let idx = self.order[self.pos + k] as usize;
            self.fill_image(epoch, idx, &mut x[k * n..(k + 1) * n]);
            y[k] = u32::from(self.set.labels[idx]);
        }
        self.pos += count;
        self.normalize_cursor()?;
        Ok(count)
    }

    /// Move a finished epoch's cursor to the start of the next one.
    fn normalize_cursor(&mut self) -> Result<(), ImageError> {
        let left = self.set.len - self.pos;
        if left == 0 || (self.cfg.drop_last && left < self.cfg.batch) {
            self.epoch = self
                .epoch
                .checked_add(1)
                .ok_or_else(|| ImageError::Config("epoch counter overflows".into()))?;
            self.pos = 0;
        }
        Ok(())
    }

    /// Make `order` the shuffle of `epoch`: identity, then Fisher-Yates
    /// driven by the `(seed, epoch)` stream. Allocation-free after `resume`.
    fn ensure_order(&mut self, epoch: u64) {
        if self.order_epoch == Some(epoch) {
            return;
        }
        self.order.clear();
        self.order.extend(0..self.set.len as u32);
        let mut rng = counter_stream(self.cfg.seed, epoch, SHUFFLE_STREAM);
        for i in (1..self.order.len()).rev() {
            let j = reduce(rng.next_u64(), i as u64 + 1);
            self.order.swap(i, j);
        }
        self.order_epoch = Some(epoch);
    }

    fn fill_image(&self, epoch: u64, idx: usize, out: &mut [f32]) {
        let set = self.set;
        let src = set.image(idx);
        let (h, w) = (set.height, set.width);
        if self.cfg.augment.is_off() {
            for c in 0..set.channels {
                let table = &self.lut[c];
                let plane = c * h * w..(c + 1) * h * w;
                for (o, &p) in out[plane.clone()].iter_mut().zip(&src[plane]) {
                    *o = table[usize::from(p)];
                }
            }
            return;
        }
        let pad = self.cfg.augment.crop_pad as isize;
        let d = augment_draw(self.cfg.seed, epoch, idx as u64, self.cfg.augment);
        // The cropped image's pixel (y, x) is padded-image pixel (y + dy, x + dx);
        // padded coordinates are source coordinates shifted by `pad`.
        for c in 0..set.channels {
            let table = &self.lut[c];
            let zero = table[0];
            let plane = c * h * w;
            for y in 0..h {
                let sy = y as isize + d.dy as isize - pad;
                let row = &mut out[plane + y * w..plane + (y + 1) * w];
                if sy < 0 || sy >= h as isize {
                    row.fill(zero);
                    continue;
                }
                let srow = &src[plane + sy as usize * w..plane + (sy as usize + 1) * w];
                for (x, o) in row.iter_mut().enumerate() {
                    let fx = if d.flip { w - 1 - x } else { x };
                    let sx = fx as isize + d.dx as isize - pad;
                    *o = if sx < 0 || sx >= w as isize {
                        zero
                    } else {
                        table[usize::from(srow[sx as usize])]
                    };
                }
            }
        }
    }
}
