# Image data formats

`ojas-data` reads two image datasets with std-only code. No decoder crate was
added (decision E4); PNG and JPEG are not supported. Code:
`ojas-data/src/image.rs`. Tests: `ojas-data/tests/image.rs` and
`ojas-metal/tests/mnist_step.rs`.

## Supported formats

| Format | Files | Layout |
| :--- | :--- | :--- |
| MNIST idx | image file, label file | Big-endian. Images: magic `0x00000803` (two zero bytes, dtype `0x08` = `u8`, 3 dimensions), then `count`, `rows`, `cols` as `u32`, then `count*rows*cols` bytes. Labels: magic `0x00000801`, `count`, then `count` bytes. |
| CIFAR-10 binary | one `data_batch_*.bin` or `test_batch.bin` | No header. Records of 3073 bytes: one label byte, then 3 x 32 x 32 `u8` in CHW order. The count is `file_len / 3073`; a remainder is an error. |

Labels are `0..10` in both. `ImageSet::from_mnist` and `ImageSet::from_cifar10`
take any `Read + Seek` (a `File` or a `Cursor`) and a `ojas_core::Budget`.

## What is refused

Each case is a typed `ImageError` and none of them panics:

- wrong magic (`BadMagic`) or a dtype byte other than `0x08` (`BadDtype`);
- a dimension product that overflows (`DimensionOverflow`, checked arithmetic);
- a stream shorter than its header declares (`Truncated`) or longer
  (`TrailingBytes`);
- image and label counts that differ (`CountMismatch`), a zero count or a
  zero-sized image (`Empty`);
- a label `>= 10` (`LabelOutOfRange`);
- a payload over the `Budget` (`OverBudget`). Sizes are checked against the
  stream length and then the budget before the payload is read or allocated;
  the buffers are `try_reserve`d and read in bounded chunks. The bytes stay
  charged to the budget until the `ImageSet` drops.

`ImageError` converts into `DataError` for callers that use that type.

## Host storage and layout (decision E3)

Pixels stay `u8` on the host (a 128-image CIFAR batch is 384 KiB as `u8`,
1.5 MiB as `f32`). `DType` has no `U8`: nothing `u8` is built as a tensor
type. `ImageBatchSampler` widens to `f32` per batch, into a caller-owned
buffer, in NCHW order (`[count, C, H, W]`), with labels as `u32`. A device-side
cast-normalise from a `U8` upload is deferred until the vision benchmark
baseline (G-BENCH) measures what the `f32` upload costs.

## Normalisation

Per channel, `(p / 255 - mean) / std`, computed in `f32` through a 256-entry
table per channel, so the batch with augmentation off is bit-identical to the
raw decode normalised that way. `Normalization::MNIST` is mean 0.1307, std
0.3081; `Normalization::CIFAR10` is mean (0.4914, 0.4822, 0.4465) and std
(0.2470, 0.2435, 0.2616). `Normalization::new` takes any 1 to 3 channels and
refuses non-finite values and a non-positive std.

## Sampling, resume and augmentation determinism

- Each epoch visits every image once, in a Fisher-Yates order driven by the
  counter RNG stream `(seed, epoch)`.
- A batch never straddles an epoch. `next_batch` returns the number of images
  written. With `drop_last: false` the last batch of an epoch is short (its
  count is below `batch`, and entries past it are untouched); with
  `drop_last: true` that tail is skipped and the cursor moves to the next
  epoch.
- The cursor is a `DataCursor`: `shard` is the epoch, `token_index` the
  position in that epoch's order. Resuming from it is bit-identical, across an
  epoch boundary too. Seed, batch size and `drop_last` are configuration.
- Augmentation is `Augment { crop_pad, flip }` (`Augment::CIFAR10` is pad 4
  plus flip). Image `i` in epoch `e` draws from the stream `(seed, e, i)`: the
  crop offset `dy`, `dx` in `0..=2*pad`, then the flip. Three draws are always
  consumed, so toggling one feature does not move the other's mask. Because the
  stream is addressed by the image's dataset index, not its position in a batch,
  the same seed gives the same augmentation whatever the batch size or resume
  point. Padding is zero in pixel space, so a padded pixel normalises to
  `(0 - mean) / std`. The flip applies to the cropped image.
- After construction `next_batch` allocates nothing (counting-allocator test,
  128 CIFAR images).
