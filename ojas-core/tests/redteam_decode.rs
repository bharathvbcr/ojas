//! Red-team gate for `Tensor::to_f32_vec` / `Tensor::to_u32_vec`.
//!
//! Bytes are injected with `Scratch<u8>` + `Tensor::from_scratch`, so no
//! pattern ever passes through an `f32` value on the way in; the reference is
//! `u32::from_ne_bytes` over the view's window. Covers every exponent and
//! sign, NaN payloads (quiet and signalling), every 16-byte phase of the
//! window start, lengths 0..=67 around any vector tail, multi-dimensional and
//! rank-0 views, sentinel bytes after the window, a > 2^20 element tensor,
//! and the error cases (non-contiguous, wrong dtype, device placement).
//!
//! The long random sweep is `#[ignore]`d; run it with
//! `cargo test -p ojas-core --release --test redteam_decode -- --ignored`.

use std::any::Any;
use std::sync::Arc;

use ojas_core::{BackendId, Budget, DType, DeviceBuffer, OjasError, Scratch, Tensor};

/// splitmix64, so the sweep needs no dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn budget() -> Budget {
    Budget::new(u64::MAX)
}

/// A host tensor of `dtype` whose storage is exactly `bytes`, shape `[len/4]`.
fn raw(bytes: &[u8], dtype: DType) -> Tensor {
    let b = budget();
    let mut s = Scratch::<u8>::try_alloc(bytes.len(), &b).unwrap();
    s.as_mut_slice().copy_from_slice(bytes);
    Tensor::from_scratch(s, &[bytes.len() / 4], dtype).unwrap()
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_ne_bytes()).collect()
}

/// Reference decode of `n` words starting at `byte_offset`.
fn reference(bytes: &[u8], byte_offset: usize, n: usize) -> Vec<u32> {
    bytes[byte_offset..byte_offset + 4 * n]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_ne_bytes(*c))
        .collect()
}

fn f32_bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn assert_words(got: &[u32], want: &[u32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{what}: first mismatch at {i}: got {:#010x} want {:#010x}",
            got[i], want[i]
        );
    }
}

/// Decode `words` through both dtypes, whole and as offset views.
fn check_all_views(words: &[u32], what: &str) {
    let bytes = words_to_bytes(words);
    let f = raw(&bytes, DType::F32);
    let u = raw(&bytes, DType::U32);
    assert_words(&f32_bits(&f), words, &format!("{what} f32 whole"));
    assert_words(
        &u.to_u32_vec().unwrap(),
        words,
        &format!("{what} u32 whole"),
    );
}

/// Every exponent (256) x both signs x mantissas that matter: zero, one, the
/// quiet bit alone, quiet bit + payload, all ones, and a few random ones.
fn special_patterns() -> Vec<u32> {
    let mut rng = SplitMix64(0xdec0de);
    let mut out = Vec::new();
    for sign in [0u32, 1] {
        for exp in 0u32..256 {
            let mut mantissas = vec![
                0u32, 1, 0x40_0000, 0x40_0001, 0x7f_ffff, 0x3f_ffff, 0x20_0000,
            ];
            for _ in 0..5 {
                mantissas.push(rng.next_u64() as u32 & 0x7f_ffff);
            }
            for m in mantissas {
                out.push(sign << 31 | exp << 23 | m);
            }
        }
    }
    out
}

#[test]
fn every_exponent_sign_and_nan_payload_decodes_bit_exact() {
    let words = special_patterns();
    assert_eq!(words.len(), 2 * 256 * 12);
    check_all_views(&words, "special patterns");
    // And one value per tensor, so a scalar path is exercised too.
    for &w in words.iter().step_by(7) {
        check_all_views(&[w], &format!("single {w:#010x}"));
    }
}

/// Window starts at every 4-byte offset 0..=60 (all 16-byte phases, past a
/// 64-byte line) and lengths 0..=67, with sentinel words before and after;
/// the decode returns exactly the window, never the sentinels.
#[test]
fn offset_windows_and_tail_lengths_decode_exactly_the_window() {
    let mut rng = SplitMix64(0x0ff5e7);
    for off_words in 0usize..=15 {
        for n in 0usize..=67 {
            let tail = 1 + rng.below(9);
            let mut words: Vec<u32> = (0..off_words + n + tail)
                .map(|_| rng.next_u64() as u32)
                .collect();
            // Sentinels: a signalling NaN before and after the window.
            if off_words > 0 {
                words[off_words - 1] = 0x7f80_0001;
            }
            words[off_words + n] = 0xff80_0001;
            let bytes = words_to_bytes(&words);
            let want = reference(&bytes, off_words * 4, n);
            for dtype in [DType::F32, DType::U32] {
                let base = raw(&bytes, dtype);
                let what = format!("offset {} bytes len {n} {dtype:?}", off_words * 4);
                // Absolute offset via view, relative via narrow.
                let v = base.view(&[n], &[1], off_words * 4).unwrap();
                let nw = base.narrow(off_words * 4, &[n], &[1]).unwrap();
                for view in [&v, &nw] {
                    assert_eq!(view.byte_offset(), off_words * 4);
                    let got = match dtype {
                        DType::F32 => f32_bits(view),
                        _ => view.to_u32_vec().unwrap(),
                    };
                    assert_words(&got, &want, &what);
                }
                // A narrow of a narrow adds offsets.
                if n >= 2 {
                    let inner = nw.narrow(4, &[n - 2], &[1]).unwrap();
                    let got = match dtype {
                        DType::F32 => f32_bits(&inner),
                        _ => inner.to_u32_vec().unwrap(),
                    };
                    assert_words(&got, &want[1..n - 1], &format!("{what} nested narrow"));
                }
            }
        }
    }
}

/// Rank-0 (one element), rank-2/3 contiguous views at an offset, an empty
/// view whose offset is the very end of storage, and zero-length shapes.
#[test]
fn shapes_rank0_multi_dim_and_empty_views() {
    let words: Vec<u32> = (0u32..64)
        .map(|i| i.wrapping_mul(0x9e37_79b9) ^ 0x7f80_0000)
        .collect();
    let bytes = words_to_bytes(&words);
    for dtype in [DType::F32, DType::U32] {
        let base = raw(&bytes, dtype);
        let dec = |t: &Tensor| match dtype {
            DType::F32 => f32_bits(t),
            _ => t.to_u32_vec().unwrap(),
        };
        let scalar = base.view(&[], &[], 12).unwrap();
        assert_words(&dec(&scalar), &words[3..4], "rank 0");
        let m = base.view(&[3, 5], &[5, 1], 8).unwrap();
        assert_words(&dec(&m), &words[2..17], "rank 2 at offset 8");
        let c = base.view(&[2, 3, 4], &[12, 4, 1], 4 * 33).unwrap();
        assert_words(&dec(&c), &words[33..57], "rank 3 at offset 132");
        let end = base.view(&[0], &[1], bytes.len()).unwrap();
        assert!(dec(&end).is_empty(), "empty view at end of storage");
        let zero = base.view(&[4, 0, 3], &[0, 3, 1], 0).unwrap();
        assert!(dec(&zero).is_empty(), "zero-extent shape");
    }
    let empty = raw(&[], DType::F32);
    assert!(empty.to_f32_vec().unwrap().is_empty());
}

/// > 2^20 elements with an odd tail: any blocked or parallel decode path.
#[test]
fn large_tensor_with_odd_tail_decodes_bit_exact() {
    let mut rng = SplitMix64(0x1a49e);
    let n = (1 << 20) + 3;
    let words: Vec<u32> = (0..n + 5).map(|_| rng.next_u64() as u32).collect();
    let bytes = words_to_bytes(&words);
    for dtype in [DType::F32, DType::U32] {
        let base = raw(&bytes, dtype);
        let v = base.view(&[n], &[1], 4).unwrap();
        let got = match dtype {
            DType::F32 => f32_bits(&v),
            _ => v.to_u32_vec().unwrap(),
        };
        assert_words(&got, &words[1..n + 1], &format!("large {dtype:?}"));
    }
}

/// Random byte patterns, random offsets and lengths.
fn random_sweep(seed: u64, rounds: usize) {
    let mut rng = SplitMix64(seed);
    for round in 0..rounds {
        let off = rng.below(33);
        let n = rng.below(5000);
        let tail = rng.below(5);
        let words: Vec<u32> = (0..off + n + tail).map(|_| rng.next_u64() as u32).collect();
        let bytes = words_to_bytes(&words);
        let want = reference(&bytes, off * 4, n);
        let f = raw(&bytes, DType::F32).view(&[n], &[1], off * 4).unwrap();
        let u = raw(&bytes, DType::U32).view(&[n], &[1], off * 4).unwrap();
        let what = format!("seed {seed:#x} round {round}: offset {off} len {n}");
        assert_words(&f32_bits(&f), &want, &what);
        assert_words(&u.to_u32_vec().unwrap(), &want, &what);
    }
}

#[test]
fn random_byte_patterns_decode_bit_exact() {
    random_sweep(0x0dec_0de5_eed0, 400);
}

#[test]
#[ignore = "slow: cargo test -p ojas-core --release --test redteam_decode -- --ignored"]
fn random_byte_patterns_long_sweep() {
    for s in 0..32u64 {
        random_sweep(
            0x0dec_0de5_eed0 ^ s.wrapping_mul(0x9e37_79b9_7f4a_7c15),
            2000,
        );
    }
}

/// The write side keeps bits too: `from_f32` / `from_u32` then decode is the
/// identity, signalling NaNs included.
#[test]
fn from_f32_and_from_u32_roundtrip_is_bit_identity() {
    let words = special_patterns();
    let floats: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();
    let b = budget();
    let f = Tensor::from_f32(&floats, &[words.len()], &b).unwrap();
    assert_words(&f32_bits(&f), &words, "from_f32 roundtrip");
    let u = Tensor::from_u32(&words, &[words.len()], &b).unwrap();
    assert_words(&u.to_u32_vec().unwrap(), &words, "from_u32 roundtrip");
}

/// Strided or padded views are `Shape`; the other dtype is `Dtype`.
#[test]
fn noncontiguous_and_wrong_dtype_are_errors() {
    let words: Vec<u32> = (0..64).collect();
    let bytes = words_to_bytes(&words);
    for dtype in [DType::F32, DType::U32] {
        let base = raw(&bytes, dtype);
        for view in [
            base.view(&[4, 3], &[1, 4], 0).unwrap(),
            base.view(&[3, 4], &[0, 1], 0).unwrap(),
            base.view(&[3, 4], &[8, 1], 4).unwrap(),
            base.view(&[8], &[2], 0).unwrap(),
        ] {
            let r = match dtype {
                DType::F32 => view.to_f32_vec().map(drop),
                _ => view.to_u32_vec().map(drop),
            };
            assert!(
                matches!(r, Err(OjasError::Shape { .. })),
                "{dtype:?} {:?}: {r:?}",
                view.strides()
            );
        }
    }
    let f = raw(&bytes, DType::F32);
    assert!(matches!(
        f.to_u32_vec(),
        Err(OjasError::Dtype {
            expected: DType::U32,
            got: DType::F32,
            ..
        })
    ));
    let u = raw(&bytes, DType::U32);
    assert!(matches!(
        u.to_f32_vec(),
        Err(OjasError::Dtype {
            expected: DType::F32,
            got: DType::U32,
            ..
        })
    ));
}

#[derive(Debug)]
struct FakeDevice(usize);

impl DeviceBuffer for FakeDevice {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }
    fn byte_len(&self) -> usize {
        self.0
    }
    fn read_bytes(&self, _offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        Ok(vec![0xab; len])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A device tensor is `Placement` from both decoders and is not read back.
#[test]
fn device_tensor_is_placement_not_a_readback() {
    let b = budget();
    let before = b.device_readbacks();
    for dtype in [DType::F32, DType::U32] {
        let d = Tensor::from_device(Arc::new(FakeDevice(64)), &[16], dtype, &b).unwrap();
        let r = match dtype {
            DType::F32 => d.to_f32_vec().map(drop),
            _ => d.to_u32_vec().map(drop),
        };
        assert!(
            matches!(
                r,
                Err(OjasError::Placement {
                    found: Some(BackendId::Metal),
                    ..
                })
            ),
            "{dtype:?}: {r:?}"
        );
    }
    assert_eq!(b.device_readbacks(), before);
}
