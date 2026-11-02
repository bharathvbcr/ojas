//! Per-entry reads and the "read into caller slab" sink: bit-equality against
//! the whole-file decode, an unaligned caller slab, hostile headers and sinks,
//! and the `Budget` charge. Every check runs against both sources: bytes in
//! memory ([`SafeTensors::parse`]) and a file ([`SafeTensors::open`]).

use ojas_core::Budget;
use ojas_io::{
    encode_safetensors, OwnedSlab, SafeTensors, SlabEntry, SlabSink, StDtype, TensorOut,
    TensorSink, MAX_SLAB_ALIGN,
};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const SENT: u8 = 0xA5;

// ---- fixtures ------------------------------------------------------------

struct Fx {
    name: &'static str,
    dtype: StDtype,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn stream(seed: u64, n: usize) -> Vec<u8> {
    let mut z = seed;
    (0..n)
        .map(|_| {
            z = z
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (z >> 56) as u8
        })
        .collect()
}

/// Eight tensors of mixed width, odd byte counts and one empty one. Sorted by
/// name their lengths are 4000, 16, 60, 0, 66, 24, 5, 14.
fn fixtures() -> Vec<Fx> {
    let spec: [(&'static str, StDtype, &[u64]); 8] = [
        ("emb.weight", StDtype::F32, &[3, 5]),
        ("norm", StDtype::BF16, &[7]),
        ("ids", StDtype::I64, &[3]),
        ("mask", StDtype::U8, &[5]),
        ("empty", StDtype::F32, &[0, 4]),
        ("c", StDtype::C64, &[2]),
        ("big", StDtype::F32, &[1000]),
        ("h", StDtype::F16, &[33]),
    ];
    spec.iter()
        .enumerate()
        .map(|(i, (name, dtype, shape))| {
            let n = shape.iter().product::<u64>() * dtype.size();
            Fx {
                name,
                dtype: *dtype,
                shape: shape.to_vec(),
                data: stream(i as u64 + 1, n as usize),
            }
        })
        .collect()
}

fn encode(fx: &[Fx]) -> Vec<u8> {
    let items: Vec<TensorOut<'_>> = fx
        .iter()
        .map(|f| TensorOut {
            name: f.name,
            dtype: f.dtype,
            shape: &f.shape,
            data: &f.data,
        })
        .collect();
    encode_safetensors(&items, &[]).unwrap()
}

static SEQ: AtomicU64 = AtomicU64::new(0);

struct TmpFile(PathBuf);

impl Drop for TmpFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn tmp(tag: &str) -> TmpFile {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    TmpFile(std::env::temp_dir().join(format!("ojas-io-slab-{}-{tag}-{n}", std::process::id())))
}

/// Run `check` against the same bytes opened both ways.
fn both_sources(bytes: &[u8], check: impl Fn(&SafeTensors<'_>)) {
    check(&SafeTensors::parse(bytes).unwrap());
    let file = tmp("src");
    std::fs::write(&file.0, bytes).unwrap();
    check(&SafeTensors::open(&file.0).unwrap());
}

/// A valid file with its header text edited and the length prefix redone:
/// counts are patched in a real header rather than generated.
fn patch_header(bytes: &[u8], from: &str, to: &str) -> Vec<u8> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
    assert!(header.contains(from), "{from} not in {header}");
    let header = header.replacen(from, to, 1);
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&bytes[8 + n..]);
    out
}

fn one_f32() -> Vec<u8> {
    encode(&[Fx {
        name: "w",
        dtype: StDtype::F32,
        shape: vec![1],
        data: 1.5f32.to_le_bytes().to_vec(),
    }])
}

fn addr(s: &[u8]) -> usize {
    s.as_ptr().addr()
}

// ---- per-entry read --------------------------------------------------------

#[test]
fn per_entry_read_is_bit_equal_to_the_whole_file_decode() {
    let fx = fixtures();
    both_sources(&encode(&fx), |st| {
        for f in &fx {
            let mut dst = vec![SENT; f.data.len()];
            st.read_tensor_into(f.name, &mut dst).unwrap();
            assert_eq!(dst, f.data, "{}: against what was written", f.name);
            assert_eq!(
                dst,
                st.read_bytes(f.name).unwrap(),
                "{}: read_bytes",
                f.name
            );
            match f.dtype {
                StDtype::F32 => {
                    let bits: Vec<u8> = st
                        .read_f32(f.name)
                        .unwrap()
                        .1
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect();
                    assert_eq!(dst, bits, "{}: read_f32", f.name);
                }
                StDtype::I64 => {
                    let bits: Vec<u8> = st
                        .read_i64(f.name)
                        .unwrap()
                        .1
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    assert_eq!(dst, bits, "{}: read_i64", f.name);
                }
                StDtype::BF16 => {
                    let bits: Vec<u8> = st
                        .read_bf16_bits(f.name)
                        .unwrap()
                        .1
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    assert_eq!(dst, bits, "{}: read_bf16_bits", f.name);
                }
                _ => {}
            }
        }
    });
}

#[test]
fn per_entry_read_refuses_a_short_or_long_destination_and_an_unknown_name() {
    let fx = fixtures();
    both_sources(&encode(&fx), |st| {
        let mut short = vec![SENT; 59];
        assert!(st.read_tensor_into("emb.weight", &mut short).is_err());
        let mut long = vec![SENT; 61];
        assert!(st.read_tensor_into("emb.weight", &mut long).is_err());
        assert!(
            short.iter().chain(&long).all(|&b| b == SENT),
            "no byte written"
        );
        let mut one = [0u8; 1];
        assert!(st.read_tensor_into("empty", &mut one).is_err());
        st.read_tensor_into("empty", &mut []).unwrap();
        assert!(st.read_tensor_into("nope", &mut []).is_err());
    });
}

// ---- slab reads ------------------------------------------------------------

/// Expected offsets of the fixtures in name order from an aligned start:
/// big, c, emb.weight, empty, h, ids, mask, norm.
const AT_64: [usize; 8] = [0, 4032, 4096, 4156, 4160, 4288, 4352, 4416];

#[test]
fn slab_len_is_the_aligned_layout_and_checks_its_arguments() {
    both_sources(&encode(&fixtures()), |st| {
        let names: Vec<&str> = st.names().collect();
        assert_eq!(st.slab_len(&names, 1).unwrap(), 4185);
        assert_eq!(st.slab_len(&names, 64).unwrap(), 4430);
        assert_eq!(st.slab_len(&[], 64).unwrap(), 0);
        assert_eq!(
            st.slab_len(&["empty"], 4096).unwrap(),
            0,
            "empty needs no padding"
        );
        for bad in [0usize, 3, 6, MAX_SLAB_ALIGN << 1, usize::MAX] {
            assert!(st.slab_len(&names, bad).is_err(), "align {bad}");
        }
        assert!(st.slab_len(&["big"], MAX_SLAB_ALIGN).is_ok());
        assert!(st.slab_len(&["big", "big"], 64).is_err(), "repeated name");
        assert!(st.slab_len(&["nope"], 64).is_err(), "unknown name");
    });
}

fn check_slab(st: &SafeTensors<'_>, fx: &[Fx], names: &[&str], align: usize, skew: usize) {
    let need = st.slab_len(names, align).unwrap() as usize;
    let mut backing = vec![SENT; need + 3 * align + 16];
    let base = backing.as_ptr().align_offset(align);
    let start = base + skew;
    let slack = if skew == 0 { 0 } else { align - 1 };
    let (before, rest) = backing.split_at_mut(start);
    let (slab, after) = rest.split_at_mut(need + slack);
    let slab_addr = addr(slab);
    let entries = st.read_into_slab(names, align, slab).unwrap();

    assert_eq!(entries.len(), names.len());
    let mut covered = vec![false; slab.len()];
    for (e, &name) in entries.iter().zip(names) {
        assert_eq!(e.name, name);
        let f = fx.iter().find(|f| f.name == name).unwrap();
        assert_eq!(e.len, f.data.len());
        if e.len > 0 {
            assert_eq!(
                (slab_addr + e.offset) % align,
                0,
                "{name}: address aligned to {align}"
            );
        }
        let got = &slab[e.offset..e.offset + e.len];
        assert_eq!(got, &f.data[..], "{name}: bytes");
        assert_eq!(got, &st.read_bytes(name).unwrap()[..], "{name}: read_bytes");
        covered[e.offset..e.offset + e.len].fill(true);
    }
    for (i, (&b, &c)) in slab.iter().zip(&covered).enumerate() {
        assert!(c || b == SENT, "padding byte {i} was written");
    }
    assert!(
        before.iter().chain(after.iter()).all(|&b| b == SENT),
        "outside the slab"
    );
    if skew == 0 && align == 64 && names.first() == Some(&"big") && names.len() == 8 {
        let offs: Vec<usize> = entries.iter().map(|e| e.offset).collect();
        assert_eq!(offs, AT_64);
    }
}

#[test]
fn slab_read_is_bit_equal_at_every_alignment_and_skew() {
    let fx = fixtures();
    both_sources(&encode(&fx), |st| {
        let sorted: Vec<&str> = st.names().collect();
        let reversed: Vec<&str> = sorted.iter().rev().copied().collect();
        let subset = vec!["mask", "emb.weight", "empty"];
        for names in [&sorted, &reversed, &subset] {
            for align in [1usize, 2, 4, 64, 4096, 16384] {
                for skew in [0usize, 1, 3, 7, align / 2, align.saturating_sub(1)] {
                    check_slab(st, &fx, names, align, skew);
                }
            }
        }
    });
}

#[test]
fn an_unaligned_caller_slab_is_used_in_place_and_too_small_is_refused_untouched() {
    let fx = fixtures();
    both_sources(&encode(&fx), |st| {
        let names: Vec<&str> = st.names().collect();
        let need = st.slab_len(&names, 64).unwrap() as usize;
        let mut backing = vec![SENT; need + 256];
        let base = backing.as_ptr().align_offset(64);
        // Find a start that is not 64-aligned.
        let start = base + 5;
        let slab = &mut backing[start..start + need];
        assert_ne!(addr(slab) % 64, 0);
        // `need` bytes are enough only from an aligned start: refused, nothing written.
        let err = st.read_into_slab(&names, 64, slab).unwrap_err();
        assert!(err.detail().contains("needs"), "{err}");
        assert!(
            slab.iter().all(|&b| b == SENT),
            "a refused plan wrote nothing"
        );
        // `need + 63` is enough from any start.
        let slab = &mut backing[start..start + need + 63];
        let entries = st.read_into_slab(&names, 64, slab).unwrap();
        for e in &entries {
            if e.len > 0 {
                assert_eq!((addr(slab) + e.offset) % 64, 0);
            }
        }
        // One byte short of the aligned total, from an aligned start.
        let slab = &mut backing[base..base + need - 1];
        assert!(st.read_into_slab(&names, 64, slab).is_err());
        // Exactly the aligned total, from an aligned start.
        let slab = &mut backing[base..base + need];
        let entries = st.read_into_slab(&names, 64, slab).unwrap();
        let offs: Vec<usize> = entries.iter().map(|e| e.offset).collect();
        assert_eq!(offs, AT_64);
    });
}

#[test]
fn a_slab_sink_hands_out_slots_across_calls() {
    let fx = fixtures();
    both_sources(&encode(&fx), |st| {
        let mut backing = vec![SENT; 8192];
        let mut sink = SlabSink::new(&mut backing[3..]);
        st.read_into_sink(&["mask", "norm"], 16, &mut sink).unwrap();
        let used = sink.used();
        st.read_into_sink(&["ids"], 16, &mut sink).unwrap();
        assert!(sink.used() > used);
        let entries: Vec<SlabEntry> = sink.into_entries();
        assert_eq!(
            entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
            ["mask", "norm", "ids"]
        );
        for w in entries.windows(2) {
            assert!(
                w[0].offset + w[0].len <= w[1].offset,
                "slots do not overlap"
            );
        }
    });
}

// ---- hostile headers -------------------------------------------------------

#[test]
fn offsets_past_eof_overlap_and_a_truncated_file_are_refused_before_any_slab_read() {
    let valid = one_f32();
    // data_offsets end past the bytes the file has.
    let past = patch_header(&valid, "[0,4]", "[0,16]");
    // shape and offsets agree on 16 bytes; the file holds 4.
    let past = patch_header(&past, "\"shape\":[1]", "\"shape\":[4]");
    let err = SafeTensors::parse(&past).unwrap_err();
    assert!(err.detail().contains("outside the 4-byte data"), "{err}");
    let f = tmp("past");
    std::fs::write(&f.0, &past).unwrap();
    let err = SafeTensors::open(&f.0).unwrap_err();
    assert!(err.detail().contains("outside the 4-byte data"), "{err}");

    // A file cut short before it is opened: the header promises more data
    // than the file holds, so it is refused at open, not at the first read.
    let fx = fixtures();
    let bytes = encode(&fx);
    std::fs::write(&f.0, &bytes[..bytes.len() - 10]).unwrap();
    assert!(SafeTensors::open(&f.0).is_err());

    // Overlapping ranges: the second tensor starts inside the first.
    let two = encode(&[
        Fx {
            name: "a",
            dtype: StDtype::F32,
            shape: vec![2],
            data: vec![1; 8],
        },
        Fx {
            name: "b",
            dtype: StDtype::F32,
            shape: vec![1],
            data: vec![2; 4],
        },
    ]);
    let overlap = patch_header(&two, "[8,12]", "[4,8]");
    let err = SafeTensors::parse(&overlap).unwrap_err();
    assert!(err.detail().contains("gap or overlap"), "{err}");
    std::fs::write(&f.0, &overlap).unwrap();
    let err = SafeTensors::open(&f.0).unwrap_err();
    assert!(err.detail().contains("gap or overlap"), "{err}");
}

/// The header is checked against the file's length when it is opened, but the
/// file can shrink afterwards. Every read through that reader is then an error
/// naming the tensor, never a short or zero-filled success.
///
/// This used to assert as well that "a tensor before the cut still reads
/// exactly". The identity re-check retires that deliberately: the header's
/// offsets describe the file as it was at open, so once the file has changed
/// the reader cannot tell which offsets still mean what they did, and
/// answering for the ranges that happen to survive is a guess. It refuses
/// every read instead, which is the contract tessl's reader already had and
/// the one this reader now owns for both. What the test still pins is the part
/// that matters: an error that names the tensor, and a caller buffer left
/// untouched.
#[test]
fn a_file_cut_short_after_open_is_an_error_on_every_read() {
    let fx = fixtures();
    let bytes = encode(&fx);
    let f = tmp("cut");
    std::fs::write(&f.0, &bytes).unwrap();
    let st = SafeTensors::open(&f.0).unwrap();
    // Fixture order is the file order, so "h" (66 bytes) is last.
    let h_len = fx.last().unwrap().data.len();
    let cut = std::fs::OpenOptions::new().write(true).open(&f.0).unwrap();
    cut.set_len((bytes.len() - 1) as u64).unwrap();
    drop(cut);

    let mut h = vec![SENT; h_len];
    let err = st.read_tensor_into("h", &mut h).unwrap_err();
    assert!(err.detail().starts_with("h: "), "{err}");
    assert!(
        err.detail().contains("file changed since it was opened"),
        "{err}"
    );
    assert!(h.iter().all(|&b| b == SENT), "the destination was written");

    // The tensor wholly before the cut is refused too, and by the same check.
    let mut ids = vec![SENT; 24];
    let err = st.read_tensor_into("ids", &mut ids).unwrap_err();
    assert!(
        err.detail().contains("file changed since it was opened"),
        "{err}"
    );
    assert!(ids.iter().all(|&b| b == SENT), "the destination was written");

    let mut slab = vec![SENT; 1 << 13];
    let err = st.read_into_slab(&["ids", "h"], 8, &mut slab).unwrap_err();
    assert!(
        err.detail().contains("file changed since it was opened"),
        "{err}"
    );
}

#[test]
fn huge_counts_patched_into_a_header_are_refused() {
    let valid = one_f32();
    let cases = [
        // shape says 2^32 elements, offsets still say 4 bytes
        patch_header(&valid, "\"shape\":[1]", "\"shape\":[4294967296]"),
        // offsets and shape agree on 16 GiB, the file holds 4 bytes
        patch_header(
            &patch_header(&valid, "[0,4]", "[0,17179869184]"),
            "\"shape\":[1]",
            "\"shape\":[4294967296]",
        ),
        // shape product overflows u64
        patch_header(
            &valid,
            "\"shape\":[1]",
            "\"shape\":[18446744073709551615,2]",
        ),
        // offsets at the top of u64
        patch_header(
            &valid,
            "[0,4]",
            "[18446744073709551611,18446744073709551615]",
        ),
        // header length prefix patched to the maximum
        {
            let mut v = valid.clone();
            v[..8].copy_from_slice(&u64::MAX.to_le_bytes());
            v
        },
    ];
    for (i, bytes) in cases.iter().enumerate() {
        assert!(SafeTensors::parse(bytes).is_err(), "case {i}");
        let f = tmp("huge");
        std::fs::write(&f.0, bytes).unwrap();
        assert!(SafeTensors::open(&f.0).is_err(), "case {i} via file");
    }
}

/// A sparse file whose header honestly declares a 4 GiB tensor: it opens (the
/// header is all that is read), the plan is 4 GiB, and neither a `Budget` nor
/// a small slab accepts it, before any byte of the tensor is read.
#[test]
fn a_four_gib_tensor_is_refused_by_the_budget_and_by_a_small_slab() {
    let header = br#"{"w":{"dtype":"F32","shape":[1073741824],"data_offsets":[0,4294967296]}}"#;
    let mut header = header.to_vec();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let total = 8 + header.len() as u64 + (1u64 << 32);
    let f = tmp("sparse");
    let mut file = File::create(&f.0).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.set_len(total).unwrap();
    drop(file);

    let st = SafeTensors::open(&f.0).unwrap();
    assert_eq!(st.slab_len(&["w"], 64).unwrap(), 1 << 32);

    let budget = Budget::new(1 << 20);
    let err = OwnedSlab::new(&budget, 1 << 32, 64).unwrap_err();
    assert!(err.detail().contains("capacity exceeded"), "{err}");
    assert_eq!((budget.live_bytes().unwrap(), budget.peak_bytes()), (0, 0));

    let mut small = vec![SENT; 4096];
    let err = st.read_into_slab(&["w"], 64, &mut small).unwrap_err();
    assert!(err.detail().contains("needs"), "{err}");
    assert!(small.iter().all(|&b| b == SENT));
}

// ---- hostile arguments and sinks -------------------------------------------

#[test]
fn bad_alignments_repeated_and_unknown_names_are_refused_before_a_read() {
    both_sources(&encode(&fixtures()), |st| {
        let mut slab = vec![SENT; 1 << 16];
        for bad in [0usize, 3, 100, MAX_SLAB_ALIGN << 1] {
            assert!(
                st.read_into_slab(&["mask"], bad, &mut slab).is_err(),
                "align {bad}"
            );
        }
        assert!(st.read_into_slab(&["mask", "mask"], 8, &mut slab).is_err());
        assert!(st.read_into_slab(&["mask", "nope"], 8, &mut slab).is_err());
        assert!(slab.iter().all(|&b| b == SENT), "nothing was written");
        assert!(SlabSink::new(&mut slab).reserve(1, 3).is_err());
    });
}

enum Mode {
    Good,
    ShortSlot,
    LongSlot,
    MisalignedSlot,
    RefuseReserve,
}

struct Hostile {
    mode: Mode,
    buf: Vec<u8>,
    slots: usize,
}

impl TensorSink for Hostile {
    fn reserve(&mut self, _total: u64, _align: usize) -> Result<(), ojas_io::IoError> {
        match self.mode {
            Mode::RefuseReserve => Err(ojas_io::IoError::new("no room")),
            _ => Ok(()),
        }
    }

    fn slot(
        &mut self,
        _name: &str,
        len: usize,
        align: usize,
    ) -> Result<&mut [u8], ojas_io::IoError> {
        self.slots += 1;
        let base = self.buf.as_ptr().align_offset(align);
        let (at, n) = match self.mode {
            Mode::ShortSlot => (base, len - 1),
            Mode::LongSlot => (base, len + 1),
            Mode::MisalignedSlot => (base + 1, len),
            _ => (base, len),
        };
        Ok(&mut self.buf[at..at + n])
    }
}

#[test]
fn a_hostile_sink_cannot_make_the_reader_write_the_wrong_place() {
    both_sources(&encode(&fixtures()), |st| {
        for (mode, wants_slot, needle) in [
            (Mode::ShortSlot, true, "returned"),
            (Mode::LongSlot, true, "returned"),
            (Mode::MisalignedSlot, true, "not aligned"),
            (Mode::RefuseReserve, false, "no room"),
        ] {
            let mut sink = Hostile {
                mode,
                buf: vec![SENT; 4096],
                slots: 0,
            };
            let err = st
                .read_into_sink(&["emb.weight"], 64, &mut sink)
                .unwrap_err();
            assert!(err.detail().contains(needle), "{needle}: {err}");
            assert_eq!(sink.slots > 0, wants_slot);
            assert!(
                sink.buf.iter().all(|&b| b == SENT),
                "{needle}: nothing was read into it"
            );
        }
        let mut good = Hostile {
            mode: Mode::Good,
            buf: vec![SENT; 4096],
            slots: 0,
        };
        st.read_into_sink(&["emb.weight"], 64, &mut good).unwrap();
        assert_eq!(good.slots, 1);
    });
}

// ---- Budget ----------------------------------------------------------------

#[test]
fn an_owned_slab_charges_before_it_allocates_and_releases_on_drop() {
    let budget = Budget::new(1063);
    let slab = OwnedSlab::new(&budget, 1000, 64).unwrap();
    assert_eq!(budget.live_bytes().unwrap(), 1000 + 63);
    assert_eq!(slab.len(), 1000);
    assert_eq!(addr(slab.as_slice()) % 64, 0);
    assert!(slab.as_slice().iter().all(|&b| b == 0));
    assert!(OwnedSlab::new(&budget, 1, 1).is_err(), "the cap is spent");
    drop(slab);
    assert_eq!(budget.live_bytes().unwrap(), 0);

    let tight = Budget::new(1062);
    let err = OwnedSlab::new(&tight, 1000, 64).unwrap_err();
    assert!(err.detail().contains("capacity exceeded"), "{err}");
    assert_eq!((tight.live_bytes().unwrap(), tight.peak_bytes()), (0, 0));
    assert!(OwnedSlab::new(&tight, 0, 1).unwrap().is_empty());
    assert!(
        OwnedSlab::new(&tight, usize::MAX, 64).is_err(),
        "size overflow"
    );
    assert!(OwnedSlab::new(&tight, 8, 3).is_err(), "bad alignment");
}

#[test]
fn an_owned_slab_loads_the_fixtures_bit_equal_and_gives_the_charge_back() {
    let fx = fixtures();
    let bytes = encode(&fx);
    both_sources(&bytes, |st| {
        let budget = Budget::new(1 << 20);
        let names: Vec<&str> = st.names().collect();
        let need = st.slab_len(&names, 64).unwrap() as usize;
        let mut slab = OwnedSlab::new(&budget, need, 64).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), need as u64 + 63);
        let entries = st.read_into_slab(&names, 64, slab.as_mut_slice()).unwrap();
        let offs: Vec<usize> = entries.iter().map(|e| e.offset).collect();
        assert_eq!(
            offs, AT_64,
            "an owned slab starts aligned, so the layout is the aligned one"
        );
        for e in &entries {
            let f = fx.iter().find(|f| f.name == e.name).unwrap();
            assert_eq!(
                &slab.as_slice()[e.offset..e.offset + e.len],
                &f.data[..],
                "{}",
                e.name
            );
        }
        drop(slab);
        assert_eq!(budget.live_bytes().unwrap(), 0);
        assert_eq!(budget.peak_bytes(), need as u64 + 63);
    });
}
