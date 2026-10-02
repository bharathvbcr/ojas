//! `SafeTensorsWriter` and `SafeTensors::from_file`.
//!
//! The byte-identity tests compare against goldens captured from
//! `encode_safetensors` before it was rebuilt on the streaming writer, so they
//! do not reduce to the writer agreeing with itself.

use ojas_io::{
    encode_safetensors, write_safetensors, SafeTensors, SafeTensorsWriter, StDtype, TensorOut,
    TensorSpec,
};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// SplitMix64, the same stream as the crate's unit-test `Mix`.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

static SEQ: AtomicU64 = AtomicU64::new(0);

struct TmpPath(PathBuf);

impl Drop for TmpPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tmp(tag: &str) -> TmpPath {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    TmpPath(std::env::temp_dir().join(format!("ojas-io-stream-{}-{tag}-{n}", std::process::id())))
}

struct Owned {
    name: String,
    dtype: StDtype,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn specs(t: &[Owned]) -> Vec<TensorSpec<'_>> {
    t.iter()
        .map(|o| TensorSpec {
            name: &o.name,
            dtype: o.dtype,
            shape: &o.shape,
        })
        .collect()
}

fn items(t: &[Owned]) -> Vec<TensorOut<'_>> {
    t.iter()
        .map(|o| TensorOut {
            name: &o.name,
            dtype: o.dtype,
            shape: &o.shape,
            data: &o.data,
        })
        .collect()
}

type Meta<'a> = Vec<(&'a str, &'a str)>;

fn meta_refs(meta: &[(String, String)]) -> Meta<'_> {
    meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
}

/// Stream `t` through a writer into `sink`, each tensor in random pieces,
/// with empty pieces mixed in and zero-byte tensors sometimes not named at all.
fn stream<W: Write>(
    sink: W,
    t: &[Owned],
    meta: &[(&str, &str)],
    rng: &mut Mix,
) -> Result<W, ojas_io::IoError> {
    let mut w = SafeTensorsWriter::new(sink, &specs(t), meta)?;
    for o in t {
        if o.data.is_empty() && rng.below(2) == 0 {
            continue;
        }
        let mut at = 0;
        loop {
            if rng.below(5) == 0 {
                w.write(&o.name, &[])?;
            }
            if at == o.data.len() {
                break;
            }
            let n = 1 + rng.below((o.data.len() - at).min(9));
            w.write(&o.name, &o.data[at..at + n])?;
            at += n;
        }
    }
    w.finish()
}

fn golden_tensors() -> Vec<Owned> {
    let t = |name: &str, dtype, shape: &[u64], data: Vec<u8>| Owned {
        name: name.to_string(),
        dtype,
        shape: shape.to_vec(),
        data,
    };
    vec![
        t("w", StDtype::F32, &[2, 2], (0u8..16).collect()),
        t("b", StDtype::BF16, &[3], (16u8..22).collect()),
        t("h", StDtype::F16, &[], vec![0x00, 0x3c]),
        t("ids", StDtype::I64, &[0, 3], vec![]),
        t("t", StDtype::U16, &[1], vec![0x2a, 0x00]),
    ]
}

const GOLDEN_TENSORS: &str = r#""w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"BF16","shape":[3],"data_offsets":[16,22]},"h":{"dtype":"F16","shape":[],"data_offsets":[22,24]},"ids":{"dtype":"I64","shape":[0,3],"data_offsets":[24,24]},"t":{"dtype":"U16","shape":[1],"data_offsets":[24,26]}}"#;

fn golden_file(header: &str) -> Vec<u8> {
    let mut v = (header.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(header.as_bytes());
    v.extend((0u8..22).chain([0x00, 0x3c, 0x2a, 0x00]));
    v
}

#[test]
fn every_writer_path_matches_the_pre_streaming_goldens() {
    let t = golden_tensors();
    let no_meta = format!("{{{GOLDEN_TENSORS}       ");
    let with_meta = format!(
        "{{{}{GOLDEN_TENSORS}       ",
        r#""__metadata__":{"format":"pt","k\"\\\n\u0001":"é😀"},"#
    );
    assert_eq!((no_meta.len(), with_meta.len()), (288, 344));
    let cases: [(Meta, Vec<u8>); 2] = [
        (vec![], golden_file(&no_meta)),
        (
            vec![("format", "pt"), ("k\"\\\n\u{1}", "é😀")],
            golden_file(&with_meta),
        ),
    ];
    let mut rng = Mix(1);
    for (meta, want) in &cases {
        assert_eq!(&encode_safetensors(&items(&t), meta).unwrap(), want);
        for _ in 0..20 {
            assert_eq!(&stream(Vec::new(), &t, meta, &mut rng).unwrap(), want);
        }
        let path = tmp("golden");
        write_safetensors(&path.0, &items(&t), meta).unwrap();
        assert_eq!(&std::fs::read(&path.0).unwrap(), want);
        let w = SafeTensorsWriter::new(Vec::new(), &specs(&t), meta).unwrap();
        assert_eq!(w.file_len(), want.len() as u64);
        assert_eq!(w.remaining(), 26);
    }
    let empty = b"\x08\0\0\0\0\0\0\0{}      ".to_vec();
    assert_eq!(encode_safetensors(&[], &[]).unwrap(), empty);
    let w = SafeTensorsWriter::new(Vec::new(), &[], &[]).unwrap();
    assert_eq!(w.finish().unwrap(), empty);
}

const DTYPES: [StDtype; 5] = [
    StDtype::F32,
    StDtype::BF16,
    StDtype::F16,
    StDtype::I64,
    StDtype::U16,
];

const SOUP: [char; 8] = ['a', 'Z', '0', '.', '"', '\\', '\n', 'é'];

fn soup(rng: &mut Mix, min: usize) -> String {
    let n = min + rng.below(5);
    (0..n).map(|_| SOUP[rng.below(SOUP.len())]).collect()
}

/// The generator the corpus golden was captured with. Do not change it
/// without recapturing the hash from a known-good encoder.
fn corpus_case(rng: &mut Mix) -> (Vec<Owned>, Vec<(String, String)>) {
    let mut tensors = Vec::new();
    for i in 0..rng.below(6) {
        let name = format!("{}{i}", soup(rng, 0));
        let dtype = DTYPES[rng.below(DTYPES.len())];
        let shape: Vec<u64> = (0..rng.below(4))
            .map(|_| {
                if rng.below(6) == 0 {
                    0
                } else {
                    1 + rng.below(5) as u64
                }
            })
            .collect();
        let n = shape.iter().product::<u64>() * dtype.size();
        let data = (0..n).map(|_| rng.next() as u8).collect();
        tensors.push(Owned {
            name,
            dtype,
            shape,
            data,
        });
    }
    let meta = (0..rng.below(3))
        .map(|i| (format!("{}{i}", soup(rng, 0)), soup(rng, 0)))
        .collect();
    (tensors, meta)
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x100000001b3);
    }
}

/// FNV-1a of the 300-case corpus as the old one-shot encoder wrote it.
const CORPUS_FNV: u64 = 0x811731f6ee3f41cd;
const CORPUS_BYTES: usize = 79062;

#[test]
fn streamed_corpus_matches_the_pre_streaming_encoder_byte_for_byte() {
    let mut gen = Mix(0x601D);
    let mut chunks = Mix(0xC4);
    let (mut h_enc, mut h_stream) = (0xcbf29ce484222325u64, 0xcbf29ce484222325u64);
    let (mut n_enc, mut n_stream) = (0, 0);
    for i in 0..300 {
        let (t, meta) = corpus_case(&mut gen);
        let meta = meta_refs(&meta);
        let enc = encode_safetensors(&items(&t), &meta).unwrap();
        let streamed = stream(Vec::new(), &t, &meta, &mut chunks).unwrap();
        assert_eq!(streamed, enc, "case {i}");
        fnv(&mut h_enc, &enc);
        fnv(&mut h_stream, &streamed);
        n_enc += enc.len();
        n_stream += streamed.len();
    }
    assert_eq!((h_enc, n_enc), (CORPUS_FNV, CORPUS_BYTES));
    assert_eq!((h_stream, n_stream), (CORPUS_FNV, CORPUS_BYTES));
}

#[test]
fn random_shapes_stream_to_a_file_and_read_back_through_from_file() {
    let mut rng = Mix(0x57AE);
    let mut accepted = 0;
    for i in 0..200 {
        let mut t = Vec::new();
        for j in 0..rng.below(7) {
            let dtype = DTYPES[rng.below(DTYPES.len())];
            let shape: Vec<u64> = (0..rng.below(5))
                .map(|_| match rng.below(6) {
                    0 => 0,
                    1 => 17 + rng.below(40) as u64,
                    _ => 1 + rng.below(4) as u64,
                })
                .collect();
            let n = shape.iter().product::<u64>() * dtype.size();
            t.push(Owned {
                name: format!("p.{j}.{}", soup(&mut rng, 0)),
                dtype,
                shape,
                data: (0..n).map(|_| rng.next() as u8).collect(),
            });
        }
        let meta: Vec<(String, String)> = (0..rng.below(3))
            .map(|k| (format!("m{k}"), soup(&mut rng, 0)))
            .collect();
        let meta = meta_refs(&meta);
        let path = tmp("prop");
        let file = File::create(&path.0).unwrap();
        stream(file, &t, &meta, &mut rng).unwrap();
        let st = SafeTensors::from_file(File::open(&path.0).unwrap()).unwrap();
        assert_eq!(st.names().count(), t.len(), "case {i}");
        for o in &t {
            let info = st.info(&o.name).unwrap();
            assert_eq!((info.dtype, &info.shape), (o.dtype, &o.shape), "case {i}");
            assert_eq!(st.read_bytes(&o.name).unwrap(), o.data, "case {i}");
            let numel = o.shape.iter().product::<u64>() as usize;
            let got = match o.dtype {
                StDtype::F32 | StDtype::BF16 | StDtype::F16 => {
                    st.read_f32_widened(&o.name).unwrap().1.len()
                }
                StDtype::I64 => st.read_i64(&o.name).unwrap().1.len(),
                StDtype::U16 => st.read_u16(&o.name).unwrap().1.len(),
            };
            assert_eq!(got, numel, "case {i}");
        }
        for (k, v) in &meta {
            assert_eq!(st.metadata()[*k], *v);
        }
        assert_eq!(
            std::fs::read(&path.0).unwrap(),
            encode_safetensors(&items(&t), &meta).unwrap()
        );
        accepted += 1;
    }
    assert_eq!(accepted, 200);
}

fn spec<'a>(name: &'a str, dtype: StDtype, shape: &'a [u64]) -> TensorSpec<'a> {
    TensorSpec { name, dtype, shape }
}

fn expect_err<T>(r: Result<T, ojas_io::IoError>, needle: &str) {
    match r {
        Ok(_) => panic!("expected an error containing {needle:?}"),
        Err(e) => assert!(e.detail().contains(needle), "expected {needle:?} in {e}"),
    }
}

/// a: 4 bytes, e: 0 bytes, b: 2 bytes, z: 0 bytes.
fn abz() -> [TensorSpec<'static>; 4] {
    [
        spec("a", StDtype::U16, &[2]),
        spec("e", StDtype::F32, &[0]),
        spec("b", StDtype::U16, &[1]),
        spec("z", StDtype::I64, &[3, 0]),
    ]
}

#[test]
fn writer_refuses_order_count_and_overrun_mistakes_and_stays_poisoned() {
    let new = || SafeTensorsWriter::new(Vec::new(), &abz(), &[]).unwrap();

    let mut w = new();
    expect_err(w.write("b", &[0, 0]), "\"a\" has 0 of 4 bytes");
    expect_err(w.write("a", &[0; 4]), "poisoned");
    expect_err(w.finish(), "poisoned");

    let mut w = new();
    w.write("a", &[0; 4]).unwrap();
    expect_err(
        w.write("z", &[]),
        "out of order; \"b\" (2 bytes) comes first",
    );

    let mut w = new();
    w.write("a", &[1, 2]).unwrap();
    expect_err(w.write("b", &[0, 0]), "\"a\" has 2 of 4 bytes");

    let mut w = new();
    expect_err(
        w.write("a", &[0; 5]),
        "runs past the tensor's end; 4 of 4 remain",
    );

    let mut w = new();
    w.write("a", &[0; 4]).unwrap();
    expect_err(
        w.write("a", &[0]),
        "runs past the tensor's end; 0 of 4 remain",
    );

    let mut w = new();
    w.write("a", &[0; 4]).unwrap();
    w.write("b", &[0; 2]).unwrap();
    expect_err(w.write("a", &[]), "already written");

    let mut w = new();
    w.write("a", &[0; 4]).unwrap();
    w.write("b", &[0; 2]).unwrap();
    expect_err(w.write("b", &[0]), "past the end of the data");

    let mut w = new();
    expect_err(w.write("nope", &[]), "not a declared tensor");

    let mut w = new();
    w.write("a", &[0; 3]).unwrap();
    assert_eq!(w.remaining(), 3);
    expect_err(
        w.finish(),
        "finish with 3 of 6 data bytes unwritten; \"a\" is short",
    );

    let mut w = new();
    w.write("a", &[0; 4]).unwrap();
    expect_err(w.finish(), "\"b\" is short");

    // Zero-byte tensors may be named or skipped; trailing ones need nothing.
    let mut w = new();
    w.write("a", &[9; 4]).unwrap();
    w.write("e", &[]).unwrap();
    w.write("b", &[7, 7]).unwrap();
    let bytes = w.finish().unwrap();
    let st = SafeTensors::parse(&bytes).unwrap();
    assert_eq!(st.read_bytes("a").unwrap(), [9; 4]);
    assert_eq!(st.read_bytes("b").unwrap(), [7, 7]);
    assert!(st.read_bytes("z").unwrap().is_empty());
}

#[test]
fn writer_refuses_layouts_the_reader_would_refuse_before_writing_a_byte() {
    struct Refuse;
    impl Write for Refuse {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            panic!("a refused layout reached the sink")
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let one = spec("w", StDtype::F32, &[1]);
    let cases: Vec<(Vec<TensorSpec<'_>>, Meta, &str)> = vec![
        (vec![one, one], vec![], "duplicate key \"w\""),
        (vec![spec("", StDtype::F32, &[1])], vec![], "empty"),
        (
            vec![spec("__metadata__", StDtype::F32, &[1])],
            vec![],
            "reserved",
        ),
        (
            vec![one],
            vec![("k", "1"), ("k", "2")],
            "duplicate metadata key",
        ),
        (
            vec![spec("w", StDtype::F32, &[u64::MAX, 2])],
            vec![],
            "shape product overflows",
        ),
        (
            vec![spec("w", StDtype::I64, &[1 << 62])],
            vec![],
            "byte size overflows",
        ),
        (
            vec![
                spec("a", StDtype::U16, &[1 << 62]),
                spec("b", StDtype::U16, &[1 << 62]),
                spec("c", StDtype::U16, &[1 << 62]),
                spec("d", StDtype::U16, &[1 << 62]),
            ],
            vec![],
            "data offsets overflow",
        ),
    ];
    for (specs, meta, needle) in &cases {
        expect_err(SafeTensorsWriter::new(Refuse, specs, meta), needle);
    }
    // 100_000 one-sized dims encode to about 200 KB of JSON, which the
    // reader's node and byte budget refuses. Before the writer ran its header
    // through the reader, `encode_safetensors` produced this file and
    // `SafeTensors::parse` then refused it.
    let ones = vec![1u64; 100_000];
    expect_err(
        SafeTensorsWriter::new(Refuse, &[spec("w", StDtype::F32, &ones)], &[]),
        "the reader would refuse this header",
    );
    let data = [0u8; 4];
    let out = [TensorOut {
        name: "w",
        dtype: StDtype::F32,
        shape: &ones,
        data: &data,
    }];
    expect_err(encode_safetensors(&out, &[]), "the reader would refuse");
    let path = tmp("ones");
    expect_err(
        write_safetensors(&path.0, &out, &[]),
        "the reader would refuse",
    );
    assert!(!path.0.exists());
}

/// A sink that keeps only a count and a running hash.
struct Count {
    n: u64,
    h: u64,
}

impl Write for Count {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.n += buf.len() as u64;
        fnv(&mut self.h, buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_tensor_far_larger_than_any_buffer_streams_in_chunks() {
    const TOTAL: u64 = 256 << 20;
    const CHUNK: usize = 64 << 10;
    let shape = [TOTAL / 4];
    let specs = [
        spec("big", StDtype::F32, &shape),
        spec("tail", StDtype::U16, &[1]),
    ];
    let meta = [("k", "v")];
    let fresh = || Count {
        n: 0,
        h: 0xcbf29ce484222325,
    };
    // The expected stream: the header as a dropped writer leaves it, then the
    // same chunks fed straight to a second counter.
    let mut want = fresh();
    drop(SafeTensorsWriter::new(&mut want, &specs, &meta).unwrap());
    let header_len = want.n;
    let mut w = SafeTensorsWriter::new(fresh(), &specs, &meta).unwrap();
    let file_len = w.file_len();
    let mut chunk = vec![0u8; CHUNK];
    let mut sent = 0u64;
    while sent < TOTAL {
        for (i, b) in chunk.iter_mut().enumerate() {
            *b = (sent as usize + i).wrapping_mul(31) as u8;
        }
        w.write("big", &chunk).unwrap();
        want.write_all(&chunk).unwrap();
        sent += CHUNK as u64;
    }
    w.write("tail", &[1, 2]).unwrap();
    want.write_all(&[1, 2]).unwrap();
    assert_eq!(w.remaining(), 0);
    let sink = w.finish().unwrap();
    assert_eq!(file_len, header_len + TOTAL + 2);
    assert_eq!((sink.n, sink.h), (want.n, want.h));
    assert_eq!(sink.n, file_len);
    assert!(header_len < 256, "header is {header_len} bytes");
    assert_eq!(TOTAL / CHUNK as u64, 4096);
}

#[test]
fn a_multi_tebibyte_declaration_writes_its_header_without_allocating_for_it() {
    let shape = [1u64 << 40];
    let sink = Count {
        n: 0,
        h: 0xcbf29ce484222325,
    };
    let mut w = SafeTensorsWriter::new(sink, &[spec("huge", StDtype::F32, &shape)], &[]).unwrap();
    assert_eq!(w.remaining(), 1 << 42);
    w.write("huge", &[0u8; 4096]).unwrap();
    assert_eq!(w.remaining(), (1 << 42) - 4096);
    expect_err(w.finish(), "is short");
}

#[test]
fn from_file_ignores_the_cursor_and_matches_open() {
    let t = golden_tensors();
    let path = tmp("cursor");
    write_safetensors(&path.0, &items(&t), &[("k", "v")]).unwrap();
    let mut file = File::open(&path.0).unwrap();
    file.seek(SeekFrom::End(-3)).unwrap();
    let st = SafeTensors::from_file(file).unwrap();
    let opened = SafeTensors::open(&path.0).unwrap();
    for o in &t {
        assert_eq!(st.read_bytes(&o.name).unwrap(), o.data);
        assert_eq!(st.info(&o.name).unwrap(), opened.info(&o.name).unwrap());
    }
    assert_eq!(st.metadata()["k"], "v");

    let mut file = File::open(&path.0).unwrap();
    file.seek(SeekFrom::Start(5)).unwrap();
    let st = SafeTensors::from_file(file.try_clone().unwrap()).unwrap();
    assert_eq!(st.read_bytes("w").unwrap(), t[0].data);
    assert_eq!(
        file.stream_position().unwrap(),
        5,
        "from_file moved the cursor"
    );
}

#[test]
fn from_file_refuses_non_files_truncation_and_oversized_lengths() {
    let dir = tmp("dir");
    std::fs::create_dir(&dir.0).unwrap();
    expect_err(
        SafeTensors::from_file(File::open(&dir.0).unwrap()),
        "not a regular file",
    );
    expect_err(SafeTensors::open(&dir.0), "not a regular file");

    let good = encode_safetensors(&items(&golden_tensors()), &[]).unwrap();
    let path = tmp("trunc");
    for n in 0..good.len() {
        std::fs::write(&path.0, &good[..n]).unwrap();
        let r = SafeTensors::from_file(File::open(&path.0).unwrap());
        assert!(
            r.is_err(),
            "a {n}-byte prefix of a {}-byte file parsed",
            good.len()
        );
    }
    std::fs::write(&path.0, &good[..3]).unwrap();
    expect_err(
        SafeTensors::from_file(File::open(&path.0).unwrap()),
        "truncated file",
    );

    // A declared header length past the file, at the cap, and past the cap.
    for (n, needle) in [
        (good.len() as u64, "runs past the end"),
        (ojas_io::MAX_HEADER_BYTES, "runs past the end"),
        (ojas_io::MAX_HEADER_BYTES + 1, "outside 1..="),
        (u64::MAX, "outside 1..="),
    ] {
        let mut bad = good.clone();
        bad[..8].copy_from_slice(&n.to_le_bytes());
        std::fs::write(&path.0, &bad).unwrap();
        expect_err(SafeTensors::from_file(File::open(&path.0).unwrap()), needle);
    }

    // data_offsets that claim a terabyte the file does not have.
    let header =
        br#"{"w":{"dtype":"U16","shape":[549755813888],"data_offsets":[0,1099511627776]}}"#;
    let mut bad = (header.len() as u64).to_le_bytes().to_vec();
    bad.extend_from_slice(header);
    bad.extend_from_slice(&[0; 16]);
    std::fs::write(&path.0, &bad).unwrap();
    expect_err(
        SafeTensors::from_file(File::open(&path.0).unwrap()),
        "outside the 16-byte data buffer",
    );
}

/// `from_file` takes the file `open_nofollow` returns; the helper's own tests
/// check the flag value against the kernel.
#[cfg(unix)]
#[test]
fn from_file_takes_a_file_opened_with_o_nofollow() {
    let real = tmp("nofollow-real");
    write_safetensors(&real.0, &items(&golden_tensors()), &[]).unwrap();
    let link = tmp("nofollow-link");
    std::os::unix::fs::symlink(&real.0, &link.0).unwrap();
    let err = ojas_io::open_nofollow(&link.0).expect_err("followed a symlink");
    assert!(err.detail().contains("symbolic link"), "{err}");
    let st = SafeTensors::from_file(ojas_io::open_nofollow(&real.0).unwrap()).unwrap();
    assert_eq!(st.read_bytes("b").unwrap(), (16u8..22).collect::<Vec<_>>());
}
