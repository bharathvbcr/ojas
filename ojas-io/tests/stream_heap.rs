//! Peak heap while streaming a large tensor through `SafeTensorsWriter`, and
//! while decoding one through the typed readers.
//!
//! Every test takes `SERIAL` first, so no other test allocates while a peak is
//! measured. The counting allocator forwards to `System`; it is test-only,
//! and the library itself stays `forbid(unsafe_code)`.

use ojas_io::{SafeTensors, SafeTensorsWriter, StDtype, TensorSpec};
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

static SERIAL: Mutex<()> = Mutex::new(());

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call forwards to `System` with the caller's arguments, so the
// `GlobalAlloc` contract is `System`'s. The counters are plain atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Discards bytes, keeping a count.
struct Sink(u64);

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn streaming_a_gibibyte_keeps_peak_heap_under_a_mebibyte() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    const TOTAL: u64 = 1 << 30;
    const CHUNK: usize = 64 << 10;
    let shape = [TOTAL / 2];
    let specs = [TensorSpec {
        name: "embed",
        dtype: StDtype::BF16,
        shape: &shape,
    }];
    let chunk = vec![0x5Au8; CHUNK];
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let mut w = SafeTensorsWriter::new(Sink(0), &specs, &[("ojas.step", "7")]).unwrap();
    let mut sent = 0u64;
    while sent < TOTAL {
        w.write("embed", &chunk).unwrap();
        sent += CHUNK as u64;
    }
    let file_len = w.file_len();
    let sink = w.finish().unwrap();
    let peak = PEAK.load(Ordering::Relaxed) - base;
    assert_eq!(sink.0, file_len);
    assert!(sink.0 > TOTAL);
    assert!(
        peak < 1 << 20,
        "peak heap above the starting point was {peak} bytes for a {TOTAL}-byte tensor"
    );
}

/// A safetensors file of one tensor `t` of `dtype`, `bytes` long, sparse:
/// the data region is a hole, so the file costs no disk.
fn sparse_file(tag: &str, dtype: &str, shape_len: u64, bytes: u64) -> std::path::PathBuf {
    let header = format!(
        r#"{{"t":{{"dtype":"{dtype}","shape":[{shape_len}],"data_offsets":[0,{bytes}]}}}}"#
    );
    let mut h = header.into_bytes();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let path =
        std::env::temp_dir().join(format!("ojas-io-{tag}-{}.safetensors", std::process::id()));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&h).unwrap();
    f.set_len(8 + h.len() as u64 + bytes).unwrap();
    path
}

#[test]
fn a_64_mebibyte_bf16_read_holds_the_output_and_one_chunk() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    const BYTES: u64 = 64 << 20;
    let path = sparse_file("bf16-64m", "BF16", BYTES / 2, BYTES);
    let st = SafeTensors::open(&path).unwrap();
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let (_, bits) = st.read_bf16_bits("t").unwrap();
    let extra = PEAK.load(Ordering::Relaxed) - base - bits.len() * 2;
    assert_eq!(bits.len() as u64, BYTES / 2);
    assert!(bits.iter().all(|&b| b == 0));
    drop(bits);

    // Widened, the output is twice the stored bytes; still one output + a chunk.
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
    let base = LIVE.load(Ordering::Relaxed);
    let (_, wide) = st.read_f32_widened("t").unwrap();
    let extra_wide = PEAK.load(Ordering::Relaxed) - base - wide.len() * 4;
    drop(wide);
    std::fs::remove_file(&path).unwrap();
    assert!(
        extra <= 1 << 20,
        "{extra} bytes of heap beyond the bf16 output"
    );
    assert!(
        extra_wide <= 1 << 20,
        "{extra_wide} bytes of heap beyond the f32 output"
    );
}

/// A tensor declared far past what the host can allocate is an error, found
/// from the header without reading the hole; it neither aborts nor waits.
#[test]
fn an_oversized_declared_tensor_on_a_sparse_file_is_an_error_quickly() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // The reader refuses by failing to reserve, so the size must be one the host's allocator
    // refuses up front. Linux (overcommit heuristic) refuses 8 TiB, and ext4 cannot hold a
    // sparse file over 16 TiB (EFBIG). macOS reserves lazily whatever fits the 47-bit address
    // space, so it needs 128 TiB, which APFS holds.
    const BYTES: u64 = if cfg!(target_os = "macos") {
        1 << 47
    } else {
        1 << 43
    };
    let path = sparse_file("oversized", "BF16", BYTES / 2, BYTES);
    let st = SafeTensors::open(&path).unwrap();
    let start = std::time::Instant::now();
    let results = [
        st.read_bf16_bits("t").map(|_| ()),
        st.read_f32_widened("t").map(|_| ()),
        st.read_bytes("t").map(|_| ()),
    ];
    let took = start.elapsed();
    std::fs::remove_file(&path).unwrap();
    for r in results {
        let msg = r.unwrap_err().to_string();
        assert!(
            msg.contains("refused") || msg.contains("too large"),
            "{msg}"
        );
    }
    assert!(took < std::time::Duration::from_secs(1), "took {took:?}");
}
