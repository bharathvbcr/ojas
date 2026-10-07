//! One static library for the Go process (R14).
//!
//! `ojas-capi` is an rlib. This crate is the only `staticlib`, and its
//! archive is `libgusset.a` because cgo passes `-lgusset`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

// Counting is installed so `gusset.Stats` and `AdviseMemoryLimit` can see Rust
// heap bytes. It forwards each allocation to `System` and returns a null
// pointer when that allocation fails. Infallible Rust allocations (`vec!`,
// `format!`, channels) still abort the process when they fail: nothing here
// installs an allocation-error hook. So every allocation on an engine path
// that scales with a model, a batch or a caller's input is made fallibly
// (`try_reserve`), and its failure comes back to Go as `E_CAPACITY`.
//
// `Ceiling` sits in front and enforces `ojas_set_heap_ceiling`, which is how
// a host (and `TestHeapCeilingIsACleanError` in go/) makes an allocation
// fail on purpose and checks that it was a fallible one.
#[global_allocator]
static GLOBAL: Ceiling<gusset::Counting<System>> = Ceiling {
    inner: gusset::Counting::new(System),
};

/// The Rust heap ceiling in bytes, compared with gusset's live byte count.
/// [`NO_HEAP_CEILING`] is none, the default.
static HEAP_CEILING: AtomicU64 = AtomicU64::new(NO_HEAP_CEILING);

/// [`ojas_set_heap_ceiling`]'s "no ceiling".
pub const NO_HEAP_CEILING: u64 = u64::MAX;

/// Requests smaller than this are never refused by the ceiling. The engine's
/// infallible allocations (error text, headers, option records) are all far
/// below it, so a ceiling never turns one of them into an abort. A request at
/// or above it is model-sized and must already be fallible.
pub const HEAP_CEILING_MIN_REQUEST: usize = 1 << 20;

/// A `GlobalAlloc` that returns null for a request of at least
/// [`HEAP_CEILING_MIN_REQUEST`] bytes when the live Rust heap plus the bytes
/// it adds would pass [`HEAP_CEILING`]. Everything else goes to `inner`.
struct Ceiling<A> {
    inner: A,
}

/// `request` bytes asked for, `added` of them new to the heap.
fn over_ceiling(request: usize, added: usize) -> bool {
    if request < HEAP_CEILING_MIN_REQUEST {
        return false;
    }
    let ceiling = HEAP_CEILING.load(Ordering::Relaxed);
    if ceiling == NO_HEAP_CEILING {
        return false;
    }
    // Atomic loads only: this allocates nothing.
    let live = gusset::get_alloc_stats().live_bytes as u64;
    live.saturating_add(added as u64) > ceiling
}

// SAFETY: every method forwards to `inner` with the caller's arguments, or
// returns null, which `GlobalAlloc` allows for any failed request.
unsafe impl<A: GlobalAlloc> GlobalAlloc for Ceiling<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if over_ceiling(layout.size(), layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's contract for `alloc`, passed through.
        unsafe { self.inner.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if over_ceiling(layout.size(), layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's contract for `alloc_zeroed`, passed through.
        unsafe { self.inner.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract for `dealloc`, passed through.
        unsafe { self.inner.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() && over_ceiling(new_size, new_size - layout.size()) {
            // Null leaves the old block valid, as a failed realloc must.
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's contract for `realloc`, passed through.
        unsafe { self.inner.realloc(ptr, layout, new_size) }
    }
}

/// Sets the Rust heap ceiling and returns the previous one.
/// [`NO_HEAP_CEILING`] (`u64::MAX`) removes it.
///
/// While a ceiling is set, a single allocation of at least
/// [`HEAP_CEILING_MIN_REQUEST`] bytes that would take the live Rust heap
/// (gusset's `Stats` count) past it fails. A fallible allocation reports
/// that as an error; an infallible one would abort the process, which is
/// the bug the ceiling exists to expose. Smaller allocations are never
/// refused. The ceiling is process-wide and takes effect at once. It does
/// not count device memory or gusset buffers, and it is not
/// `SetMemoryCeiling`, which caps what sessions may charge.
#[no_mangle]
pub extern "C" fn ojas_set_heap_ceiling(bytes: u64) -> u64 {
    HEAP_CEILING.swap(bytes, Ordering::Relaxed)
}

/// # Safety
/// `ptr` is readable for `len` bytes for the duration of this call, or
/// `ptr` is null when `len` is 0. The bytes are copied before return.
#[no_mangle]
pub unsafe extern "C" fn ojas_set_model_root(ptr: *const u8, len: usize) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let bytes = if ptr.is_null() {
            if len == 0 {
                &[][..]
            } else {
                ojas_capi::set_last_error("model root pointer is null");
                return -1;
            }
        } else {
            // SAFETY: the caller keeps `ptr` valid for `len` bytes until return.
            unsafe { std::slice::from_raw_parts(ptr, len) }
        };
        let text = match std::str::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => {
                ojas_capi::set_last_error("model root is not utf-8");
                return -1;
            }
        };
        match ojas_capi::set_model_root(text) {
            Ok(()) => {
                ojas_capi::clear_last_error();
                0
            }
            Err(err) => {
                ojas_capi::set_last_error(err);
                -1
            }
        }
    }));
    match result {
        Ok(code) => code,
        Err(_) => {
            ojas_capi::set_last_error("Rust panic in ojas_set_model_root");
            -2
        }
    }
}

/// Registers opcode handlers on the first successful call.
///
/// A later call is a no-op: it does not clear or reinstall handlers.
/// The first call runs gusset's `clear_engine_handlers` before installing
/// ojas opcodes, which removes handlers already registered in the process.
/// That clear is left as gusset defines it. Later calls do not clear again.
#[no_mangle]
pub extern "C" fn ojas_engine_init() -> i32 {
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || match ojas_capi::install_engine() {
                Ok(()) => {
                    ojas_capi::clear_last_error();
                    0
                }
                Err(err) => {
                    ojas_capi::set_last_error(err);
                    -1
                }
            },
        ));
    match result {
        Ok(code) => code,
        Err(_) => {
            ojas_capi::set_last_error("Rust panic in ojas_engine_init");
            -2
        }
    }
}

/// Drops every session. Engine hooks stay registered.
#[no_mangle]
pub extern "C" fn ojas_engine_reset() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ojas_capi::reset_sessions();
    }));
    if result.is_err() {
        ojas_capi::set_last_error("Rust panic in ojas_engine_reset");
    } else {
        ojas_capi::clear_last_error();
    }
}

#[no_mangle]
pub extern "C" fn ojas_last_error_len() -> usize {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ojas_capi::last_error().len()
    }))
    .unwrap_or(0)
}

/// # Safety
/// `dst` is writable for `cap` bytes when it is non-null. The bytes are
/// copied and the pointer is not stored.
#[no_mangle]
pub unsafe extern "C" fn ojas_copy_last_error(dst: *mut u8, cap: usize) -> usize {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if dst.is_null() || cap == 0 {
            return 0;
        }
        let msg = ojas_capi::last_error();
        let n = msg.len().min(cap);
        // SAFETY: `dst` is writable for `cap` bytes, and `n` is at most `cap`.
        unsafe {
            std::ptr::copy_nonoverlapping(msg.as_ptr(), dst, n);
        }
        n
    }))
    .unwrap_or(0)
}

/// Copies the current error in one call and returns its full length.
///
/// # Safety
/// `dst` is writable for `cap` bytes when it is non-null. The bytes are
/// copied and the pointer is not stored. When `cap` is shorter than the
/// message, the prefix is copied and the return value is the full length.
/// The staging buffer is the message's size, not `cap`'s, and is reserved
/// fallibly; if even that is refused, nothing is copied and 0 is returned.
#[no_mangle]
pub unsafe extern "C" fn ojas_take_last_error(dst: *mut u8, cap: usize) -> usize {
    /// Re-reads allowed when another thread replaces the message with a
    /// longer one between sizing the buffer and taking it.
    const ATTEMPTS: usize = 4;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if dst.is_null() || cap == 0 {
            return ojas_capi::last_error().len();
        }
        // `cap` is the caller's buffer size, unbounded by anything here; only
        // the message is staged, so a huge `cap` allocates nothing extra.
        for _ in 0..ATTEMPTS {
            let len = cap.min(ojas_capi::last_error().len());
            let mut buf = Vec::new();
            if buf.try_reserve_exact(len).is_err() {
                return 0;
            }
            buf.resize(len, 0u8);
            let full = ojas_capi::take_last_error(&mut buf);
            if full > len && len < cap {
                // The message grew past the buffer it was sized for and was
                // left in place (`take_last_error` clears only on a whole
                // copy); size again.
                continue;
            }
            let n = full.min(len);
            // SAFETY: `dst` is writable for `cap` bytes; `n <= len <= cap`
            // and `buf` holds `len` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, n);
            }
            return full;
        }
        0
    }))
    .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test owns the process-wide ceiling, so nothing runs beside it.
    #[test]
    fn the_heap_ceiling_refuses_only_large_requests_and_take_last_error_survives_it() {
        // A 4 MiB caller buffer, allocated while there is no ceiling.
        let cap = 4 * HEAP_CEILING_MIN_REQUEST;
        let mut dst: Vec<u8> = Vec::with_capacity(cap);

        assert_eq!(ojas_set_heap_ceiling(0), NO_HEAP_CEILING, "default is none");

        // At or above the floor: refused, and a fallible caller sees an error.
        let mut big: Vec<u8> = Vec::new();
        assert!(big.try_reserve_exact(HEAP_CEILING_MIN_REQUEST).is_err());
        // Growth past the floor through realloc is refused too, and the old
        // block survives it.
        let mut grown: Vec<u8> = Vec::with_capacity(1024);
        grown.extend_from_slice(&[7u8; 1024]);
        assert!(grown.try_reserve_exact(HEAP_CEILING_MIN_REQUEST).is_err());
        assert_eq!(grown, vec![7u8; 1024]);
        // Below the floor: never refused, whatever the ceiling.
        let small = vec![1u8; HEAP_CEILING_MIN_REQUEST - 1];
        assert_eq!(small.len(), HEAP_CEILING_MIN_REQUEST - 1);
        drop(small);

        // The caller's `cap` is past the floor: before the fix this staged
        // `vec![0u8; cap]`, which the ceiling refuses, and an infallible
        // `vec!` that cannot allocate aborts the process (this test binary).
        ojas_capi::set_last_error("boom");
        // SAFETY: `dst` has `cap` bytes of capacity; the call writes at most
        // `cap` and the length is set from what it reports.
        let full = unsafe { ojas_take_last_error(dst.as_mut_ptr(), cap) };
        assert_eq!(full, 4);
        // SAFETY: the call wrote `full` bytes at the front of `dst`.
        unsafe { dst.set_len(full) };
        assert_eq!(dst, b"boom");
        assert_eq!(ojas_capi::last_error(), "", "a whole copy clears it");

        // A short buffer gets the prefix, the full length, and keeps the text.
        ojas_capi::set_last_error("hello world");
        let mut short = [0u8; 5];
        // SAFETY: `short` is writable for 5 bytes.
        let full = unsafe { ojas_take_last_error(short.as_mut_ptr(), short.len()) };
        assert_eq!((full, &short), (11, b"hello"));
        assert_eq!(ojas_capi::last_error(), "hello world");

        // A device upload's host staging copy (`Tensor::to_ne_bytes`) that
        // the allocator refuses is the kinded capacity error Go maps to
        // ErrCapacity, not an unkinded range error.
        let tensor = {
            ojas_set_heap_ceiling(NO_HEAP_CEILING);
            let budget = ojas_core::Budget::new(1 << 30);
            let t = ojas_core::Tensor::zeros(
                &[HEAP_CEILING_MIN_REQUEST],
                ojas_core::DType::F32,
                &budget,
            )
            .unwrap();
            ojas_set_heap_ceiling(0);
            (t, budget)
        };
        match tensor.0.to_ne_bytes() {
            Err(ojas_core::OjasError::CapacityExceeded { requested, .. }) => {
                assert_eq!(requested, 4 * HEAP_CEILING_MIN_REQUEST as u64);
            }
            other => panic!("to_ne_bytes under the ceiling: {other:?}"),
        }
        drop(tensor);

        // Lifted: the large request goes through.
        assert_eq!(ojas_set_heap_ceiling(NO_HEAP_CEILING), 0);
        assert!(big.try_reserve_exact(HEAP_CEILING_MIN_REQUEST).is_ok());
        ojas_capi::clear_last_error();
    }
}
