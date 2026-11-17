//! Counting global allocators. Each forwards every call to [`System`]
//! unchanged and only does bookkeeping. They answer two different questions,
//! so they are two types:
//!
//! - [`PeakAlloc`]: how many heap bytes are live, and what was the high-water
//!   mark since a reset. Process-wide, so a test that measures with it must
//!   keep every other thread quiet (hold a `SERIAL` mutex, or be the binary's
//!   only test).
//! - [`ThreadAllocCount`]: how many allocation calls the current thread made.
//!   Per thread, so parallel tests do not see each other.
//!
//! A test binary declares its own `#[global_allocator] static`, since the
//! allocator is per binary:
//!
//! ```text
//! #[global_allocator]
//! static ALLOC: ojas_test_support::PeakAlloc = ojas_test_support::PeakAlloc::new();
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The system allocator, counting live heap bytes and their high-water mark.
///
/// `realloc` charges what the process physically held. When the block grows
/// or shrinks in place, only the net change is counted. When the block moves,
/// both blocks are live during the copy, so the peak sees old plus new before
/// the old block is released.
pub struct PeakAlloc {
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl PeakAlloc {
    pub const fn new() -> Self {
        Self {
            live: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    fn grow(&self, bytes: usize) {
        let now = self.live.fetch_add(bytes, Ordering::SeqCst) + bytes;
        self.peak.fetch_max(now, Ordering::SeqCst);
    }

    fn shrink(&self, bytes: usize) {
        self.live.fetch_sub(bytes, Ordering::SeqCst);
    }

    /// Heap bytes live now.
    pub fn live(&self) -> usize {
        self.live.load(Ordering::SeqCst)
    }

    /// The highest [`live`](Self::live) value since the last
    /// [`reset_peak`](Self::reset_peak).
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }

    /// Restart the high-water mark at the bytes live now.
    pub fn reset_peak(&self) {
        self.peak.store(self.live(), Ordering::SeqCst);
    }

    /// Run `f` and return its output with the heap high-water growth over the
    /// bytes live when it started. The output is still alive, so its bytes
    /// are inside the growth; drop it after reading the number.
    pub fn measure<T>(&self, f: impl FnOnce() -> T) -> (T, usize) {
        let base = self.live();
        self.peak.store(base, Ordering::SeqCst);
        let out = f();
        let grew = self
            .peak()
            .checked_sub(base)
            .expect("the heap peak fell below its starting point");
        (out, grew)
    }
}

impl Default for PeakAlloc {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged, so the `GlobalAlloc` contract is `System`'s; the counters are
// atomics updated only on success.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            self.grow(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            self.grow(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) };
        self.shrink(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if p != ptr {
                // Moved: `System` held both blocks while it copied.
                self.grow(new_size);
                self.shrink(layout.size());
            } else if new_size >= layout.size() {
                self.grow(new_size - layout.size());
            } else {
                self.shrink(layout.size() - new_size);
            }
        }
        p
    }
}

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

/// Allocation calls (`alloc`, `alloc_zeroed`, `realloc`) made by the current
/// thread, read with [`thread_allocs`]. Deallocations are not counted.
pub struct ThreadAllocCount;

/// Allocation calls the current thread has made so far.
pub fn thread_allocs() -> usize {
    ALLOCS.with(Cell::get)
}

fn bump() {
    // `try_with`: an allocation during thread teardown must not panic.
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

// SAFETY: every method forwards to `System` unchanged and only bumps a
// thread-local counter.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for ThreadAllocCount {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
