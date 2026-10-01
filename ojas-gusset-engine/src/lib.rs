//! One static library for the Go process (R14).
//!
//! `ojas-capi` is an rlib. This crate is the only `staticlib`, and its
//! archive is `libgusset.a` because cgo passes `-lgusset`.

// Counting is installed so `gusset.Stats` and `AdviseMemoryLimit` can see Rust
// heap bytes. It forwards each allocation to `System` and returns a null
// pointer when that allocation fails. Infallible Rust allocations (`vec!`,
// `format!`, channels) still abort the process. This static does not install
// an allocation-error hook and does not make those allocations fallible.
#[global_allocator]
static GLOBAL: gusset::Counting<std::alloc::System> = gusset::Counting::new(std::alloc::System);

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
#[no_mangle]
pub unsafe extern "C" fn ojas_take_last_error(dst: *mut u8, cap: usize) -> usize {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if dst.is_null() || cap == 0 {
            return ojas_capi::last_error().len();
        }
        let mut buf = vec![0u8; cap];
        let full = ojas_capi::take_last_error(&mut buf);
        let n = full.min(cap);
        // SAFETY: `dst` is writable for `cap` bytes, and `n` is at most `cap`.
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, n);
        }
        full
    }))
    .unwrap_or(0)
}
