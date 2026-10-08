//! Stable C ABI. C++, Swift, and Python/ctypes wrappers own no Metal objects.

use std::ffi::{c_char, CStr};
use std::path::Path;
use std::ptr;
use std::sync::{Mutex, OnceLock};

use crate::engine::{EngineCreateConfig, TrainingEngine};

pub const ARCH02_OK: i32 = 0;
pub const ARCH02_ERROR: i32 = -1;
pub const ARCH02_INVALID_ARGUMENT: i32 = -2;

#[repr(C)]
pub struct Arch02EngineHandle {
    engine: TrainingEngine,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Arch02StepMetrics {
    pub completed_step: u64,
    pub loss: f32,
    pub grad_norm: f32,
    pub clip_factor: f32,
    pub lr_multiplier: f32,
    pub dispatches: u64,
}

fn error_slot() -> &'static Mutex<String> {
    static LAST_ERROR: OnceLock<Mutex<String>> = OnceLock::new();
    LAST_ERROR.get_or_init(|| Mutex::new(String::new()))
}

fn set_error(error: impl Into<String>) -> i32 {
    if let Ok(mut slot) = error_slot().lock() {
        *slot = error.into();
    }
    ARCH02_ERROR
}

/// Run an `extern "C"` body. A panic must become an error code, not abort the host.
fn ffi_catch<T: Send>(name: &str, fallback: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => {
            let _ = set_error(format!("{name} panicked"));
            fallback
        }
    }
}

unsafe fn required_str<'a>(value: *const c_char, name: &str) -> Result<&'a str, i32> {
    if value.is_null() {
        set_error(format!("{name} must not be null"));
        return Err(ARCH02_INVALID_ARGUMENT);
    }
    CStr::from_ptr(value).to_str().map_err(|_| {
        set_error(format!("{name} is not valid UTF-8"));
        ARCH02_INVALID_ARGUMENT
    })
}

#[no_mangle]
pub unsafe extern "C" fn arch02_engine_create(
    config_json: *const c_char,
    out_engine: *mut *mut Arch02EngineHandle,
) -> i32 {
    ffi_catch("arch02_engine_create", ARCH02_ERROR, || unsafe {
        arch02_engine_create_inner(config_json, out_engine)
    })
}

unsafe fn arch02_engine_create_inner(config_json: *const c_char, out_engine: *mut *mut Arch02EngineHandle) -> i32 {
    if out_engine.is_null() {
        return set_error("out_engine must not be null");
    }
    *out_engine = ptr::null_mut();
    let config = if config_json.is_null() {
        EngineCreateConfig::default()
    } else {
        let json = match required_str(config_json, "config_json") {
            Ok(value) => value,
            Err(code) => return code,
        };
        match serde_json::from_str(json) {
            Ok(value) => value,
            Err(error) => return set_error(format!("invalid engine config JSON: {error}")),
        }
    };
    match TrainingEngine::create(config) {
        Ok(engine) => {
            *out_engine = Box::into_raw(Box::new(Arch02EngineHandle { engine }));
            ARCH02_OK
        }
        Err(error) => set_error(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn arch02_engine_load(
    checkpoint_path: *const c_char,
    out_engine: *mut *mut Arch02EngineHandle,
) -> i32 {
    ffi_catch("arch02_engine_load", ARCH02_ERROR, || unsafe {
        arch02_engine_load_inner(checkpoint_path, out_engine)
    })
}

unsafe fn arch02_engine_load_inner(
    checkpoint_path: *const c_char,
    out_engine: *mut *mut Arch02EngineHandle,
) -> i32 {
    if out_engine.is_null() {
        return set_error("out_engine must not be null");
    }
    *out_engine = ptr::null_mut();
    let path = match required_str(checkpoint_path, "checkpoint_path") {
        Ok(value) => value,
        Err(code) => return code,
    };
    match TrainingEngine::load(Path::new(path)) {
        Ok(engine) => {
            *out_engine = Box::into_raw(Box::new(Arch02EngineHandle { engine }));
            ARCH02_OK
        }
        Err(error) => set_error(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn arch02_engine_expected_tokens(engine: *const Arch02EngineHandle) -> usize {
    ffi_catch("arch02_engine_expected_tokens", 0, || unsafe {
        arch02_engine_expected_tokens_inner(engine)
    })
}

unsafe fn arch02_engine_expected_tokens_inner(engine: *const Arch02EngineHandle) -> usize {
    engine
        .as_ref()
        .map(|handle| handle.engine.expected_tokens())
        .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn arch02_engine_train(
    engine: *mut Arch02EngineHandle,
    input_ids: *const i32,
    target_ids: *const i32,
    token_count: usize,
    out_metrics: *mut Arch02StepMetrics,
) -> i32 {
    ffi_catch("arch02_engine_train", ARCH02_ERROR, || unsafe {
        arch02_engine_train_inner(engine, input_ids, target_ids, token_count, out_metrics)
    })
}

unsafe fn arch02_engine_train_inner(
    engine: *mut Arch02EngineHandle,
    input_ids: *const i32,
    target_ids: *const i32,
    token_count: usize,
    out_metrics: *mut Arch02StepMetrics,
) -> i32 {
    let Some(handle) = engine.as_mut() else {
        return set_error("engine must not be null");
    };
    if input_ids.is_null() || target_ids.is_null() || out_metrics.is_null() {
        return set_error("input_ids, target_ids, and out_metrics must not be null");
    }
    let inputs = std::slice::from_raw_parts(input_ids, token_count);
    let targets = std::slice::from_raw_parts(target_ids, token_count);
    match handle.engine.train_step(inputs, targets) {
        Ok(metrics) => {
            *out_metrics = Arch02StepMetrics {
                completed_step: metrics.completed_step as u64,
                loss: metrics.loss,
                grad_norm: metrics.grad_norm,
                clip_factor: metrics.clip_factor,
                lr_multiplier: metrics.lr_multiplier,
                dispatches: metrics.dispatches as u64,
            };
            ARCH02_OK
        }
        Err(error) => set_error(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn arch02_engine_evaluate(
    engine: *mut Arch02EngineHandle,
    input_ids: *const i32,
    target_ids: *const i32,
    token_count: usize,
    out_loss: *mut f32,
) -> i32 {
    ffi_catch("arch02_engine_evaluate", ARCH02_ERROR, || unsafe {
        arch02_engine_evaluate_inner(engine, input_ids, target_ids, token_count, out_loss)
    })
}

unsafe fn arch02_engine_evaluate_inner(
    engine: *mut Arch02EngineHandle,
    input_ids: *const i32,
    target_ids: *const i32,
    token_count: usize,
    out_loss: *mut f32,
) -> i32 {
    let Some(handle) = engine.as_mut() else {
        return set_error("engine must not be null");
    };
    if input_ids.is_null() || target_ids.is_null() || out_loss.is_null() {
        return set_error("input_ids, target_ids, and out_loss must not be null");
    }
    let inputs = std::slice::from_raw_parts(input_ids, token_count);
    let targets = std::slice::from_raw_parts(target_ids, token_count);
    match handle.engine.evaluate(inputs, targets) {
        Ok(loss) => {
            *out_loss = loss;
            ARCH02_OK
        }
        Err(error) => set_error(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn arch02_engine_save(
    engine: *const Arch02EngineHandle,
    checkpoint_path: *const c_char,
) -> i32 {
    ffi_catch("arch02_engine_save", ARCH02_ERROR, || unsafe {
        arch02_engine_save_inner(engine, checkpoint_path)
    })
}

unsafe fn arch02_engine_save_inner(engine: *const Arch02EngineHandle, checkpoint_path: *const c_char) -> i32 {
    let Some(handle) = engine.as_ref() else {
        return set_error("engine must not be null");
    };
    let path = match required_str(checkpoint_path, "checkpoint_path") {
        Ok(value) => value,
        Err(code) => return code,
    };
    match handle.engine.save(Path::new(path)) {
        Ok(()) => ARCH02_OK,
        Err(error) => set_error(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn arch02_engine_destroy(engine: *mut Arch02EngineHandle) {
    ffi_catch("arch02_engine_destroy", (), || unsafe {
        if !engine.is_null() {
            drop(Box::from_raw(engine));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn arch02_last_error(buffer: *mut c_char, capacity: usize) -> usize {
    ffi_catch("arch02_last_error", 0, || unsafe { arch02_last_error_inner(buffer, capacity) })
}

unsafe fn arch02_last_error_inner(buffer: *mut c_char, capacity: usize) -> usize {
    let message = error_slot()
        .lock()
        .map(|slot| slot.clone())
        .unwrap_or_else(|_| "error lock poisoned".into());
    let required = message.len() + 1;
    if buffer.is_null() || capacity == 0 {
        return required;
    }
    let copy_len = message.len().min(capacity - 1);
    ptr::copy_nonoverlapping(message.as_ptr(), buffer.cast::<u8>(), copy_len);
    *buffer.add(copy_len) = 0;
    required
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_error_reports_required_capacity_and_terminates() {
        set_error("ffi-test");
        let required = unsafe { arch02_last_error(ptr::null_mut(), 0) };
        assert_eq!(required, 9);
        let mut buffer = vec![0i8; required];
        unsafe { arch02_last_error(buffer.as_mut_ptr(), buffer.len()) };
        assert_eq!(
            unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_str().unwrap(),
            "ffi-test"
        );
    }

    /// A panic inside an export used to unwind out of `extern "C"` and abort the host.
    #[test]
    fn a_panic_in_an_export_is_an_error_not_an_abort() {
        let code = ffi_catch("unit", ARCH02_ERROR, || -> i32 { panic!("boom") });
        assert_eq!(code, ARCH02_ERROR);
        let required = unsafe { arch02_last_error(ptr::null_mut(), 0) };
        assert!(required > 1, "the panic must be recorded");
    }
}
