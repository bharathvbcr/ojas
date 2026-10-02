//! macOS `sysctlbyname` reads shared by the host and topology probes.
//!
//! A missing name, a failed call, or a value of an unexpected width is
//! `None`. Nothing here substitutes a default.

#![allow(unsafe_code)]

/// Longest string value read. `hw.perflevelN.name` is a short word.
const STRING_MAX: usize = 256;

/// An unsigned integer sysctl of 1 to 8 bytes, little-endian.
///
/// `name` must end in a NUL byte; one that does not is refused rather than
/// read past.
pub(crate) fn u64_by_name(name: &[u8]) -> Option<u64> {
    if name.last() != Some(&0) {
        return None;
    }
    let mut buf = [0u8; 8];
    let mut len = buf.len();
    // SAFETY: `name` is NUL-terminated (checked above); `buf` is writable for
    // `len` bytes and the kernel writes at most `len` bytes, updating `len`.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const libc::c_char,
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len == 0 || len > buf.len() {
        return None;
    }
    let mut tmp = [0u8; 8];
    tmp[..len].copy_from_slice(&buf[..len]);
    Some(u64::from_le_bytes(tmp))
}

/// A NUL-terminated UTF-8 string sysctl of at most [`STRING_MAX`] bytes.
pub(crate) fn string_by_name(name: &[u8]) -> Option<String> {
    if name.last() != Some(&0) {
        return None;
    }
    let mut buf = [0u8; STRING_MAX];
    let mut len = buf.len();
    // SAFETY: as in `u64_by_name`; a value longer than the buffer fails with
    // ENOMEM and is `None`.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const libc::c_char,
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len == 0 || len > buf.len() {
        return None;
    }
    let bytes = &buf[..len];
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let text = std::str::from_utf8(&bytes[..end]).ok()?.trim();
    if text.is_empty() {
        return None;
    }
    Some(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unterminated_names_are_refused_without_a_call() {
        assert_eq!(u64_by_name(b"hw.memsize"), None);
        assert_eq!(string_by_name(b"hw.perflevel0.name"), None);
        assert_eq!(u64_by_name(b""), None);
    }

    #[test]
    fn missing_names_are_none() {
        assert_eq!(u64_by_name(b"hw.ojas.no.such.name\0"), None);
        assert_eq!(string_by_name(b"hw.ojas.no.such.name\0"), None);
    }

    #[test]
    fn known_names_read() {
        assert!(matches!(u64_by_name(b"hw.memsize\0"), Some(n) if n >= 1 << 20));
        assert!(matches!(u64_by_name(b"hw.pagesize\0"), Some(n) if n.is_power_of_two()));
        // A string sysctl present on every macOS release.
        assert!(string_by_name(b"kern.ostype\0").is_some());
    }
}
