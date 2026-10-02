//! Which CUDA libraries the process can load, and which it did load.
//!
//! cudarc 0.19.10 `dlopen`s bare sonames from a fixed candidate list
//! (`cudarc/src/lib.rs:204-243`: `libcublas.so`, ..., `libcublas.so.12`, ...)
//! and **panics** when none loads (`cudarc/src/lib.rs:199-201`, called from
//! every `sys::culib()`). With bare names, glibc's loader searches
//! `LD_LIBRARY_PATH` as it was **when the process started**, then the
//! ld.so cache and the default directories. So:
//! - the box's venv libraries (`.../site-packages/nvidia/{cublas,cuda_nvrtc}/lib`)
//!   are found only when `LD_LIBRARY_PATH` names them at launch; setting it
//!   from inside the process changes nothing;
//! - [`crate::runtime`] probes every library with cudarc's own
//!   `is_culib_present` before the first cudarc call and refuses by name,
//!   using this module to say where it looked.
//!
//! `libcublas.so.12` needs `libcublasLt.so.12` from the same directory, and
//! NVRTC loads `libnvrtc-builtins.so.12.8` by name at compile time, so both
//! directories must be on `LD_LIBRARY_PATH`, not only the files cudarc names.

use std::path::{Path, PathBuf};

/// A library cudarc loads, by the names its `sys` module searches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LibrarySpec {
    /// What reports and refusals call it.
    pub label: &'static str,
    /// The `lib_names` cudarc expands into candidate sonames.
    pub cudarc_names: &'static [&'static str],
    /// File-name prefixes that identify it in `/proc/self/maps`.
    pub mapped_prefixes: &'static [&'static str],
}

/// The driver: `cudarc/src/driver/sys/mod.rs` searches `cuda`, `nvcuda`.
pub const DRIVER: LibrarySpec = LibrarySpec {
    label: "libcuda",
    cudarc_names: &["cuda", "nvcuda"],
    mapped_prefixes: &["libcuda.so"],
};

/// NVRTC: `cudarc/src/nvrtc/sys/mod.rs` searches `nvrtc`.
pub const NVRTC: LibrarySpec = LibrarySpec {
    label: "libnvrtc",
    cudarc_names: &["nvrtc"],
    mapped_prefixes: &["libnvrtc.so", "libnvrtc-builtins.so"],
};

/// cuBLAS: `cudarc/src/cublas/sys/mod.rs` searches `cublas`.
pub const CUBLAS: LibrarySpec = LibrarySpec {
    label: "libcublas",
    cudarc_names: &["cublas"],
    mapped_prefixes: &["libcublas.so", "libcublasLt.so"],
};

/// Every library the runtime needs before it makes a cudarc call.
pub const REQUIRED: [LibrarySpec; 3] = [DRIVER, NVRTC, CUBLAS];

/// The directories in an `LD_LIBRARY_PATH` value, in order, empties dropped.
pub fn path_dirs(value: Option<&str>) -> Vec<PathBuf> {
    value
        .map(|v| {
            v.split(':')
                .filter(|d| !d.is_empty())
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default()
}

/// Every `dir/name` that exists, for each dir in order and each candidate.
pub fn locate(dirs: &[PathBuf], candidates: &[String]) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for dir in dirs {
        for name in candidates {
            let p = dir.join(name);
            if p.is_file() {
                found.push(p);
            }
        }
    }
    found
}

/// The refusal text for a library that did not load: the names searched,
/// the `LD_LIBRARY_PATH` the process started with, and any candidate file
/// that exists in those directories (present but not loadable points at a
/// missing dependency such as `libcublasLt.so.12`).
pub fn missing_detail(spec: &LibrarySpec, candidates: &[String], ld_path: Option<&str>) -> String {
    let dirs = path_dirs(ld_path);
    let present = locate(&dirs, candidates);
    let present = if present.is_empty() {
        "none of them exists in those directories".to_string()
    } else {
        format!(
            "present but not loadable (a dependency is missing?): {}",
            present
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    format!(
        "{}: searched {:?}; LD_LIBRARY_PATH={}; {present}",
        spec.label,
        candidates,
        ld_path.unwrap_or("<unset>")
    )
}

/// The absolute paths in a `/proc/self/maps` text whose file name starts with
/// one of `prefixes`, sorted and de-duplicated.
pub fn mapped_paths(maps: &str, prefixes: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = maps
        .lines()
        .filter_map(|line| line.split_whitespace().nth(5))
        .filter(|path| path.starts_with('/'))
        .filter(|path| {
            Path::new(path)
                .file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| prefixes.iter().any(|p| f.starts_with(p)))
        })
        .map(str::to_string)
        .collect();
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ld_library_path_splits_in_order_and_drops_empties() {
        assert_eq!(
            path_dirs(Some("/a::/b/c:")),
            vec![PathBuf::from("/a"), PathBuf::from("/b/c")]
        );
        assert!(path_dirs(None).is_empty());
    }

    #[test]
    fn maps_parsing_finds_the_loaded_cuda_libraries() {
        let maps = "\
aaaa-bbbb r-xp 00000000 08:01 1 /usr/lib/aarch64-linux-gnu/libcuda.so.580.105.08
cccc-dddd r--p 00001000 08:01 1 /usr/lib/aarch64-linux-gnu/libcuda.so.580.105.08
eeee-ffff r-xp 00000000 08:01 2 /home/u/venv/nvidia/cublas/lib/libcublasLt.so.12
eeee-ffff r-xp 00000000 08:01 3 /home/u/venv/nvidia/cublas/lib/libcublas.so.12
1111-2222 rw-p 00000000 00:00 0 [heap]
3333-4444 r-xp 00000000 08:01 4 /usr/lib/libc.so.6
";
        assert_eq!(
            mapped_paths(maps, CUBLAS.mapped_prefixes),
            vec![
                "/home/u/venv/nvidia/cublas/lib/libcublas.so.12".to_string(),
                "/home/u/venv/nvidia/cublas/lib/libcublasLt.so.12".to_string(),
            ]
        );
        assert_eq!(
            mapped_paths(maps, DRIVER.mapped_prefixes),
            vec!["/usr/lib/aarch64-linux-gnu/libcuda.so.580.105.08".to_string()]
        );
        assert!(mapped_paths(maps, NVRTC.mapped_prefixes).is_empty());
    }

    #[test]
    fn the_refusal_says_whether_a_candidate_file_exists() {
        let dir = std::env::temp_dir().join(format!("qd-libprobe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("libcublas.so.12");
        std::fs::write(&file, b"not an elf").unwrap();
        let ld = dir.display().to_string();
        let names = vec!["libcublas.so".to_string(), "libcublas.so.12".to_string()];
        let with_file = missing_detail(&CUBLAS, &names, Some(&ld));
        assert!(
            with_file.contains("present but not loadable"),
            "{with_file}"
        );
        assert!(with_file.contains("libcublas.so.12"), "{with_file}");
        let unset = missing_detail(&CUBLAS, &names, None);
        assert!(unset.contains("LD_LIBRARY_PATH=<unset>"), "{unset}");
        assert!(unset.contains("none of them exists"), "{unset}");
        std::fs::remove_file(&file).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }
}
