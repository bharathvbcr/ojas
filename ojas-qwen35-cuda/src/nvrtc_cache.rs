//! A bounded cache of compiled kernel modules.
//!
//! The key is (module name, FNV-1a 64 of the source, source length, the NVRTC
//! options, the NVRTC version): the design's "(source hash, options,
//! architecture, NVRTC version)" (`cuda-backend-scoping.md` §2.C item 1), with
//! the architecture inside the options. A hit also compares the stored source
//! text with the caller's, so a hash collision is refused, not served.
//!
//! The cache holds at most `max_entries` modules and `max_bytes` of compiled
//! image. A new entry evicts the least recently used ones until both bounds
//! hold; an image larger than `max_bytes` on its own is refused. Eviction only
//! drops the cache's handle: a function already loaded from an evicted module
//! keeps its module alive (cudarc's `CudaFunction` holds an `Arc<CudaModule>`).

use crate::error::CudaError;
use crate::kernels::{CompileSpec, KernelModule};

/// FNV-1a, 64-bit: deterministic across runs, platforms and Rust versions,
/// unlike `std`'s `DefaultHasher`.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// What a compiled module is cached under.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// [`KernelModule::name`].
    pub module: &'static str,
    /// [`fnv1a64`] of the source.
    pub source_hash: u64,
    /// Source length in bytes.
    pub source_len: usize,
    /// The NVRTC option strings ([`CompileSpec::options`]).
    pub options: Vec<String>,
    /// `nvrtcVersion` as `(major, minor)`.
    pub nvrtc_version: (i32, i32),
}

impl CacheKey {
    /// The key for `module` compiled with `spec` by NVRTC `nvrtc_version`.
    pub fn new(module: &KernelModule, spec: &CompileSpec, nvrtc_version: (i32, i32)) -> Self {
        CacheKey {
            module: module.name,
            source_hash: fnv1a64(module.source.as_bytes()),
            source_len: module.source.len(),
            options: spec.options(),
            nvrtc_version,
        }
    }
}

/// Hit, miss and eviction counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups served from the cache.
    pub hits: u64,
    /// Lookups that compiled.
    pub misses: u64,
    /// Entries dropped to stay within the bounds.
    pub evictions: u64,
}

struct Entry<V> {
    key: CacheKey,
    source: &'static str,
    value: V,
    bytes: usize,
    last_used: u64,
}

/// The cache. `V` is the loaded module (an `Arc<CudaModule>` on a device).
pub struct NvrtcCache<V> {
    entries: Vec<Entry<V>>,
    max_entries: usize,
    max_bytes: usize,
    bytes: usize,
    clock: u64,
    stats: CacheStats,
}

impl<V: Clone> NvrtcCache<V> {
    /// A cache of at most `max_entries` modules and `max_bytes` image bytes.
    pub fn new(max_entries: usize, max_bytes: usize) -> Result<Self, CudaError> {
        if max_entries == 0 || max_bytes == 0 {
            return Err(CudaError::invalid(
                "NvrtcCache::new",
                format!("bounds must be positive, got {max_entries} entries and {max_bytes} bytes"),
            ));
        }
        Ok(NvrtcCache {
            entries: Vec::new(),
            max_entries,
            max_bytes,
            bytes: 0,
            clock: 0,
            stats: CacheStats::default(),
        })
    }

    /// Counts so far.
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Entries held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds nothing.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Image bytes held.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The cached value for `key`, or the result of `compile`, which returns
    /// the value and its image size in bytes. A failed compile caches nothing.
    pub fn get_or_compile(
        &mut self,
        key: CacheKey,
        source: &'static str,
        compile: impl FnOnce() -> Result<(V, usize), CudaError>,
    ) -> Result<V, CudaError> {
        self.clock += 1;
        if let Some(entry) = self.entries.iter_mut().find(|e| e.key == key) {
            if entry.source != source {
                return Err(CudaError::invalid(
                    "NvrtcCache",
                    format!(
                        "module {} hashes to {:#018x} like a cached module with different source",
                        key.module, key.source_hash
                    ),
                ));
            }
            entry.last_used = self.clock;
            self.stats.hits += 1;
            return Ok(entry.value.clone());
        }
        let (value, bytes) = compile()?;
        if bytes > self.max_bytes {
            return Err(CudaError::capacity(
                format!("NvrtcCache {}", key.module),
                format!(
                    "a {bytes} byte image exceeds the cache bound of {} bytes",
                    self.max_bytes
                ),
            ));
        }
        self.stats.misses += 1;
        while self.entries.len() >= self.max_entries || self.bytes + bytes > self.max_bytes {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i);
            match oldest {
                Some(i) => {
                    let gone = self.entries.swap_remove(i);
                    self.bytes -= gone.bytes;
                    self.stats.evictions += 1;
                }
                None => break,
            }
        }
        self.bytes += bytes;
        self.entries.push(Entry {
            key,
            source,
            value: value.clone(),
            bytes,
            last_used: self.clock,
        });
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{STRICT_SM90, STRICT_SM90A};

    const SRC_A: &str = "extern \"C\" __global__ void a() {}";
    const SRC_B: &str = "extern \"C\" __global__ void b() {}";
    const MOD_A: KernelModule = KernelModule {
        name: "a",
        source: SRC_A,
        entries: &["a"],
    };
    const MOD_B: KernelModule = KernelModule {
        name: "b",
        source: SRC_B,
        entries: &["b"],
    };

    #[test]
    fn fnv1a64_matches_its_published_test_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn a_second_lookup_hits_and_does_not_compile() {
        let mut cache = NvrtcCache::<u32>::new(4, 1000).unwrap();
        let key = CacheKey::new(&MOD_A, &STRICT_SM90, (12, 8));
        let mut compiles = 0;
        for _ in 0..3 {
            let v = cache
                .get_or_compile(key.clone(), SRC_A, || {
                    compiles += 1;
                    Ok((7, 10))
                })
                .unwrap();
            assert_eq!(v, 7);
        }
        assert_eq!(compiles, 1);
        assert_eq!(
            cache.stats(),
            CacheStats {
                hits: 2,
                misses: 1,
                evictions: 0
            }
        );
    }

    #[test]
    fn options_and_nvrtc_version_are_part_of_the_key() {
        let a90 = CacheKey::new(&MOD_A, &STRICT_SM90, (12, 8));
        assert_ne!(a90, CacheKey::new(&MOD_A, &STRICT_SM90A, (12, 8)));
        assert_ne!(a90, CacheKey::new(&MOD_A, &STRICT_SM90, (12, 9)));
        let fmad_on = CompileSpec {
            fmad: true,
            ..STRICT_SM90
        };
        assert_ne!(a90, CacheKey::new(&MOD_A, &fmad_on, (12, 8)));
    }

    #[test]
    fn a_colliding_hash_with_different_source_is_refused() {
        let mut cache = NvrtcCache::<u32>::new(4, 1000).unwrap();
        let key = CacheKey::new(&MOD_A, &STRICT_SM90, (12, 8));
        cache
            .get_or_compile(key.clone(), SRC_A, || Ok((1, 1)))
            .unwrap();
        // Same key, other text: what a 64-bit collision would look like.
        let err = cache.get_or_compile(key, SRC_B, || Ok((2, 1))).unwrap_err();
        assert_eq!(err.kind(), "invalid", "{err}");
    }

    #[test]
    fn the_least_recently_used_entry_is_evicted_at_the_entry_bound() {
        let mut cache = NvrtcCache::<u32>::new(1, 1000).unwrap();
        let ka = CacheKey::new(&MOD_A, &STRICT_SM90, (12, 8));
        let kb = CacheKey::new(&MOD_B, &STRICT_SM90, (12, 8));
        cache
            .get_or_compile(ka.clone(), SRC_A, || Ok((1, 10)))
            .unwrap();
        cache.get_or_compile(kb, SRC_B, || Ok((2, 10))).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats().evictions, 1);
        let mut recompiled = false;
        cache
            .get_or_compile(ka, SRC_A, || {
                recompiled = true;
                Ok((1, 10))
            })
            .unwrap();
        assert!(recompiled);
    }

    #[test]
    fn the_byte_bound_evicts_and_an_oversized_image_is_refused() {
        let mut cache = NvrtcCache::<u32>::new(8, 100).unwrap();
        let ka = CacheKey::new(&MOD_A, &STRICT_SM90, (12, 8));
        let kb = CacheKey::new(&MOD_B, &STRICT_SM90, (12, 8));
        cache.get_or_compile(ka, SRC_A, || Ok((1, 60))).unwrap();
        cache
            .get_or_compile(kb.clone(), SRC_B, || Ok((2, 60)))
            .unwrap();
        assert_eq!((cache.len(), cache.bytes()), (1, 60));
        let kc = CacheKey::new(&MOD_B, &STRICT_SM90A, (12, 8));
        let err = cache
            .get_or_compile(kc, SRC_B, || Ok((3, 101)))
            .unwrap_err();
        assert_eq!(err.kind(), "capacity");
        assert_eq!(
            cache.bytes(),
            60,
            "a refused image must not change the cache"
        );
    }

    #[test]
    fn a_failed_compile_caches_nothing() {
        let mut cache = NvrtcCache::<u32>::new(2, 100).unwrap();
        let ka = CacheKey::new(&MOD_A, &STRICT_SM90, (12, 8));
        let err = cache
            .get_or_compile(ka, SRC_A, || {
                Err(CudaError::Compile {
                    module: "a".into(),
                    detail: "syntax".into(),
                })
            })
            .unwrap_err();
        assert_eq!(err.kind(), "compile");
        assert!(cache.is_empty());
        assert!(NvrtcCache::<u32>::new(0, 1).is_err());
    }
}
