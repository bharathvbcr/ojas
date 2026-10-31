//! The AdamW element math comes from tessl's `qwen35_adamw_math.h`. That
//! replaced `ojas_adamw_elem` (a 7-float `OjasAdamW` scalar struct) in
//! `ojas_backend.metal`; this file proves the swap changed no bit.
//!
//! The reference is the retired kernel pair, written out below verbatim and
//! compiled at test time with the flags `build.rs` gives `ojas_backend.metal`
//! (`-ffp-contract=off -fmetal-math-mode=safe`, `-std=metal4.0`). It runs on
//! its own tessl runtime. `MetalBackend::adamw_step` is the new path. Both see
//! the same inputs and the same f32 scalars (formed here, independently of
//! `adamw_scalars`, so the 8th field `grad_scale = 1` and the field order are
//! under test too).
//!
//! The kernels are transactional (any non-finite input or result blocks the
//! whole call), so a bulk compare of edge inputs would mostly compare
//! "unchanged". The reference therefore classifies every element on the
//! device with the old arithmetic, and the compare runs on the clean
//! elements only, compacted into one call: if the new arithmetic flagged an
//! element the old did not, `adamw_step` fails and the test names it. The
//! uncompacted array is then run through both and must be refused by both,
//! with p, m and v untouched.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::*;
use ojas_core::{check_adamw, AdamWConfig, Backend, OjasError};
use tessl::dispatch::{dispatch_1d, set_gpu_buf_offset, set_u32};
use tessl::runtime::GpuRuntime;

/// The kernels `ojas_backend.metal` carried before the swap, with a
/// per-element classifier added. `ojas_adamw_elem` and `OjasAdamW` are the
/// retired code, unedited.
const REFERENCE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define ST_IN 0u
#define ST_OUT 1u

struct OjasAdamW {
    float decay_mul;       // 1 - lr * weight_decay
    float lerp_w;          // 1 - beta1
    float beta2;
    float one_minus_beta2;
    float step_size;       // lr / (1 - beta1^step)
    float bc2_sqrt;        // sqrt(1 - beta2^step)
    float eps;
};

inline void ojas_adamw_elem(
    float p, float g, float m, float v, constant OjasAdamW &a,
    thread float &p_out, thread float &m_out, thread float &v_out)
{
    float w = p * a.decay_mul;
    const float diff = g - m;
    const float mi = a.lerp_w < 0.5f ? m + a.lerp_w * diff : g - diff * (1.0f - a.lerp_w);
    const float vi = v * a.beta2 + (a.one_minus_beta2 * g) * g;
    const float denom = precise::divide(precise::sqrt(vi), a.bc2_sqrt) + a.eps;
    w = w + (-a.step_size) * precise::divide(mi, denom);
    p_out = w;
    m_out = mi;
    v_out = vi;
}

kernel void ref_adamw_check(
    device const float *p [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device const float *m [[buffer(2)]],
    device const float *v [[buffer(3)]],
    device atomic_uint *st [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant OjasAdamW &a [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float pi = p[i], gi = g[i], mi = m[i], vi = v[i];
    if (!(isfinite(pi) && isfinite(gi) && isfinite(mi) && isfinite(vi))) {
        atomic_store_explicit(&st[ST_IN], 1u, memory_order_relaxed);
        return;
    }
    float po, mo, vo;
    ojas_adamw_elem(pi, gi, mi, vi, a, po, mo, vo);
    if (!(isfinite(po) && isfinite(mo) && isfinite(vo))) {
        atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
    }
}

kernel void ref_adamw_apply(
    device float *p [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device float *m [[buffer(2)]],
    device float *v [[buffer(3)]],
    device const uint *st [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant OjasAdamW &a [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (st[ST_IN] != 0u || st[ST_OUT] != 0u) {
        return;
    }
    float po, mo, vo;
    ojas_adamw_elem(p[i], g[i], m[i], v[i], a, po, mo, vo);
    p[i] = po;
    m[i] = mo;
    v[i] = vo;
}

// Per element: 0 clean, 1 non-finite input, 2 non-finite result.
kernel void ref_adamw_classify(
    device const float *p [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device const float *m [[buffer(2)]],
    device const float *v [[buffer(3)]],
    device uint *flag [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant OjasAdamW &a [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float pi = p[i], gi = g[i], mi = m[i], vi = v[i];
    if (!(isfinite(pi) && isfinite(gi) && isfinite(mi) && isfinite(vi))) {
        flag[i] = 1u;
        return;
    }
    float po, mo, vo;
    ojas_adamw_elem(pi, gi, mi, vi, a, po, mo, vo);
    flag[i] = (isfinite(po) && isfinite(mo) && isfinite(vo)) ? 0u : 2u;
}
"#;

fn xcrun(args: &[&str]) -> String {
    let mut cmd = Command::new("xcrun");
    if std::env::var_os("DEVELOPER_DIR").is_none() {
        let xcode = "/Applications/Xcode.app/Contents/Developer";
        if Path::new(xcode).is_dir() {
            cmd.env("DEVELOPER_DIR", xcode);
        }
    }
    let out = cmd
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("xcrun {args:?}: {e}"));
    assert!(
        out.status.success(),
        "xcrun {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// Compile [`REFERENCE_MSL`] the way `build.rs` compiles the production
/// kernels and return the metallib's path.
fn build_reference_metallib() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("adamw_msl_parity");
    std::fs::create_dir_all(&dir).expect("create the scratch directory");
    let src = dir.join("adamw_reference.metal");
    let air = dir.join("adamw_reference.air");
    let lib = dir.join("adamw_reference.metallib");
    std::fs::write(&src, REFERENCE_MSL).expect("write the reference kernel");
    let sdk = xcrun(&["--sdk", "macosx", "--show-sdk-path"]);
    let metal = xcrun(&["-f", "metal"]);
    let metallib = xcrun(&["-f", "metallib"]);
    let mut compile = Command::new(&metal);
    if std::env::var_os("DEVELOPER_DIR").is_none() {
        let xcode = "/Applications/Xcode.app/Contents/Developer";
        if Path::new(xcode).is_dir() {
            compile.env("DEVELOPER_DIR", xcode);
        }
    }
    let status = compile
        .args([
            "-std=metal4.0",
            "-O2",
            "-ffp-contract=off",
            "-fmetal-math-mode=safe",
            "-isysroot",
            &sdk,
            "-mmacosx-version-min=26.0",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&air)
        .status()
        .expect("spawn metal");
    assert!(status.success(), "the reference kernel failed to compile");
    let mut link = Command::new(&metallib);
    if std::env::var_os("DEVELOPER_DIR").is_none() {
        let xcode = "/Applications/Xcode.app/Contents/Developer";
        if Path::new(xcode).is_dir() {
            link.env("DEVELOPER_DIR", xcode);
        }
    }
    let status = link
        .arg(&air)
        .arg("-o")
        .arg(&lib)
        .status()
        .expect("spawn metallib");
    assert!(status.success(), "the reference metallib failed to link");
    lib
}

/// The retired scalar struct, formed as `adamw_scalars` formed it. `None`
/// when a scalar does not survive the f32 rounding (the host refuses before
/// any device sees it).
fn reference_scalars(config: AdamWConfig, step: u64) -> Option<[f32; 7]> {
    let (_, bc1, bc2) = check_adamw(config, step).expect("a valid config");
    let s = [
        (1.0 - config.lr * config.weight_decay) as f32,
        (1.0 - config.beta1) as f32,
        config.beta2 as f32,
        (1.0 - config.beta2) as f32,
        (config.lr / bc1) as f32,
        bc2.sqrt() as f32,
        config.eps as f32,
    ];
    s.iter().all(|v| v.is_finite()).then_some(s)
}

struct Reference {
    rt: std::sync::Arc<GpuRuntime>,
}

/// What the reference pipeline left behind.
struct RefOut {
    p: Vec<u32>,
    m: Vec<u32>,
    v: Vec<u32>,
    /// Either status word was set.
    refused: bool,
}

impl Reference {
    fn open() -> Self {
        let rt = GpuRuntime::new().expect("a Metal device");
        rt.add_metallib(&build_reference_metallib())
            .expect("load the reference metallib");
        Self { rt }
    }

    fn buffer_f32(&self, data: &[f32]) -> tessl::tensor::GpuBuffer {
        let buf = self
            .rt
            .alloc_buffer(data.len() * 4)
            .expect("allocate a buffer");
        {
            let mut map = buf.try_contents_f32().expect("map a buffer");
            map.copy_from_slice(data);
        }
        buf
    }

    fn read_bits(buf: &tessl::tensor::GpuBuffer) -> Vec<u32> {
        let map = buf.try_contents_f32().expect("map a buffer");
        map.iter().map(|x| x.to_bits()).collect()
    }

    fn scalar_bytes(s: &[f32; 7]) -> Vec<u8> {
        s.iter().flat_map(|x| x.to_ne_bytes()).collect()
    }

    /// Per-element classification with the old arithmetic.
    fn classify(&self, p: &[f32], g: &[f32], m: &[f32], v: &[f32], s: &[f32; 7]) -> Vec<u32> {
        let n = p.len();
        let (pb, gb, mb, vb) = (
            self.buffer_f32(p),
            self.buffer_f32(g),
            self.buffer_f32(m),
            self.buffer_f32(v),
        );
        let flags = self.rt.alloc_buffer(n * 4).expect("allocate the flags");
        flags.try_contents_u32().expect("map the flags").fill(0);
        let bytes = Self::scalar_bytes(s);
        let pipe = self.rt.pipeline("ref_adamw_classify").expect("pipeline");
        dispatch_1d(&self.rt, &pipe, n, |b| {
            set_gpu_buf_offset(b, &pb, 0, 0);
            set_gpu_buf_offset(b, &gb, 0, 1);
            set_gpu_buf_offset(b, &mb, 0, 2);
            set_gpu_buf_offset(b, &vb, 0, 3);
            set_gpu_buf_offset(b, &flags, 0, 4);
            set_u32(b, n as u32, 5);
            b.bind_bytes(&bytes, 6);
        })
        .expect("dispatch the classifier");
        self.rt.synchronize().expect("synchronize");
        let map = flags.try_contents_u32().expect("map the flags");
        map.to_vec()
    }

    /// The old `check` then `apply` pair.
    fn step(&self, p: &[f32], g: &[f32], m: &[f32], v: &[f32], s: &[f32; 7]) -> RefOut {
        let n = p.len();
        let (pb, gb, mb, vb) = (
            self.buffer_f32(p),
            self.buffer_f32(g),
            self.buffer_f32(m),
            self.buffer_f32(v),
        );
        let st = self.rt.alloc_buffer(16).expect("allocate the status words");
        st.try_contents_u32().expect("map the status").fill(0);
        let bytes = Self::scalar_bytes(s);
        for name in ["ref_adamw_check", "ref_adamw_apply"] {
            let pipe = self.rt.pipeline(name).expect("pipeline");
            dispatch_1d(&self.rt, &pipe, n, |b| {
                set_gpu_buf_offset(b, &pb, 0, 0);
                set_gpu_buf_offset(b, &gb, 0, 1);
                set_gpu_buf_offset(b, &mb, 0, 2);
                set_gpu_buf_offset(b, &vb, 0, 3);
                set_gpu_buf_offset(b, &st, 0, 4);
                set_u32(b, n as u32, 5);
                b.bind_bytes(&bytes, 6);
            })
            .expect("dispatch the reference kernel");
            // Check, then apply: no reliance on an in-buffer barrier.
            self.rt.synchronize().expect("synchronize");
        }
        let refused = {
            let words = st.try_contents_u32().expect("map the status");
            words[0] != 0 || words[1] != 0
        };
        RefOut {
            p: Self::read_bits(&pb),
            m: Self::read_bits(&mb),
            v: Self::read_bits(&vb),
            refused,
        }
    }
}

fn bits(data: &[f32]) -> Vec<u32> {
    data.iter().map(|x| x.to_bits()).collect()
}

/// `MetalBackend::adamw_step` on these inputs: the outputs' bits, and whether
/// the step was refused (a deferred fault at `sync`, or the host's refusal).
fn production(
    m: &ojas_metal::MetalBackend,
    inputs: [&[f32]; 4],
    config: AdamWConfig,
    step: u64,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, bool) {
    let n = inputs[0].len();
    let mut p = up(m, &host(inputs[0], &[n]));
    let g = up(m, &host(inputs[1], &[n]));
    let mut m1 = up(m, &host(inputs[2], &[n]));
    let mut m2 = up(m, &host(inputs[3], &[n]));
    let refused = match m.adamw_step(&mut p, &g, &mut m1, &mut m2, step, config) {
        Err(OjasError::NonFinite { op: "adamw_step" }) => true,
        Err(e) => panic!("adamw_step: {e:?}"),
        Ok(()) => match m.sync() {
            Ok(()) => false,
            Err(OjasError::NonFinite { op: "adamw_step" }) => true,
            Err(e) => panic!("sync: {e:?}"),
        },
    };
    (
        down(&p).iter().map(|x| x.to_bits()).collect(),
        down(&m1).iter().map(|x| x.to_bits()).collect(),
        down(&m2).iter().map(|x| x.to_bits()).collect(),
        refused,
    )
}

/// Edge values for p, g, m and v: signed zeros, subnormals, the smallest
/// normal, large and tiny magnitudes, f32::MAX (so `g * g` overflows) and
/// values one ulp from 1.
fn edge_values() -> Vec<f32> {
    let mut v = vec![
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        1.0e-40,
        -1.0e-40,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1.0e-30,
        1.0e-20,
        0.5,
        f32::from_bits(0x3f80_0001),
        f32::from_bits(0x3f7f_ffff),
        1.0,
        -1.0,
        65504.0,
        1.0e19,
        -1.0e19,
        f32::MAX,
        -f32::MAX,
    ];
    v.dedup_by_key(|x| x.to_bits());
    v
}

/// Every combination of the edge values over (p, g, m, v).
fn edge_product() -> [Vec<f32>; 4] {
    let e = edge_values();
    let k = e.len();
    let n = k * k * k * k;
    let mut out = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for i in 0..n {
        out[0].push(e[i % k]);
        out[1].push(e[(i / k) % k]);
        out[2].push(e[(i / (k * k)) % k]);
        out[3].push(e[(i / (k * k * k)) % k]);
    }
    out
}

/// Uniform random bit patterns: every exponent, NaNs and infinities included.
/// `v` has its sign cleared in the second variant so the second moment is
/// usually a legal one and more elements stay clean.
fn random_bits(n: usize, seed: u64, clear_v_sign: bool) -> [Vec<f32>; 4] {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 16) as u32
    };
    let mut out = [vec![], vec![], vec![], vec![]];
    for _ in 0..n {
        for (j, o) in out.iter_mut().enumerate() {
            let mut b = next();
            if j == 3 && clear_v_sign {
                b &= 0x7fff_ffff;
            }
            o.push(f32::from_bits(b));
        }
    }
    out
}

/// Random values over a wide exponent range, all finite and legal: the bulk
/// of an AdamW step in practice, with a mix of magnitudes.
fn random_wide(n: usize, seed: u64) -> [Vec<f32>; 4] {
    let mut s = seed.wrapping_mul(0xD6E8_FEB8_6659_FD93).wrapping_add(3);
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 16) as u32
    };
    let mut out = [vec![], vec![], vec![], vec![]];
    for _ in 0..n {
        for (j, o) in out.iter_mut().enumerate() {
            let mantissa = next() & 0x007f_ffff;
            // Exponents 60..=194 (about 1e-19 .. 1e20): most of `g * g` stays
            // finite, and the classifier drops the elements where it does not.
            let exp = 60 + next() % 135;
            let sign = if j == 3 { 0 } else { (next() & 1) << 31 };
            o.push(f32::from_bits(sign | (exp << 23) | mantissa));
        }
    }
    out
}

fn configs() -> Vec<(&'static str, AdamWConfig, u64)> {
    let base = AdamWConfig::nanolab(1e-3, 0.1);
    let with = |f: fn(&mut AdamWConfig)| {
        let mut c = base;
        f(&mut c);
        c
    };
    vec![
        ("nanolab step 0", base, 0),
        ("nanolab step 1", base, 1),
        ("nanolab step 9", base, 9),
        ("nanolab step 100000", base, 100_000),
        // lerp weight 0.7: torch's second lerp form.
        ("beta1 0.3", with(|c| c.beta1 = 0.3), 3),
        // lerp weight exactly 0.5: the boundary of the switch.
        ("beta1 0.5", with(|c| c.beta1 = 0.5), 3),
        ("beta1 0", with(|c| c.beta1 = 0.0), 3),
        ("beta2 0", with(|c| c.beta2 = 0.0), 3),
        ("beta2 0.9999999", with(|c| c.beta2 = 0.999_999_9), 0),
        ("no decay, lr 0", AdamWConfig::nanolab(0.0, 0.0), 5),
        ("lr 1e30", AdamWConfig::nanolab(1e30, 0.0), 2),
        // eps is positive in f64 and zero in f32: a zero denominator.
        ("eps 1e-50", with(|c| c.eps = 1e-50), 4),
        // eps subnormal in f32.
        ("eps 1e-40", with(|c| c.eps = 1e-40), 4),
        // `1 - lr * weight_decay` overflows: the host refuses.
        ("decay overflows", AdamWConfig::nanolab(1e300, 1e300), 1),
    ]
}

#[test]
fn the_adamw_kernels_match_the_retired_ones_bit_for_bit() {
    let reference = Reference::open();
    let m = metal();
    let product = edge_product();
    let random = random_bits(1 << 17, 11, false);
    let random_v_legal = random_bits(1 << 17, 12, true);
    let wide = random_wide(1 << 17, 13);
    let inputs: [(&str, &[Vec<f32>; 4]); 4] = [
        ("edge product", &product),
        ("random bits", &random),
        ("random bits, v >= 0", &random_v_legal),
        ("wide finite", &wide),
    ];
    let (mut clean_total, mut refused_total, mut host_refused) = (0usize, 0usize, 0usize);
    for (cname, config, step) in configs() {
        let Some(scalars) = reference_scalars(config, step) else {
            host_refused += 1;
            let one = [0.5f32; 4];
            let n = one.len();
            let mut p = up(&m, &host(&one, &[n]));
            let g = up(&m, &host(&one, &[n]));
            let mut m1 = up(&m, &host(&one, &[n]));
            let mut m2 = up(&m, &host(&one, &[n]));
            let r = m.adamw_step(&mut p, &g, &mut m1, &mut m2, step, config);
            assert!(
                matches!(r, Err(OjasError::NonFinite { op: "adamw_step" })),
                "{cname}: the host must refuse scalars that do not survive f32: {r:?}"
            );
            continue;
        };
        for (iname, set) in inputs {
            let what = format!("{cname} / {iname}");
            let [p, g, m0, v] = set;
            // The old arithmetic decides which elements are clean.
            let flags = reference.classify(p, g, m0, v, &scalars);
            let keep: Vec<usize> = (0..p.len()).filter(|&i| flags[i] == 0).collect();
            let pick = |x: &[f32]| keep.iter().map(|&i| x[i]).collect::<Vec<f32>>();
            let (cp, cg, cm, cv) = (pick(p), pick(g), pick(m0), pick(v));
            if !cp.is_empty() {
                clean_total += cp.len();
                let want = reference.step(&cp, &cg, &cm, &cv, &scalars);
                assert!(!want.refused, "{what}: the reference refused clean data");
                let (gp, gm, gv, refused) = production(&m, [&cp, &cg, &cm, &cv], config, step);
                assert!(
                    !refused,
                    "{what}: adamw_step refused elements the retired kernel accepted"
                );
                for (name, got, exp) in [
                    ("p", &gp, &want.p),
                    ("m", &gm, &want.m),
                    ("v", &gv, &want.v),
                ] {
                    if let Some(i) = (0..got.len()).find(|&i| got[i] != exp[i]) {
                        panic!(
                            "{what}: {name}[{i}] differs: new {:#010x} old {:#010x} \
                             (p {:#010x} g {:#010x} m {:#010x} v {:#010x}, {} clean elements)",
                            got[i],
                            exp[i],
                            cp[i].to_bits(),
                            cg[i].to_bits(),
                            cm[i].to_bits(),
                            cv[i].to_bits(),
                            got.len()
                        );
                    }
                }
            }
            // The whole array: both refuse it when any element is bad, and
            // neither writes a thing.
            if keep.len() != p.len() {
                refused_total += 1;
                let want = reference.step(p, g, m0, v, &scalars);
                assert!(want.refused, "{what}: the reference accepted a bad element");
                let (gp, gm, gv, refused) = production(&m, [p, g, m0, v], config, step);
                assert!(refused, "{what}: adamw_step accepted a bad element");
                assert_eq!(gp, bits(p), "{what}: p written despite the refusal");
                assert_eq!(gm, bits(m0), "{what}: m written despite the refusal");
                assert_eq!(gv, bits(v), "{what}: v written despite the refusal");
                assert_eq!(want.p, bits(p), "{what}: the reference wrote p");
                assert_eq!(want.m, bits(m0), "{what}: the reference wrote m");
                assert_eq!(want.v, bits(v), "{what}: the reference wrote v");
            }
        }
    }
    // The compare must have had something to compare and something to refuse.
    assert!(clean_total > 100_000, "only {clean_total} clean elements");
    assert!(refused_total > 10, "only {refused_total} refused arrays");
    assert!(host_refused >= 1, "no config exercised the host refusal");
}

/// `ojas_backend.metal` takes the update from tessl's header and keeps no copy
/// of it. Fails against the source before the swap.
#[test]
fn the_production_kernel_includes_tessls_adamw_math_and_keeps_no_copy() {
    let src = include_str!("../kernels/ojas_backend.metal");
    assert!(
        src.contains("#include \"qwen35_adamw_math.h\""),
        "ojas_backend.metal must include tessl's qwen35_adamw_math.h"
    );
    assert!(
        src.contains("qwen35_adamw_update("),
        "the check and apply kernels must call the header's function"
    );
    for retired in ["struct OjasAdamW", "ojas_adamw_elem"] {
        assert!(!src.contains(retired), "{retired} is a retired copy");
    }
}
