//! Shipped gemma-metal matmul: M=1 stays on the GEMV, M>1 with a supported
//! group runs the cooperative TensorOps kernel (Q4 row-major and Q8), and a
//! residual on that path is added rather than dropped. An unsupported group,
//! a non-row-major Q4 bank, classic Q4, and a short output return before any
//! encode.

use gemma_metal::kernels::{upload_quant_hot, GemmaGpu, HotGemvLayout};
use gemma_metal::quant::{
    bf16_bits_to_f32, f32_to_bf16_bits, quantize_affine_f32, QuantMatrix, QuantScheme,
};
use gemma_metal::tessl::runtime::{kernel_trace_enabled, traced_kernel_count};
use gemma_metal::tessl::GpuBuffer;

const ROWS: usize = 32;
const COLS: usize = 256;
const SENTINEL: f32 = -7.7e28;

fn in_traced_child(test_name: &str, body: impl FnOnce()) -> Option<String> {
    const CHILD: &str = "TESSL_TRACED_CHILD";
    if std::env::var_os(CHILD).is_some() {
        body();
        return None;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["--exact", "--test-threads=1", "--nocapture", test_name])
        .env(CHILD, "1")
        .env("TESSL_KERNEL_TRACE", "1")
        .output()
        .expect("spawn child test");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "child {test_name} failed:\n{stdout}\n{stderr}"
    );
    assert!(
        stdout.contains("1 passed"),
        "child ran no test (filter out of sync with {test_name}?):\n{stdout}"
    );
    Some(stdout)
}

fn round_bf16(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(x))
}

fn gamma(terms: usize) -> f64 {
    let tu = terms as f64 * 2f64.powi(-24);
    tu / (1.0 - tu)
}

fn weights(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((i as u32).wrapping_mul(seed) % 200) as f32 / 50.0 - 2.0)
        .collect()
}

fn quantize(scheme: QuantScheme) -> QuantMatrix {
    quantize_affine_f32(ROWS, COLS, &weights(ROWS, COLS, 13), scheme).unwrap()
}

fn nibbles(packed: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = packed[i / 2];
        out.push(if i % 2 == 0 { b & 0x0f } else { b >> 4 });
    }
    out
}

fn q4_reference(x: &[f32], w: &QuantMatrix, m: usize) -> (Vec<f64>, Vec<f64>) {
    let g = w.scheme.group_size().unwrap();
    let groups = COLS / g;
    let q = nibbles(&w.packed, ROWS * COLS);
    let mut want = vec![0.0f64; m * ROWS];
    let mut mag = vec![0.0f64; m * ROWS];
    for mi in 0..m {
        for ni in 0..ROWS {
            let (mut acc, mut s_mag) = (0.0, 0.0);
            for gi in 0..groups {
                let s = round_bf16(w.scales[ni * groups + gi]) as f64;
                let b = round_bf16(w.zeros[ni * groups + gi]) as f64;
                let (mut qx, mut aqx, mut xsum, mut ax) = (0.0, 0.0, 0.0, 0.0);
                for j in 0..g {
                    let xv = x[mi * COLS + gi * g + j] as f64;
                    let qv = q[ni * COLS + gi * g + j] as f64;
                    qx += qv * xv;
                    aqx += qv * xv.abs();
                    xsum += xv;
                    ax += xv.abs();
                }
                acc += s * qx + b * xsum;
                s_mag += s.abs() * aqx + b.abs() * ax;
            }
            want[mi * ROWS + ni] = acc;
            mag[mi * ROWS + ni] = s_mag;
        }
    }
    (want, mag)
}

fn q8_reference(x: &[f32], w: &QuantMatrix, m: usize) -> (Vec<f64>, Vec<f64>) {
    let g = w.scheme.group_size().unwrap();
    let groups = COLS / g;
    let mut want = vec![0.0f64; m * ROWS];
    let mut mag = vec![0.0f64; m * ROWS];
    for mi in 0..m {
        for ni in 0..ROWS {
            let (mut acc, mut s_mag) = (0.0, 0.0);
            for gi in 0..groups {
                let s = w.scales[ni * groups + gi] as f64;
                let z = w.zeros[ni * groups + gi] as f64;
                let (mut qx, mut aqx, mut xsum, mut ax) = (0.0, 0.0, 0.0, 0.0);
                for j in 0..g {
                    let xv = x[mi * COLS + gi * g + j] as f64;
                    let qv = w.packed[ni * COLS + gi * g + j] as i8 as f64;
                    qx += qv * xv;
                    aqx += qv.abs() * xv.abs();
                    xsum += xv;
                    ax += xv.abs();
                }
                acc += s * qx + (-s * z) * xsum;
                s_mag += s.abs() * aqx + (s * z).abs() * ax;
            }
            want[mi * ROWS + ni] = acc;
            mag[mi * ROWS + ni] = s_mag;
        }
    }
    (want, mag)
}

fn acts(gpu: &GemmaGpu, m: usize, seed: u32) -> (GpuBuffer, Vec<f32>) {
    let x: Vec<f32> = (0..m * COLS)
        .map(|i| round_bf16((((i as u32).wrapping_mul(seed)) % 128) as f32 / 64.0 - 1.0))
        .collect();
    let bits: Vec<u16> = x.iter().copied().map(f32_to_bf16_bits).collect();
    let mut bytes = Vec::with_capacity(bits.len() * 2);
    for b in bits {
        bytes.extend_from_slice(&b.to_le_bytes());
    }
    let buf = gpu.rt.alloc_buffer(bytes.len()).unwrap();
    buf.write_bytes(&bytes);
    (buf, x)
}

fn out(gpu: &GemmaGpu, m: usize) -> GpuBuffer {
    let y = gpu.rt.alloc_buffer(m * ROWS * 4).unwrap();
    // The pool rounds a small allocation up, so the buffer can be longer than
    // `m * ROWS` floats. The prefix is the view the kernel writes.
    y.write_f32_prefix(&vec![SENTINEL; m * ROWS]);
    y
}

fn assert_close(got: &[f32], want: &[f64], mag: &[f64], terms: usize, what: &str) {
    let bound = gamma(terms);
    let mut worst = 0.0f64;
    for i in 0..got.len() {
        assert!(got[i].is_finite() && got[i] != SENTINEL, "{what}: unwritten {i}");
        let err = (got[i] as f64 - want[i]).abs();
        let allow = bound * mag[i] + 1e-5;
        worst = worst.max(if allow == 0.0 { err } else { err / allow });
        assert!(err <= allow, "{what} [{i}]: |{err}| > {allow} (got {} want {})", got[i], want[i]);
    }
    println!("{what}: worst fraction {worst:.4}");
}

fn c(name: &str) -> u64 {
    traced_kernel_count(name)
}

#[test]
fn shipped_q4_q8_cooperative_matmul() {
    const NAME: &str = "shipped_q4_q8_cooperative_matmul";
    let Some(out) = in_traced_child(NAME, run) else {
        return;
    };
    println!("{out}");
}

fn run() {
    assert!(kernel_trace_enabled(), "TESSL_KERNEL_TRACE did not arm the trace");
    let gpu = GemmaGpu::new().expect("Metal device");
    assert!(gpu.rt.has_tensorops(), "this device has no TensorOps");

    let q4 = quantize(QuantScheme::Q4Mlx { group_size: 64 });
    let q4_bank = upload_quant_hot(&gpu, &q4).unwrap();
    assert_eq!(q4_bank.layout, HotGemvLayout::RowMajor);
    let packed_before = q4_bank.packed.contents_u8()[..q4.packed.len()].to_vec();

    // Unsupported group, non-row-major layout, classic Q4, short y: no encode.
    let mut bad_group = upload_quant_hot(&gpu, &q4).unwrap();
    bad_group.group_size = 256;
    let (x2, _) = acts(&gpu, 2, 3);
    let y2 = out(&gpu, 2);
    let before = c("gemm_q4_mlx_tensorops_g64");
    let err = bad_group.gemm_bf16_x(&gpu, &x2, &y2, 2).unwrap_err();
    assert!(err.to_string().contains("group_size 256"), "{err}");
    assert_eq!(c("gemm_q4_mlx_tensorops_g64"), before);
    assert_eq!(c("cast_bf16_to_f32"), 0);

    let mut bad_layout = upload_quant_hot(&gpu, &q4).unwrap();
    bad_layout.layout = HotGemvLayout::Interleaved4;
    let err = bad_layout.gemm_bf16_x(&gpu, &x2, &y2, 2).unwrap_err();
    assert!(err.to_string().contains("Interleaved4"), "{err}");
    assert_eq!(c("gemm_q4_mlx_tensorops_g64"), before);

    let classic = quantize(QuantScheme::Q4 { group_size: 32 });
    let classic_bank = upload_quant_hot(&gpu, &classic).unwrap();
    let err = classic_bank.gemm_bf16_x(&gpu, &x2, &y2, 2).unwrap_err();
    assert!(err.to_string().contains("classic Q4"), "{err}");
    assert_eq!(c("gemm_q4_mlx_tensorops_g32"), 0);
    assert_eq!(c("gemv_q4_mlx_simd"), 0);

    // A 4-byte request rounds up to the 256-byte pool minimum, which is exactly
    // one M=2 output. M=8 needs 1024 bytes, so the rounded buffer is still short.
    let short = gpu.rt.alloc_buffer(4).unwrap();
    let err = q4_bank.gemm_bf16_x(&gpu, &x2, &short, 8).unwrap_err();
    assert!(err.to_string().contains("short"), "{err}");
    assert_eq!(c("gemm_q4_mlx_tensorops_g64"), before);

    let q4_terms = COLS + 2 * (COLS / 64) + 64;
    for m in [2usize, 8, 16] {
        let (x, xv) = acts(&gpu, m, 5 + m as u32);
        let y = out(&gpu, m);
        let gemv = c("gemv_q4_mlx_simd");
        let simd = c("gemm_q4_mlx_simd");
        let coop = c("gemm_q4_mlx_tensorops_g64");
        q4_bank.gemm_bf16_x(&gpu, &x, &y, m as u32).unwrap();
        gpu.rt.synchronize().unwrap();
        let (want, mag) = q4_reference(&xv, &q4, m);
        assert_close(&y.read_f32()[..m * ROWS], &want, &mag, q4_terms, &format!("gemma Q4 M={m}"));
        assert_eq!(c("gemm_q4_mlx_tensorops_g64"), coop + 1);
        assert_eq!(c("gemv_q4_mlx_simd"), gemv);
        assert_eq!(c("gemm_q4_mlx_simd"), simd);
    }

    // M=1 stays on the GEMV. The residual add for M>1 keeps the residual.
    let (x, xv) = acts(&gpu, 1, 19);
    let y = out(&gpu, 1);
    let coop = c("gemm_q4_mlx_tensorops_g64");
    let gemv = c("gemv_q4_mlx_simd");
    q4_bank.gemm_bf16_x(&gpu, &x, &y, 1).unwrap();
    gpu.rt.synchronize().unwrap();
    let (want, mag) = q4_reference(&xv, &q4, 1);
    assert_close(&y.read_f32()[..ROWS], &want, &mag, q4_terms, "gemma Q4 M=1");
    assert_eq!(c("gemv_q4_mlx_simd"), gemv + 1);
    assert_eq!(c("gemm_q4_mlx_tensorops_g64"), coop);

    let (x, xv) = acts(&gpu, 4, 23);
    let resid = out(&gpu, 4);
    let resid0 = vec![0.25f32; 4 * ROWS];
    resid.write_f32_prefix(&resid0);
    let coop = c("gemm_q4_mlx_tensorops_g64");
    let fused = c("gemm_q4_mlx_simd_add");
    let add = c("add_inplace_f32");
    q4_bank.gemm_add_into_bf16_x(&gpu, &x, &resid, 4).unwrap();
    gpu.rt.synchronize().unwrap();
    let (want, mag) = q4_reference(&xv, &q4, 4);
    let got = resid.read_f32();
    for i in 0..4 * ROWS {
        let expect = want[i] + resid0[i] as f64;
        let err = (got[i] as f64 - expect).abs();
        let allow = gamma(q4_terms) * mag[i] + 1e-4;
        assert!(err <= allow, "Q4 residual [{i}]: {err} > {allow}");
    }
    assert_eq!(c("gemm_q4_mlx_tensorops_g64"), coop + 1);
    assert_eq!(c("gemm_q4_mlx_simd_add"), fused, "M>1 fused the old simd residual");
    assert_eq!(c("add_inplace_f32"), add + 1);

    let packed_after = q4_bank.packed.contents_u8()[..q4.packed.len()].to_vec();
    assert_eq!(packed_before, packed_after, "Q4 packed weights were rewritten on the host");

    // Q8. The shipped caller is gemm_bf16_x: M=1 casts on device and runs the
    // GEMV; M>1 runs the cooperative matmul. Group 256 is refused before the cast.
    let q8 = quantize(QuantScheme::q8_default());
    assert_eq!(q8.scheme.group_size(), Some(32));
    let q8_bank = upload_quant_hot(&gpu, &q8).unwrap();
    let q8_packed = q8_bank.packed.contents_u8()[..q8.packed.len()].to_vec();

    let mut wide_bank = upload_quant_hot(&gpu, &q8).unwrap();
    wide_bank.group_size = 256;
    let cast = c("cast_bf16_to_f32");
    let g128 = c("gemm_q8_tensorops_g128");
    let err = wide_bank.gemm_bf16_x(&gpu, &x2, &y2, 2).unwrap_err();
    assert!(err.to_string().contains("group_size 256"), "{err}");
    assert_eq!(c("cast_bf16_to_f32"), cast, "unsupported Q8 group encoded the cast");
    assert_eq!(c("gemm_q8_tensorops_g32"), 0);
    assert_eq!(g128, c("gemm_q8_tensorops_g128"));

    let q8_terms = COLS + 2 * (COLS / 32) + 32;
    let (x, xv) = acts(&gpu, 1, 31);
    let y = out(&gpu, 1);
    let gemv = c("gemv_q8");
    let coop = c("gemm_q8_tensorops_g32");
    let cast = c("cast_bf16_to_f32");
    q8_bank.gemm_bf16_x(&gpu, &x, &y, 1).unwrap();
    gpu.rt.synchronize().unwrap();
    let (want, mag) = q8_reference(&xv, &q8, 1);
    assert_close(&y.read_f32()[..ROWS], &want, &mag, q8_terms, "gemma Q8 M=1");
    assert_eq!(c("gemv_q8"), gemv + 1);
    assert_eq!(c("gemm_q8_tensorops_g32"), coop);
    assert!(c("cast_bf16_to_f32") > cast, "Q8 M=1 did not cast activations on device");

    for m in [2usize, 8] {
        let (x, xv) = acts(&gpu, m, 37 + m as u32);
        let y = out(&gpu, m);
        let gemv = c("gemv_q8");
        let coop = c("gemm_q8_tensorops_g32");
        let cast = c("cast_bf16_to_f32");
        q8_bank.gemm_bf16_x(&gpu, &x, &y, m as u32).unwrap();
        gpu.rt.synchronize().unwrap();
        let (want, mag) = q8_reference(&xv, &q8, m);
        assert_close(&y.read_f32()[..m * ROWS], &want, &mag, q8_terms, &format!("gemma Q8 M={m}"));
        assert_eq!(c("gemm_q8_tensorops_g32"), coop + 1);
        assert_eq!(c("gemv_q8"), gemv);
        assert!(c("cast_bf16_to_f32") > cast, "Q8 M={m} did not cast activations on device");
    }

    let q8_after = q8_bank.packed.contents_u8()[..q8.packed.len()].to_vec();
    assert_eq!(q8_packed, q8_after, "Q8 packed weights were rewritten on the host");
    println!("gemma router: Q4 and Q8 M>1 cooperative, M=1 GEMV, residual kept");
}
