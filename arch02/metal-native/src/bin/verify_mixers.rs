//! Verification suite for MSL mixer implementations (minGRU, VR blend, Mamba2).
//! Loads PyTorch `.npy` dumps from `golden/` and asserts output and gradient parity.

use tessl_arch02::gemm::{
    gemm_nt_accum_train, gemm_nt_train, gemm_tn_accum_train, gemm_train, select_backend,
};
use tessl_arch02::mixers::{
    mamba2_bwd, mamba2_fwd, mingru_bwd, mingru_fwd, mingru_vr_blend_bwd, mingru_vr_blend_fwd,
};
use tessl_arch02::npy::{read_npy, transpose_last2};
use tessl_arch02::runtime::GpuRuntime;
use tessl_arch02::tensor::Tensor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn load_tensor_f32(rt: &Arc<GpuRuntime>, path: &Path, expected_shape: &[usize]) -> Tensor {
    let raw = read_npy(path).unwrap();
    let data_f32 = raw.f32_slice().unwrap();

    let t = rt.alloc_tensor_f32(expected_shape).unwrap();
    t.buffer.write_f32(data_f32);
    t
}

/// Load a Python `[out, in]` linear weight as metal-native `[in, out]`.
fn load_matrix_python(rt: &Arc<GpuRuntime>, path: &Path, rust_shape: &[usize]) -> Tensor {
    let raw = read_npy(path).unwrap();
    let mut data = raw.f32_slice().unwrap().to_vec();
    let mut shape = raw.shape.clone();
    transpose_last2(&mut data, &mut shape).unwrap();
    assert_eq!(shape, rust_shape, "weight shape mismatch for {}", path.display());
    let t = rt.alloc_tensor_f32(rust_shape).unwrap();
    t.buffer.write_f32(&data);
    t
}

fn reshape_view(t: &Tensor, shape: &[usize]) -> Tensor {
    assert_eq!(t.byte_offset(), 0, "reshape_view requires zero-offset tensor");
    t.view(shape, 0)
}

fn assert_parity(name: &str, actual: &Tensor, expected: &Tensor, atol: f32) {
    let a = actual.buffer.read_f32();
    let b = expected.buffer.read_f32();
    assert_eq!(a.len(), b.len(), "length mismatch for {name}");

    let mut max_err = 0.0f32;
    let mut max_idx = 0;

    for i in 0..a.len() {
        let err = (a[i] - b[i]).abs();
        if err > max_err {
            max_err = err;
            max_idx = i;
        }
    }

    println!("{}: max err = {:.2e} (at {})", name, max_err, max_idx);
    if max_err > atol {
        panic!("{} parity failed! err = {} > {}", name, max_err, atol);
    }
}

/// Compare metal-native `[in, out]` against golden Python `[out, in]`.
fn assert_matrix_parity_python(name: &str, actual_in_out: &Tensor, expected_path: &Path, atol: f32) {
    let raw = read_npy(expected_path).unwrap();
    let expected = raw.f32_slice().unwrap();
    let in_dim = actual_in_out.shape()[0];
    let out_dim = actual_in_out.shape()[1];
    assert_eq!(
        raw.shape,
        vec![out_dim, in_dim],
        "golden shape mismatch for {name}"
    );
    let actual = actual_in_out.buffer.read_f32();
    let mut max_err = 0.0f32;
    let mut max_idx = 0usize;
    for o in 0..out_dim {
        for i in 0..in_dim {
            let exp = expected[o * in_dim + i];
            let got = actual[i * out_dim + o];
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
                max_idx = o * in_dim + i;
            }
        }
    }
    println!("{}: max err = {:.2e} (at {})", name, max_err, max_idx);
    if max_err > atol {
        panic!("{} parity failed! err = {} > {}", name, max_err, atol);
    }
}

fn verify_mingru(rt: &Arc<GpuRuntime>) {
    println!("--- Verifying minGRU ---");
    let golden_dir = PathBuf::from("golden/mingru");

    let z_in = load_tensor_f32(rt, &golden_dir.join("Z_in.npy"), &[2, 32, 128]);
    let h_in = load_tensor_f32(rt, &golden_dir.join("H_in.npy"), &[2, 32, 128]);
    let expected_h_out = load_tensor_f32(rt, &golden_dir.join("h_out.npy"), &[2, 32, 128]);

    let h_out = mingru_fwd(rt, &z_in, &h_in).unwrap();
    rt.synchronize().unwrap();
    assert_parity("mingru_fwd (h_out)", &h_out, &expected_h_out, 1e-4);

    let grad_h_out = load_tensor_f32(rt, &golden_dir.join("grad_h_out.npy"), &[2, 32, 128]);
    let expected_grad_z = load_tensor_f32(rt, &golden_dir.join("grad_Z_in.npy"), &[2, 32, 128]);
    let expected_grad_h_in = load_tensor_f32(rt, &golden_dir.join("grad_H_in.npy"), &[2, 32, 128]);

    let (grad_z, grad_h_in_actual) =
        mingru_bwd(rt, &z_in, &h_in, &expected_h_out, &grad_h_out).unwrap();
    rt.synchronize().unwrap();
    assert_parity("mingru_bwd (grad_Z_in)", &grad_z, &expected_grad_z, 1e-4);
    assert_parity("mingru_bwd (grad_H_in)", &grad_h_in_actual, &expected_grad_h_in, 1e-4);
}

fn verify_mingru_vr(rt: &Arc<GpuRuntime>) {
    println!("\n--- Verifying minGRU VR blend ---");
    let golden_dir = PathBuf::from("golden/mingru_vr");

    let h_raw = load_tensor_f32(rt, &golden_dir.join("h_raw.npy"), &[2, 32, 128]);
    let v0_up = load_tensor_f32(rt, &golden_dir.join("v0_up.npy"), &[2, 32, 128]);
    let vr_lambda = load_tensor_f32(rt, &golden_dir.join("vr_lambda.npy"), &[2]);
    let expected_h_pre = load_tensor_f32(rt, &golden_dir.join("h_pre.npy"), &[2, 32, 128]);

    let h_pre = rt.alloc_tensor_f32(&[2, 32, 128]).unwrap();
    mingru_vr_blend_fwd(rt, &h_raw, &v0_up, &vr_lambda, &h_pre, true).unwrap();
    rt.synchronize().unwrap();
    assert_parity("mingru_vr_blend_fwd (h_pre)", &h_pre, &expected_h_pre, 1e-4);

    let grad_h_pre = load_tensor_f32(rt, &golden_dir.join("grad_h_pre.npy"), &[2, 32, 128]);
    let expected_grad_h_raw =
        load_tensor_f32(rt, &golden_dir.join("grad_h_raw.npy"), &[2, 32, 128]);
    let expected_grad_v0_up =
        load_tensor_f32(rt, &golden_dir.join("grad_v0_up.npy"), &[2, 32, 128]);
    let expected_grad_vr_lambda =
        load_tensor_f32(rt, &golden_dir.join("grad_vr_lambda.npy"), &[2]);

    let d_h_raw = rt.alloc_tensor_f32(&[2, 32, 128]).unwrap();
    let d_v0_up = rt.alloc_tensor_f32(&[2, 32, 128]).unwrap();
    let d_vr_lambda = rt.alloc_tensor_f32(&[2]).unwrap();
    d_h_raw.buffer.zero();
    d_v0_up.buffer.zero();
    d_vr_lambda.buffer.zero();

    mingru_vr_blend_bwd(
        rt,
        &grad_h_pre,
        &h_raw,
        &v0_up,
        &vr_lambda,
        &d_h_raw,
        &d_v0_up,
        &d_vr_lambda,
        true,
    )
    .unwrap();
    rt.synchronize().unwrap();
    assert_parity("mingru_vr_blend_bwd (grad_h_raw)", &d_h_raw, &expected_grad_h_raw, 1e-4);
    assert_parity("mingru_vr_blend_bwd (grad_v0_up)", &d_v0_up, &expected_grad_v0_up, 1e-4);
    assert_parity(
        "mingru_vr_blend_bwd (grad_vr_lambda)",
        &d_vr_lambda,
        &expected_grad_vr_lambda,
        1e-3,
    );
}

fn verify_mingru_vr_layer(rt: &Arc<GpuRuntime>) {
    println!("\n--- Verifying minGRU+VR full layer ---");
    let golden_dir = PathBuf::from("golden/mingru_vr_layer");
    let backend = select_backend(rt);

    let b = 2;
    let t = 16;
    let c = 64;
    let kv = 32;
    let hid = 128;
    let hkv = 2;
    let d = 16;
    let bt = b * t;

    let x = load_tensor_f32(rt, &golden_dir.join("x.npy"), &[b, t, c]);
    let v0 = load_tensor_f32(rt, &golden_dir.join("v0.npy"), &[b, t, hkv, d]);
    let expected_out = load_tensor_f32(rt, &golden_dir.join("out.npy"), &[b, t, c]);
    let expected_raw_v = load_tensor_f32(rt, &golden_dir.join("raw_v.npy"), &[b, t, hkv, d]);

    let z_w = load_matrix_python(rt, &golden_dir.join("to_z_weight.npy"), &[c, hid]);
    let h_w = load_matrix_python(rt, &golden_dir.join("to_h_weight.npy"), &[c, hid]);
    let out_w = load_matrix_python(rt, &golden_dir.join("out_weight.npy"), &[hid, c]);
    let v_w = load_matrix_python(rt, &golden_dir.join("v_proj_weight.npy"), &[c, kv]);
    let v0_up_w = load_matrix_python(rt, &golden_dir.join("v0_up_weight.npy"), &[kv, hid]);
    let vr_lambda = load_tensor_f32(rt, &golden_dir.join("vr_lambda.npy"), &[2]);

    let x_flat = reshape_view(&x, &[bt, c]);
    let z_raw = rt.alloc_tensor_f32(&[bt, hid]).unwrap();
    let h_raw = rt.alloc_tensor_f32(&[bt, hid]).unwrap();
    gemm_train(&x_flat, &z_w, &z_raw, backend).unwrap();
    gemm_train(&x_flat, &h_w, &h_raw, backend).unwrap();

    let v_flat = rt.alloc_tensor_f32(&[bt, kv]).unwrap();
    gemm_train(&x_flat, &v_w, &v_flat, backend).unwrap();
    let raw_v = reshape_view(&v_flat, &[b, t, hkv, d]);

    let v0_flat = reshape_view(&v0, &[bt, kv]);
    let v0_up = rt.alloc_tensor_f32(&[bt, hid]).unwrap();
    gemm_train(&v0_flat, &v0_up_w, &v0_up, backend).unwrap();

    let h_pre = rt.alloc_tensor_f32(&[bt, hid]).unwrap();
    mingru_vr_blend_fwd(rt, &h_raw, &v0_up, &vr_lambda, &h_pre, true).unwrap();

    let z_3d = reshape_view(&z_raw, &[b, t, hid]);
    let h_3d = reshape_view(&h_pre, &[b, t, hid]);
    let h_out = mingru_fwd(rt, &z_3d, &h_3d).unwrap();

    let h_flat = reshape_view(&h_out, &[bt, hid]);
    let out_flat = rt.alloc_tensor_f32(&[bt, c]).unwrap();
    gemm_train(&h_flat, &out_w, &out_flat, backend).unwrap();
    let out = reshape_view(&out_flat, &[b, t, c]);

    rt.synchronize().unwrap();
    assert_parity("mingru_vr_layer_fwd (out)", &out, &expected_out, 1e-4);
    assert_parity("mingru_vr_layer_fwd (raw_v)", &raw_v, &expected_raw_v, 1e-4);

    let grad_out = load_tensor_f32(rt, &golden_dir.join("grad_out.npy"), &[b, t, c]);
    let d_out_flat = reshape_view(&grad_out, &[bt, c]);

    let d_h_flat = rt.alloc_tensor_f32(&[bt, hid]).unwrap();
    gemm_nt_train(&d_out_flat, &out_w, &d_h_flat, backend).unwrap();

    let dw_out = rt.alloc_tensor_f32(&[hid, c]).unwrap();
    dw_out.buffer.zero();
    gemm_tn_accum_train(&h_flat, &d_out_flat, &dw_out, backend).unwrap();

    let d_h_3d = reshape_view(&d_h_flat, &[b, t, hid]);
    let (d_z, d_h_pre) = mingru_bwd(rt, &z_3d, &h_3d, &h_out, &d_h_3d).unwrap();

    let d_h_pre_flat = reshape_view(&d_h_pre, &[bt, hid]);
    let h_raw_flat = reshape_view(&h_raw, &[bt, hid]);
    let d_h_raw_flat = rt.alloc_tensor_f32(&[bt, hid]).unwrap();
    d_h_raw_flat.buffer.zero();
    let d_v0_up = rt.alloc_tensor_f32(&[bt, hid]).unwrap();
    d_v0_up.buffer.zero();
    let d_vr_lambda = rt.alloc_tensor_f32(&[2]).unwrap();
    d_vr_lambda.buffer.zero();
    mingru_vr_blend_bwd(
        rt,
        &d_h_pre_flat,
        &h_raw_flat,
        &v0_up,
        &vr_lambda,
        &d_h_raw_flat,
        &d_v0_up,
        &d_vr_lambda,
        true,
    )
    .unwrap();

    let dv0_flat = rt.alloc_tensor_f32(&[bt, kv]).unwrap();
    dv0_flat.buffer.zero();
    gemm_nt_accum_train(&d_v0_up, &v0_up_w, &dv0_flat, backend).unwrap();

    let dw_v0_up = rt.alloc_tensor_f32(&[kv, hid]).unwrap();
    dw_v0_up.buffer.zero();
    gemm_tn_accum_train(&v0_flat, &d_v0_up, &dw_v0_up, backend).unwrap();

    let d_z_flat = reshape_view(&d_z, &[bt, hid]);
    let d_x_flat = rt.alloc_tensor_f32(&[bt, c]).unwrap();
    d_x_flat.buffer.zero();
    gemm_nt_accum_train(&d_z_flat, &z_w, &d_x_flat, backend).unwrap();
    gemm_nt_accum_train(&d_h_raw_flat, &h_w, &d_x_flat, backend).unwrap();

    let dw_z = rt.alloc_tensor_f32(&[c, hid]).unwrap();
    dw_z.buffer.zero();
    gemm_tn_accum_train(&x_flat, &d_z_flat, &dw_z, backend).unwrap();
    let dw_h = rt.alloc_tensor_f32(&[c, hid]).unwrap();
    dw_h.buffer.zero();
    gemm_tn_accum_train(&x_flat, &d_h_raw_flat, &dw_h, backend).unwrap();

    rt.synchronize().unwrap();

    let grad_x = reshape_view(&d_x_flat, &[b, t, c]);
    assert_parity("mingru_vr_layer_bwd (grad_x)", &grad_x, &load_tensor_f32(rt, &golden_dir.join("grad_x.npy"), &[b, t, c]), 1e-4);
    assert_matrix_parity_python(
        "mingru_vr_layer_bwd (grad_to_z)",
        &dw_z,
        &golden_dir.join("grad_to_z.npy"),
        1e-4,
    );
    assert_matrix_parity_python(
        "mingru_vr_layer_bwd (grad_to_h)",
        &dw_h,
        &golden_dir.join("grad_to_h.npy"),
        1e-4,
    );
    assert_matrix_parity_python(
        "mingru_vr_layer_bwd (grad_out_w)",
        &dw_out,
        &golden_dir.join("grad_out_w.npy"),
        1e-4,
    );
    assert_matrix_parity_python(
        "mingru_vr_layer_bwd (grad_v0_up)",
        &dw_v0_up,
        &golden_dir.join("grad_v0_up.npy"),
        1e-4,
    );
    assert_parity(
        "mingru_vr_layer_bwd (grad_vr_lambda)",
        &d_vr_lambda,
        &load_tensor_f32(rt, &golden_dir.join("grad_vr_lambda.npy"), &[2]),
        1e-3,
    );
}

fn verify_mamba2(rt: &Arc<GpuRuntime>) {
    println!("\n--- Verifying Mamba-2 ---");
    let golden_dir = PathBuf::from("golden/mamba2");

    let b = 2;
    let l = 32;
    let h = 4;
    let p = 16;
    let n = 16;

    let x_scaled = load_tensor_f32(rt, &golden_dir.join("x_scaled.npy"), &[b, l, h, p]);
    let b_h = load_tensor_f32(rt, &golden_dir.join("B_h.npy"), &[b, l, n]);
    let c_h = load_tensor_f32(rt, &golden_dir.join("C_h.npy"), &[b, l, n]);
    let log_da = load_tensor_f32(rt, &golden_dir.join("log_dA.npy"), &[b, l, h]);
    let expected_y = load_tensor_f32(rt, &golden_dir.join("y.npy"), &[b, l, h, p]);

    let fwd = mamba2_fwd(rt, &x_scaled, &b_h, &c_h, &log_da).unwrap();
    rt.synchronize().unwrap();
    assert_parity("mamba2_fwd (y)", &fwd.y, &expected_y, 1e-4);

    let grad_y = load_tensor_f32(rt, &golden_dir.join("grad_y.npy"), &[b, l, h, p]);

    let expected_grad_x_scaled =
        load_tensor_f32(rt, &golden_dir.join("grad_x_scaled.npy"), &[b, l, h, p]);
    let expected_grad_b_h = load_tensor_f32(rt, &golden_dir.join("grad_B_h.npy"), &[b, l, n]);
    let expected_grad_c_h = load_tensor_f32(rt, &golden_dir.join("grad_C_h.npy"), &[b, l, n]);
    let expected_grad_log_da =
        load_tensor_f32(rt, &golden_dir.join("grad_log_dA.npy"), &[b, l, h]);

    let bwd = mamba2_bwd(rt, &x_scaled, &b_h, &c_h, &log_da, &fwd.h_states, &grad_y).unwrap();
    rt.synchronize().unwrap();

    assert_parity(
        "mamba2_bwd (grad_x_scaled)",
        &bwd.grad_x_scaled,
        &expected_grad_x_scaled,
        1e-4,
    );
    assert_parity("mamba2_bwd (grad_B_h)", &bwd.grad_b_h, &expected_grad_b_h, 1e-3);
    assert_parity("mamba2_bwd (grad_C_h)", &bwd.grad_c_h, &expected_grad_c_h, 1e-3);
    assert_parity(
        "mamba2_bwd (grad_log_dA)",
        &bwd.grad_log_da,
        &expected_grad_log_da,
        1e-3,
    );
}

fn main() {
    let rt = tessl_arch02::gpu_runtime().unwrap();
    verify_mingru(&rt);
    verify_mingru_vr(&rt);
    verify_mingru_vr_layer(&rt);
    verify_mamba2(&rt);
    println!("\nAll mixer kernels passed verification!");
}
