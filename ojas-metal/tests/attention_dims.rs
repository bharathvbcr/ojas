//! Cases `attention_general.rs` does not reach, on the general attention
//! kernel (`ojas_attn_gen_fwd`) against the CPU reference: head dim 16 (SAM
//! mask-decoder cross-attention), head dim 256 with a bias and key lengths
//! (memory attention), the SAM ViT window (9 x 576) and a global-length row,
//! and `Tq = 1` / `Tk = 1`. The last test pins the deferred-fault contract
//! (`docs/metal-deferred-faults.md` section 6): a non-finite operand passed
//! to `attention_forward` is reported under that name, not under
//! `attention_forward_saving`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{AttentionOpts, Backend, Tensor};
use ojas_metal::MetalBackend;

#[derive(Clone, Debug)]
struct Case {
    b: usize,
    h: usize,
    hkv: usize,
    tq: usize,
    tk: usize,
    d: usize,
    bias: Option<[usize; 3]>,
    lens: Option<Vec<u32>>,
}

impl Case {
    fn new(b: usize, h: usize, hkv: usize, tq: usize, tk: usize, d: usize) -> Self {
        Case {
            b,
            h,
            hkv,
            tq,
            tk,
            d,
            bias: None,
            lens: None,
        }
    }
    fn bias(mut self, s: [usize; 3]) -> Self {
        self.bias = Some(s);
        self
    }
    fn lens(mut self, l: &[u32]) -> Self {
        self.lens = Some(l.to_vec());
        self
    }
}

fn tol(tk: usize) -> f32 {
    4e-5 * (tk as f32).sqrt().max(1.0)
}

fn check(m: &MetalBackend, c: &Case, seed: u64) {
    let cpu = cpu();
    let q = rand(&[c.b, c.h, c.tq, c.d], seed, 1.0);
    let k = rand(&[c.b, c.hkv, c.tk, c.d], seed + 1, 1.0);
    let v = rand(&[c.b, c.hkv, c.tk, c.d], seed + 2, 1.0);
    let bias: Option<Tensor> = c
        .bias
        .map(|[b, h, t]| rand(&[b, h, t, c.tk], seed + 3, 2.0));
    let lens: Option<Tensor> = c.lens.as_ref().map(|l| host_u32(l, &[l.len()]));
    let host_opts = AttentionOpts {
        causal: false,
        window: None,
        bias: bias.as_ref(),
        key_lens: lens.as_ref(),
    };
    let tag = format!("{c:?}");
    let (want, want_lse) = ok(&tag, cpu.attention_forward_saving(&q, &k, &v, host_opts));
    let (dq, dk, dv) = (up(m, &q), up(m, &k), up(m, &v));
    let dbias = bias.as_ref().map(|t| up(m, t));
    let dlens = lens.as_ref().map(|t| up(m, t));
    let dev_opts = AttentionOpts {
        causal: false,
        window: None,
        bias: dbias.as_ref(),
        key_lens: dlens.as_ref(),
    };
    let (got, got_lse) = ok(&tag, m.attention_forward_saving(&dq, &dk, &dv, dev_opts));
    same_tensor(&format!("{tag} output"), &got, &want, tol(c.tk), 1e-4);
    same_tensor(&format!("{tag} lse"), &got_lse, &want_lse, tol(c.tk), 1e-4);
}

#[test]
fn head_dim_16_mask_decoder_cross_attention() {
    let m = metal();
    // Tokens over an image, then the image over tokens (both directions).
    check(&m, &Case::new(1, 8, 8, 7, 4096, 16), 1);
    check(&m, &Case::new(1, 8, 8, 4096, 7, 16).bias([1, 8, 4096]), 2);
    check(&m, &Case::new(2, 4, 2, 9, 33, 16).lens(&[33, 5]), 3);
}

#[test]
fn head_dim_256_memory_attention_with_bias_and_key_lengths() {
    let m = metal();
    check(
        &m,
        &Case::new(1, 1, 1, 40, 70, 256).bias([1, 1, 40]).lens(&[50]),
        4,
    );
    check(
        &m,
        &Case::new(2, 4, 2, 9, 33, 256)
            .bias([2, 4, 1])
            .lens(&[33, 5]),
        5,
    );
}

#[test]
fn sam_vit_window_and_global_row_lengths_at_head_dim_64() {
    let m = metal();
    // The windowed layers: 9 windows of 576 tokens (4 heads stand in for 16).
    check(&m, &Case::new(9, 4, 4, 576, 576, 64), 6);
    // A global-length key set with a single query row, and a short query block.
    check(&m, &Case::new(1, 2, 2, 1, 5184, 64).lens(&[5000]), 7);
    check(&m, &Case::new(1, 2, 2, 64, 5184, 64), 8);
}

#[test]
fn single_key_and_single_query_extremes() {
    let m = metal();
    check(&m, &Case::new(2, 3, 3, 4, 1, 64), 9);
    check(&m, &Case::new(2, 3, 3, 4, 1, 64).lens(&[1, 0]), 10);
    check(&m, &Case::new(1, 1, 1, 1, 1, 64), 11);
    check(&m, &Case::new(1, 2, 2, 1, 37, 16).bias([1, 2, 1]), 12);
}

/// The sync after a faulting `attention_forward` names `attention_forward`.
#[test]
fn a_deferred_fault_names_the_inference_form_that_was_called() {
    let m = metal();
    let c = Case::new(1, 2, 2, 3, 4, 64);
    let q = rand(&[1, 2, 3, 64], 1, 1.0);
    let mut kv = ok("keys", rand(&[1, 2, 4, 64], 2, 1.0).to_f32_vec());
    kv[0] = f32::NAN;
    let k = host(&kv, &[c.b, c.hkv, c.tk, c.d]);
    let v = rand(&[1, 2, 4, 64], 3, 1.0);
    let r = m.attention_forward(
        &up(&m, &q),
        &up(&m, &k),
        &up(&m, &v),
        AttentionOpts::default(),
    );
    deferred(&m, "nan key via attention_forward", r, "attention_forward");
}
