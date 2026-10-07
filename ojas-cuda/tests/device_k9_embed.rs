//! K9: the token embedding, a gather copied as bits, and its backward, each
//! id's rows summed in position order and added once onto the tied head's
//! gradient (no atomics).
//!
//! - **Host** (run everywhere): the host mirror (`embed::*_mirror`) against
//!   L-cuda-oracle's float64 reference on the torch golden's f32-rounded
//!   inputs (ids repeated and out of order), and at the 2B's hidden size over
//!   600 rows onto a prior gradient. (A NaN payload's bits through the gather
//!   are rung a's `k9_checks` and `embed`'s unit test.)
//! - **Device** (`--features cuda`, `#[ignore]`: an sm_90 GPU; on the Mac they
//!   compile and are NOT RUN): the kernels against the same reference, bit for
//!   bit against the mirror and a repeat run, and rung a's checks.
//!
//! Bounds, written before any run: the gather is **bit-exact** (each output
//! is the f32 table's bits); the backward within `1e-4 * max|ref|`
//! (`tessl/tests/qwen35_bwd.rs:43-66`, the bound of tessl's `run_embed` at
//! `:1091-1132`), and every table row no id reads keeps its prior bits.

mod device_small_common;
mod reference;

use device_small_common::{
    assert_bits, f32s, golden, wide, Goldens, BWD_PEAK_REL, BWD_PEAK_SOURCE,
};
use ojas_cuda::check::Check;
use ojas_cuda::embed::{embed_rows_bwd_mirror, embed_rows_fwd_mirror, EmbedPlan};
use ojas_cuda::inputs::{splitmix_bits, splitmix_f32};
use ojas_cuda::small_common::peak_check;
use reference::embed::{embed_rows_bwd_f64, embed_rows_fwd_f64};

const GOLDENS: Goldens = Goldens(&[
    golden!("embed_ids"),
    golden!("embed_table"),
    golden!("embed_dy"),
]);

/// One embedding case: f32 inputs and the reference on them.
struct EmbedCase {
    plan: EmbedPlan,
    ids: Vec<u32>,
    table: Vec<f32>,
    dh: Vec<f32>,
    prior: Vec<f32>,
    y_ref: Vec<f64>,
    dw_ref: Vec<f64>,
}

impl EmbedCase {
    fn new(
        (vocab, hidden): (usize, usize),
        ids: Vec<usize>,
        (table, dh, prior): (Vec<f32>, Vec<f32>, Vec<f32>),
    ) -> Self {
        let plan = EmbedPlan::new(ids.len() as u64, vocab as u64, hidden as u64).expect("plan");
        let y_ref = embed_rows_fwd_f64(&wide(&table), &ids, vocab, hidden);
        let dw_ref = embed_rows_bwd_f64(&ids, &wide(&dh), vocab, hidden, Some(&wide(&prior)));
        EmbedCase {
            plan,
            ids: ids
                .iter()
                .map(|&i| u32::try_from(i).expect("id fits u32"))
                .collect(),
            table,
            dh,
            prior,
            y_ref,
            dw_ref,
        }
    }

    fn golden() -> Self {
        let ids = GOLDENS.indices("embed_ids");
        let (ts, table) = GOLDENS.shaped("embed_table");
        let (vocab, hidden) = (ts[0], ts[1]);
        let prior = splitmix_f32(0x9c1, vocab * hidden, 0.25);
        EmbedCase::new(
            (vocab, hidden),
            ids,
            (f32s(&table), f32s(&GOLDENS.f64s("embed_dy")), prior),
        )
    }

    /// 600 rows at the 2B's hidden size, ids repeated and out of order and at
    /// both ends of a 1000-row table.
    fn two_b() -> Self {
        let (n, vocab, hidden) = (600, 1000, 2048);
        let mut ids: Vec<usize> = splitmix_bits(0x9d1, n)
            .iter()
            .map(|b| *b as usize % vocab)
            .collect();
        ids[0] = vocab - 1;
        ids[1] = 0;
        EmbedCase::new(
            (vocab, hidden),
            ids,
            (
                splitmix_f32(0x9d2, vocab * hidden, 1.0),
                splitmix_f32(0x9d3, n * hidden, 1.0),
                splitmix_f32(0x9d4, vocab * hidden, 0.01),
            ),
        )
    }

    /// The gather bit-exact, rows no id reads unchanged, the sum within bound.
    fn judge(&self, label: &str, (y, dw): (&[f32], &[f32])) -> Vec<Check> {
        let want_y = f32s(&self.y_ref);
        assert_bits(&format!("{label} y"), y, &want_y);
        let hidden = self.plan.hidden as usize;
        for v in 0..self.plan.vocab as usize {
            if !self.ids.contains(&(v as u32)) {
                let row = v * hidden..(v + 1) * hidden;
                assert_bits(
                    &format!("{label} untouched dW row {v}"),
                    &dw[row.clone()],
                    &self.prior[row],
                );
            }
        }
        vec![peak_check(
            &format!("{label} dW"),
            dw,
            &self.dw_ref,
            BWD_PEAK_REL,
            BWD_PEAK_SOURCE,
        )]
    }

    fn mirror(&self) -> (Vec<f32>, Vec<f32>) {
        let y = embed_rows_fwd_mirror(&self.plan, &self.ids, &self.table).expect("fwd mirror");
        let mut dw = self.prior.clone();
        embed_rows_bwd_mirror(&self.plan, &self.ids, &self.dh, &mut dw).expect("bwd mirror");
        (y, dw)
    }
}

#[test]
fn the_embedded_goldens_are_l_cuda_oracles_pinned_bytes() {
    assert_eq!(GOLDENS.verify(), 3);
}

#[test]
fn the_mirror_matches_the_reference_on_the_torch_golden() {
    let case = EmbedCase::golden();
    let (y, dw) = case.mirror();
    device_small_common::assert_pass(&case.judge("mirror embed golden", (&y, &dw)));
}

#[test]
fn the_mirror_matches_the_reference_at_the_2b_hidden_size() {
    let case = EmbedCase::two_b();
    let (y, dw) = case.mirror();
    device_small_common::assert_pass(&case.judge("mirror embed 600x1000x2048", (&y, &dw)));
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use device_small_common::{assert_pass, runtime};
    use ojas_cuda::embed_cuda::{embed_rows, embed_rows_bwd};
    use ojas_cuda::runtime::CudaRuntime;
    use ojas_cuda::small_smoke::k9_checks;

    fn run(rt: &CudaRuntime, c: &EmbedCase) -> (Vec<f32>, Vec<f32>) {
        let n = c.ids.len() * c.plan.hidden as usize;
        let table = rt.upload(&c.table, "table").expect("table");
        let mut y = rt.upload(&vec![f32::NAN; n], "y").expect("y");
        embed_rows(rt, &c.plan, &c.ids, &table, &mut y).expect("embed fwd");
        let dh = rt.upload(&c.dh, "dh").expect("dh");
        let mut dw = rt.upload(&c.prior, "dW").expect("dW");
        embed_rows_bwd(rt, &c.plan, &c.ids, &dh, &mut dw).expect("embed bwd");
        (rt.download(&y).expect("y"), rt.download(&dw).expect("dW"))
    }

    fn check(rt: &CudaRuntime, label: &str, c: &EmbedCase) -> Vec<Check> {
        let got = run(rt, c);
        let again = run(rt, c);
        let want = c.mirror();
        assert_bits(&format!("{label} y vs mirror"), &got.0, &want.0);
        assert_bits(&format!("{label} dW vs mirror"), &got.1, &want.1);
        assert_bits(&format!("{label} y repeat"), &again.0, &got.0);
        assert_bits(&format!("{label} dW repeat"), &again.1, &got.1);
        c.judge(label, (&got.0, &got.1))
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn the_device_embedding_matches_the_reference_and_the_mirror_bitwise() {
        let rt = runtime();
        let mut checks = check(&rt, "device embed golden", &EmbedCase::golden());
        checks.extend(check(
            &rt,
            "device embed 600x1000x2048",
            &EmbedCase::two_b(),
        ));
        assert_pass(&checks);
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn rung_a_k9_checks_pass() {
        assert_pass(&k9_checks(&runtime()));
    }
}
