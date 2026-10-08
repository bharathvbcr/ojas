// HEAD: the backward consumes the forward's output and lse; grouped-query
// heads are read natively. W=256 is timed beside the full prefix.
#[path = "/Users/bharath/Code/research/ojas/target-attn-ab/attn_ab_common.rs"]
mod ab;

use ojas_core::{Backend, Budget, OjasError, Tensor};

struct New;

impl<B: Backend> ab::Api<B> for New {
    const TREE: &'static str = "new";
    const WINDOWS: &'static [Option<usize>] = &[None, Some(256)];
    fn fwd(
        be: &B,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        w: Option<usize>,
    ) -> Result<Vec<Tensor>, OjasError> {
        let (y, lse) = be.causal_sdpa_forward(q, k, v, w)?;
        Ok(vec![y, lse])
    }
    fn bwd(
        be: &B,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        saved: &[Tensor],
        gy: &Tensor,
        w: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        be.causal_sdpa_backward(q, k, v, &saved[0], &saved[1], gy, w)
    }
}

fn main() {
    let budget = Budget::new(16 << 30);
    BACKEND_MAIN!(budget, New);
}
