// c74f3ba: the backward forms its own row statistics; K/V are expanded for
// grouped-query heads.
#[path = "/Users/bharath/Code/research/ojas/target-attn-ab/attn_ab_common.rs"]
mod ab;

use ojas_core::{Backend, Budget, OjasError, Tensor};

struct Old;

impl<B: Backend> ab::Api<B> for Old {
    const TREE: &'static str = "old";
    const WINDOWS: &'static [Option<usize>] = &[None];
    fn fwd(
        be: &B,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _w: Option<usize>,
    ) -> Result<Vec<Tensor>, OjasError> {
        Ok(vec![be.causal_sdpa_forward(q, k, v)?])
    }
    fn bwd(
        be: &B,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _saved: &[Tensor],
        gy: &Tensor,
        _w: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        be.causal_sdpa_backward(q, k, v, gy)
    }
}

fn main() {
    let budget = Budget::new(16 << 30);
    BACKEND_MAIN!(budget, Old);
}
