//! Why `Probe`'s forwarding is proved behaviourally
//! (`probe_forwarding::probe_reaches_the_wrapped_backend_once_per_op`) and
//! not by scanning `common/mod.rs`: a method compiled out with `#[cfg]`
//! leaves a wrapper that builds and silently runs the trait default.

use ojas_core::{forward_backend_methods, Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;

/// A compiled demonstration of the same class: an excluded name whose
/// replacement is `#[cfg]`-erased. The wrapper compiles, and runs the trait
/// default (Unsupported) where the wrapped CPU answers.
#[test]
fn a_cfg_erased_override_silently_runs_the_trait_default() {
    macro_rules! all_but_sigmoid {
        (sigmoid_forward, { $($item:tt)* }) => {};
        ($name:ident, { $($item:tt)* }) => { $($item)* };
    }
    struct Erased<B>(B);
    impl<B: Backend> Backend for Erased<B> {
        ojas_core::with_backend_methods!(
            forward_backend_methods,
            { self, self.0, none, all_but_sigmoid }
        );
        #[cfg(any())]
        fn sigmoid_forward(&self, x: &Tensor) -> Result<Tensor, ojas_core::OjasError> {
            self.0.sigmoid_forward(x)
        }
    }
    let cpu = CpuBackend::new(Budget::new(1 << 20));
    let x = Tensor::from_f32(&[0.0, 1.0], &[2], cpu.budget()).unwrap();
    assert!(cpu.sigmoid_forward(&x).is_ok());
    assert!(
        Erased(&cpu).sigmoid_forward(&x).is_err(),
        "the erased override unexpectedly forwarded"
    );
}
