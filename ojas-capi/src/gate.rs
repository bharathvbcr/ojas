//! Cancellation inside a model op.
//!
//! `ojas_model::Trainer::step` and `ojas_infer::DeviceDecoder::generate` take
//! no cancel hook. Each session's backend is wrapped in [`Gated`], which polls
//! the call's cancel check before every op that can fail without changing
//! what the trainer owns, and refuses the op once the check fails.
//!
//! Which ops poll is the commit contract. `Trainer::step`
//! (`ojas-model/src/trainer.rs`, `run`) computes gradients and clips them
//! (phases 2-4) before it touches a parameter; an error there leaves
//! parameters, moments, step and cursor unchanged. Its optimizer phase
//! (`muon_ns5_step`, `adamw_step`) and its finish (`sync`, then one
//! `download`) mark the trainer poisoned on any error. So those four never
//! poll: a cancel always lands in phases 2-4, and a cancelled step commits
//! nothing. Every other op polls.
//!
//! The check is armed for one C-ABI call by [`CancelSlot::arm`] and disarmed
//! when the returned [`Armed`] drops, so a later call never sees an earlier
//! call's context. A failed poll is recorded in the slot; the call reports
//! that recorded cancel ([`Armed::tripped`]), never a reading of error text.

use std::sync::{Arc, Mutex};

use ojas_core::{forward_backend_keep, forward_backend_methods, Backend, OjasError};

/// The per-call cancel check: `Err` carries the cancel's own message
/// (`"cancelled: Explicit"`, which gusset maps to `context.Canceled`).
pub type Check = Box<dyn FnMut() -> Result<(), String> + Send>;

/// A test fault: the op it fires on and the error it returns.
#[cfg(test)]
type Fault = (&'static str, fn() -> OjasError);

#[derive(Default)]
struct SlotState {
    check: Option<Check>,
    tripped: Option<String>,
    #[cfg(test)]
    fault: Option<Fault>,
}

/// One per session, shared by every clone of its [`Gated`] backend.
#[derive(Default)]
pub struct CancelSlot {
    state: Mutex<SlotState>,
}

impl CancelSlot {
    fn lock(&self) -> std::sync::MutexGuard<'_, SlotState> {
        // The check is caller code; a panic in it unwinds out of the job and
        // gusset poisons the handle. The slot's own fields stay consistent.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Install `check` for one call. The call ends when the guard drops.
    pub fn arm(self: &Arc<Self>, check: Check) -> Armed {
        let mut state = self.lock();
        state.check = Some(check);
        state.tripped = None;
        Armed {
            slot: Arc::clone(self),
        }
    }

    /// Run the armed check. A failure is recorded and stays failed for the
    /// rest of the call.
    fn poll(&self) -> Result<(), ()> {
        let mut state = self.lock();
        if state.tripped.is_some() {
            return Err(());
        }
        let Some(check) = state.check.as_mut() else {
            return Ok(());
        };
        match check() {
            Ok(()) => Ok(()),
            Err(msg) => {
                state.tripped = Some(msg);
                Err(())
            }
        }
    }

    /// A test fault: the next call of `op` fails with `make()`, once.
    #[cfg(test)]
    pub fn inject(&self, op: &'static str, make: fn() -> OjasError) {
        self.lock().fault = Some((op, make));
    }

    #[cfg(test)]
    fn injected(&self, op: &'static str) -> Result<(), OjasError> {
        let mut state = self.lock();
        match state.fault {
            Some((at, make)) if at == op => {
                state.fault = None;
                Err(make())
            }
            _ => Ok(()),
        }
    }
}

/// An armed slot. Dropping it disarms the check and forgets the cancel.
pub struct Armed {
    slot: Arc<CancelSlot>,
}

impl Armed {
    /// The cancel message, when a poll in this call failed.
    pub fn tripped(&self) -> Option<String> {
        self.slot.lock().tripped.clone()
    }

    /// `err` as the call reports it: the recorded cancel when one tripped
    /// (the refused op's error is the gate's own), otherwise `err` with its
    /// kind ([`crate::ojas_error`]).
    pub fn report(&self, context: &str, err: &OjasError) -> String {
        match self.tripped() {
            Some(msg) => msg,
            None => crate::ojas_error(context, err),
        }
    }
}

impl Drop for Armed {
    fn drop(&mut self) {
        let mut state = self.slot.lock();
        state.check = None;
        state.tripped = None;
    }
}

/// `B` with a cancel poll before every op that may fail without a commit.
#[derive(Clone)]
pub struct Gated<B> {
    inner: B,
    slot: Arc<CancelSlot>,
}

impl<B: Backend> Gated<B> {
    pub fn new(inner: B, slot: Arc<CancelSlot>) -> Self {
        Self { inner, slot }
    }

    pub fn inner(&self) -> &B {
        &self.inner
    }

    /// `op` may be refused: by a test fault, then, when `cancellable`, by
    /// the armed cancel check.
    fn enter(&self, op: &'static str, cancellable: bool) -> Result<(), OjasError> {
        #[cfg(test)]
        self.slot.injected(op)?;
        if cancellable && self.slot.poll().is_err() {
            return Err(OjasError::Backend {
                id: self.inner.id(),
                detail: format!("{op}: refused after the call was cancelled"),
            });
        }
        Ok(())
    }
}

/// The ops that never poll: the trainer's optimizer phase (`adamw_step`,
/// `muon_ns5_step`) and its finish (`sync`, then one `download`). See the
/// module docs for why a cancel must never land after phase 4.
const NEVER_POLL: [&str; 4] = ["download", "adamw_step", "muon_ns5_step", "sync"];

impl<B: Backend> Gated<B> {
    /// Admission for one `op` method: a test fault, then the cancel poll
    /// unless `op` is in [`NEVER_POLL`]. Run by the generated forwarding
    /// before the inner backend is touched.
    fn admit(&self, op: &'static str) -> Result<(), OjasError> {
        self.enter(op, !NEVER_POLL.contains(&op))
    }
}

/// Every method is generated from `ojas_core::with_backend_methods!`: each
/// `op` runs [`Gated::admit`] then the inner backend's own method, so a
/// provided method the inner backend overrides is never replaced by the
/// trait default, and a method added to the trait is gated without an edit
/// here. A size query or accessor (`pure`) is not gated.
impl<B: Backend> Backend for Gated<B> {
    ojas_core::with_backend_methods!(
        forward_backend_methods,
        { self, self.inner, admit, forward_backend_keep }
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_core::{AdamWConfig, BackendId, Budget, GdnInputs, Tensor};
    use ojas_cpu::CpuBackend;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn gated(slot: &Arc<CancelSlot>) -> Gated<CpuBackend> {
        Gated::new(CpuBackend::new(Budget::new(1 << 20)), Arc::clone(slot))
    }

    #[test]
    fn a_cancel_refuses_polling_ops_and_never_the_commit_ops() {
        let slot = Arc::new(CancelSlot::default());
        let g = gated(&slot);
        let budget = Budget::new(1 << 20);
        let x = Tensor::from_f32(&[1.0, 2.0], &[1, 2], &budget).unwrap();
        let armed = slot.arm(Box::new(|| Err("cancelled: Explicit".to_string())));
        let err = g.linear_forward(&x, &x).unwrap_err();
        assert!(matches!(err, OjasError::Backend { .. }), "{err}");
        assert_eq!(armed.tripped().as_deref(), Some("cancelled: Explicit"));
        assert_eq!(armed.report("op", &err), "cancelled: Explicit");
        // The commit ops still run once the call is cancelled.
        g.sync().unwrap();
        g.download(&x).unwrap();
        let mut p = Tensor::from_f32(&[0.5], &[1], &budget).unwrap();
        let grad = Tensor::from_f32(&[0.1], &[1], &budget).unwrap();
        let mut m = Tensor::from_f32(&[0.0], &[1], &budget).unwrap();
        let mut v = Tensor::from_f32(&[0.0], &[1], &budget).unwrap();
        g.adamw_step(
            &mut p,
            &grad,
            &mut m,
            &mut v,
            0,
            AdamWConfig::nanolab(1e-3, 0.0),
        )
        .unwrap();
        drop(armed);
        // Disarmed: the next call's ops run and nothing is remembered.
        g.linear_forward(&x, &x).unwrap();
        let armed = slot.arm(Box::new(|| Ok(())));
        assert_eq!(armed.tripped(), None);
        g.linear_forward(&x, &x).unwrap();
    }

    /// The gated delta rule goes through the gate to the inner backend
    /// (not the trait default's refusal), and a cancel refuses it, as it
    /// does every polling op.
    #[test]
    fn the_gated_delta_rule_reaches_the_inner_backend_and_polls() {
        let slot = Arc::new(CancelSlot::default());
        let g = gated(&slot);
        let b = Budget::new(1 << 20);
        let f = |shape: &[usize], v: f32| {
            Tensor::from_f32(&vec![v; shape.iter().product()], shape, &b).unwrap()
        };
        let (q, v, gate, beta) = (
            f(&[1, 3, 1, 2], 0.5),
            f(&[1, 3, 1, 2], 0.25),
            f(&[1, 3, 1], -0.5),
            f(&[1, 3, 1], 0.5),
        );
        let x = GdnInputs {
            q: &q,
            k: &q,
            v: &v,
            g: &gate,
            beta: &beta,
            initial_state: None,
        };
        let fwd = g.chunked_gdn_forward(x).unwrap();
        let want = g.inner.chunked_gdn_forward(x).unwrap();
        assert_eq!(
            fwd.output.to_f32_vec().unwrap(),
            want.output.to_f32_vec().unwrap()
        );
        let grad = g
            .chunked_gdn_backward(x, &fwd.checkpoints, &f(&[1, 3, 1, 2], 1.0), None)
            .unwrap();
        assert!(grad.initial_state.is_none());
        let armed = slot.arm(Box::new(|| Err("cancelled: Explicit".to_string())));
        assert!(g.chunked_gdn_forward(x).is_err());
        assert!(g
            .chunked_gdn_backward(x, &fwd.checkpoints, &f(&[1, 3, 1, 2], 1.0), None)
            .is_err());
        assert_eq!(armed.tripped().as_deref(), Some("cancelled: Explicit"));
    }

    /// `scale_grad` reaches the inner backend: the CPU scales a uniquely
    /// owned gradient in place with no charge, where the trait default
    /// would build and charge a broadcast operand and a new product. A
    /// cancel refuses it, as it does every polling op.
    #[test]
    fn scale_grad_reaches_the_inner_backend_in_place_and_polls() {
        let slot = Arc::new(CancelSlot::default());
        let g = gated(&slot);
        let mut grad = Tensor::from_f32(&[1.0, -2.0, 4.0, 0.5], &[2, 2], g.budget()).unwrap();
        let live = g.budget().live_bytes().unwrap();
        g.budget().reset_peak();
        g.scale_grad(&mut grad, 0.25).unwrap();
        assert_eq!(grad.to_f32_vec().unwrap(), vec![0.25, -0.5, 1.0, 0.125]);
        assert_eq!(g.budget().peak_bytes(), live, "scaled in place, no charge");
        let armed = slot.arm(Box::new(|| Err("cancelled: Explicit".to_string())));
        assert!(g.scale_grad(&mut grad, 0.5).is_err());
        assert_eq!(armed.tripped().as_deref(), Some("cancelled: Explicit"));
        assert_eq!(grad.to_f32_vec().unwrap(), vec![0.25, -0.5, 1.0, 0.125]);
    }

    /// Forwards every method to `inner` except
    /// `causal_sdpa_backward_recompute`, which `Spy` overrides.
    macro_rules! all_but_recompute {
        (causal_sdpa_backward_recompute, { $($item:tt)* }) => {};
        ($name:ident, { $($item:tt)* }) => {
            $($item)*
        };
    }

    /// A backend whose `causal_sdpa_backward_recompute` is an override
    /// (the trait default would compose the forward and the backward): it
    /// counts the call and answers with its own marker error.
    struct Spy {
        inner: CpuBackend,
        recomputes: AtomicUsize,
    }

    impl Spy {
        fn new() -> Self {
            Self {
                inner: CpuBackend::new(Budget::new(1 << 20)),
                recomputes: AtomicUsize::new(0),
            }
        }
    }

    impl Backend for Spy {
        ojas_core::with_backend_methods!(
            forward_backend_methods,
            { self, self.inner, none, all_but_recompute }
        );
        fn causal_sdpa_backward_recompute(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: Option<usize>,
        ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
            self.recomputes.fetch_add(1, Ordering::SeqCst);
            Err(OjasError::Backend {
                id: BackendId::Cpu,
                detail: "spy:recompute".into(),
            })
        }
    }

    /// A backend's own `causal_sdpa_backward_recompute` is what a gated
    /// call runs (not the trait default's forward and backward), and a
    /// cancel refuses it before it runs.
    #[test]
    fn the_gated_recompute_override_is_reached_and_polls() {
        let slot = Arc::new(CancelSlot::default());
        let g = Gated::new(Spy::new(), Arc::clone(&slot));
        let b = Budget::new(1 << 10);
        let t = Tensor::from_f32(&[1.0], &[1], &b).unwrap();
        let err = g
            .causal_sdpa_backward_recompute(&t, &t, &t, &t, None)
            .unwrap_err();
        assert!(err.to_string().contains("spy:recompute"), "{err}");
        assert_eq!(g.inner().recomputes.load(Ordering::SeqCst), 1);
        let armed = slot.arm(Box::new(|| Err("cancelled: Explicit".to_string())));
        let err = g
            .causal_sdpa_backward_recompute(&t, &t, &t, &t, None)
            .unwrap_err();
        assert!(err.to_string().contains("refused after"), "{err}");
        assert_eq!(armed.tripped().as_deref(), Some("cancelled: Explicit"));
        assert_eq!(g.inner().recomputes.load(Ordering::SeqCst), 1);
    }

    /// `(kind, name)` of every method in `with_backend_methods!`.
    const METHODS: &[(&str, &str)] = {
        macro_rules! kinds {
            ({} $($kind:ident fn $name:ident(&self $(, $arg:ident : $ty:ty)*) -> $ret:ty;)*) => {
                &[$((stringify!($kind), stringify!($name))),*]
            };
        }
        ojas_core::with_backend_methods!(kinds, {})
    };

    /// The commit contract in the module docs: a cancel refuses every `op`
    /// method but the four in `NEVER_POLL`, which are real `op` methods.
    #[test]
    fn only_the_commit_ops_skip_the_cancel_poll() {
        let slot = Arc::new(CancelSlot::default());
        let g = gated(&slot);
        let _armed = slot.arm(Box::new(|| Err("cancelled: Explicit".to_string())));
        for &(_, name) in METHODS.iter().filter(|(kind, _)| *kind == "op") {
            assert_eq!(
                g.admit(name).is_err(),
                !NEVER_POLL.contains(&name),
                "{name}: wrong polling under a cancel"
            );
        }
        for name in NEVER_POLL {
            assert!(
                METHODS.contains(&("op", name)),
                "{name} is not an op method"
            );
        }
    }

    /// Per-call cost of the gate (admission, then the inner call) on ops that
    /// do almost no work, so the wrapper dominates: `sync` (never polls) and a
    /// four-element add (polls, then adds). Prints ns per call.
    #[test]
    #[ignore = "benchmark; run with --release --ignored --nocapture"]
    fn gate_overhead_bench() {
        const CALLS: usize = 1_000_000;
        let slot = Arc::new(CancelSlot::default());
        let g = gated(&slot);
        let b = Budget::new(1 << 20);
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], &b).unwrap();
        let per_call = |f: &mut dyn FnMut()| {
            let start = std::time::Instant::now();
            for _ in 0..CALLS {
                f();
            }
            start.elapsed().as_nanos() as f64 / CALLS as f64
        };
        let sync = per_call(&mut || g.sync().unwrap());
        let add = per_call(&mut || {
            std::hint::black_box(g.residual_add_forward(&x, &x).unwrap());
        });
        println!("GATE-BENCH sync_ns={sync:.3} add_ns={add:.3}");
    }

    #[test]
    fn an_error_with_no_cancel_keeps_its_kind() {
        let slot = Arc::new(CancelSlot::default());
        let armed = slot.arm(Box::new(|| Ok(())));
        let err = OjasError::NonFinite { op: "x" };
        assert!(armed.report("ctx", &err).starts_with("ojas:E_NONFINITE:"));
    }
}
