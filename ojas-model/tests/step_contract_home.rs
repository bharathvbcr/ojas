//! The Tape providers implement the one `StepProvider` contract, which lives in `ojas-core`
//! (re-exported by `ojas_train::lappi::step`). Passing a provider where the core-path bound is
//! required compiles only when the paths name one trait.

use ojas_core::step::StepProvider;
use ojas_cpu::CpuBackend;
use ojas_model::image_mlp::ImageMlpProvider;
use ojas_model::qwen35::TapeStepProvider;

fn is_core_provider<P: StepProvider>() {}

#[test]
fn the_tape_providers_implement_the_core_contract() {
    is_core_provider::<TapeStepProvider<CpuBackend>>();
    is_core_provider::<ImageMlpProvider<CpuBackend>>();
}
