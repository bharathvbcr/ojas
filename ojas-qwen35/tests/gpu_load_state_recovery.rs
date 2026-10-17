//! `load_state` and a malformed or damaged state directory. Opens a Metal
//! runtime: `#[ignore]`, run in the agreed GPU window.
//!
//! ```text
//! cargo test -p ojas-qwen35 --release --test gpu_load_state_recovery -- --ignored --test-threads=1
//! ```

#![cfg(target_os = "macos")]

mod support;

use ojas_qwen35::{Qwen35Error, StateIndex, Which};
use support::{backward_one, cut_short, scratch, stepped, table_bits, take_step};

/// A malformed state is refused before the first write: nothing moved and
/// the provider is still usable.
#[test]
#[ignore]
fn gpu_load_state_refuses_a_malformed_file_and_stays_usable() {
    let mut s = stepped();
    s.discard_gradients();
    let dir = scratch("malformed");
    s.save_state(&dir).unwrap();
    backward_one(&mut s, 11);
    take_step(&mut s);
    s.discard_gradients();
    let before = table_bits(&s, Which::Parameters);
    cut_short(&dir, "exp_avg_sq-");
    s.load_state(&dir).unwrap_err();
    assert_eq!(
        table_bits(&s, Which::Parameters),
        before,
        "a refused load wrote"
    );
    // Usable: it still trains.
    backward_one(&mut s, 2);
    take_step(&mut s);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A failure after a write has landed (a file cut short underneath an index
/// that had already passed) poisons the provider: it holds a mix of states.
#[test]
#[ignore]
fn gpu_load_state_poisons_only_after_a_write() {
    let mut s = stepped();
    s.discard_gradients();
    let dir = scratch("poison");
    s.save_state(&dir).unwrap();
    let idx = StateIndex::read(&dir, s.parameter_table(), &s.config().canonical_summary()).unwrap();
    cut_short(&dir, "exp_avg_sq-");
    let e = s.load_state_from_index(&idx).unwrap_err();
    eprintln!("refused: {e}");
    match s.read_table(Which::Parameters) {
        Err(Qwen35Error::Poisoned(why)) => assert!(why.contains("load_state"), "{why}"),
        Err(other) => panic!("expected Poisoned, got {other:?}"),
        Ok(_) => panic!("a provider holding a mix of two states was readable"),
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
