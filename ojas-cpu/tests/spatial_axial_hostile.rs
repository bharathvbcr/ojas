//! `axial_rope_2d_tables` against hostile arguments. Its own test binary: on
//! the code before the budget check the first test aborted the process (the
//! frequency vector was allocated before the Budget was consulted), which
//! would take every other test in a shared binary with it.

mod common;

use common::{assert_capacity, assert_range};
use ojas_core::{axial_rope_2d_tables, Budget};

#[test]
fn a_huge_head_dim_is_refused_before_any_uncharged_allocation() {
    let budget = Budget::new(1 << 20);
    // head_dim / 4 = 2^58 f32 = 2^60 bytes: far past the budget, and it must
    // be refused before a vector of that size is asked of the allocator.
    let r = axial_rope_2d_tables(1usize << 60, 1, 1, 10_000.0, 1.0, &budget);
    assert_capacity(r);
    assert_eq!(budget.live_bytes().unwrap(), 0);
}

#[test]
fn a_head_dim_whose_byte_count_overflows_is_an_error_not_a_panic() {
    let budget = Budget::new(1 << 20);
    // (usize::MAX - 3) is a multiple of 4; its table bytes overflow usize.
    assert_range(axial_rope_2d_tables(
        usize::MAX - 3,
        1,
        1,
        10_000.0,
        1.0,
        &budget,
    ));
    // A grid that overflows on its own.
    assert_range(axial_rope_2d_tables(
        8,
        usize::MAX,
        2,
        10_000.0,
        1.0,
        &budget,
    ));
    // Tokens times half overflows although neither does.
    assert_range(axial_rope_2d_tables(
        1usize << 40,
        1usize << 40,
        1,
        10_000.0,
        1.0,
        &budget,
    ));
    assert_eq!(budget.live_bytes().unwrap(), 0);
}

#[test]
fn a_table_larger_than_the_budget_is_refused_by_the_budget() {
    // 4096 x 4096 tokens at head_dim 64: 2 x 2^24 x 32 x 4 bytes = 4 GiB.
    let budget = Budget::new(1 << 30);
    assert_capacity(axial_rope_2d_tables(64, 4096, 4096, 10_000.0, 1.0, &budget));
    assert_eq!(budget.live_bytes().unwrap(), 0);
}

#[test]
fn the_budget_covers_both_tables_and_the_frequencies_to_the_byte() {
    // head_dim 8, a 3 x 2 grid: 6 tokens x 4 half = 96 bytes a table, and
    // 2 frequencies = 8 bytes: 200 bytes at the peak.
    let fits = Budget::new(200);
    let (cos, sin) = axial_rope_2d_tables(8, 3, 2, 10_000.0, 1.0, &fits).unwrap();
    // The frequencies are released; the two tables stay charged.
    assert_eq!(fits.live_bytes().unwrap(), 192);
    drop((cos, sin));
    assert_eq!(fits.live_bytes().unwrap(), 0);
    let short = Budget::new(199);
    assert_capacity(axial_rope_2d_tables(8, 3, 2, 10_000.0, 1.0, &short));
    assert_eq!(short.live_bytes().unwrap(), 0);
}

#[test]
fn a_theta_beyond_f32_is_refused_like_any_other_unusable_theta() {
    let budget = Budget::new(1 << 20);
    // 1e300 is finite as an f64 but +inf as the f32 the powers are formed
    // in: it used to return Ok with frequencies [1, 0, 0, ...].
    assert_range(axial_rope_2d_tables(8, 2, 2, 1e300, 1.0, &budget));
    assert_range(axial_rope_2d_tables(
        8,
        2,
        2,
        f64::from(f32::MAX) * 2.0,
        1.0,
        &budget,
    ));
    // Above 1 as an f64, but 1.0 as an f32: every frequency would be 1.
    assert_range(axial_rope_2d_tables(8, 2, 2, 1.0 + 1e-12, 1.0, &budget));
    for theta in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0, 0.5, -3.0] {
        assert_range(axial_rope_2d_tables(8, 2, 2, theta, 1.0, &budget));
    }
    // The largest theta an f32 holds is usable.
    assert!(axial_rope_2d_tables(8, 2, 2, f64::from(f32::MAX), 1.0, &budget).is_ok());
}
