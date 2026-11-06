//! A panicking task is `Err(Backend)` carrying its message on every path an
//! op can take, at every thread count: inline on the caller, split across
//! scoped threads, or on pool workers. It never unwinds into the caller, its
//! captures are dropped, and the next op on the same pool is unaffected.

use std::sync::Arc;

use ojas_core::{Numerics, OjasError};

use super::{scoped, Exec, Pool};
use crate::attn::{fill_parts, TASK_WORK};

const MESSAGE: &str = "task five exploded";

/// Run `op` on a pool of `threads` with a captured `Arc` and demand the
/// error, the message, the dropped capture and a healthy pool afterwards.
fn expect_panic_error(
    what: &str,
    threads: usize,
    op: impl FnOnce(Exec<'_>, Arc<Vec<u8>>) -> Result<(), OjasError>,
) {
    let pool = Arc::new(Pool::new(threads).unwrap());
    let exec = Exec {
        pool: &pool,
        numerics: Numerics::Fast,
    };
    let sentinel = Arc::new(vec![0u8; 16]);
    let got = op(exec, Arc::clone(&sentinel));
    match got {
        Err(OjasError::Backend { detail, .. }) => assert!(
            detail.contains(MESSAGE),
            "{what} at threads {threads}: panic text lost: {detail}"
        ),
        other => panic!("{what} at threads {threads}: expected Backend, got {other:?}"),
    }
    assert_eq!(
        Arc::strong_count(&sentinel),
        1,
        "{what} at threads {threads}: a capture outlived the failed op"
    );
    let after = scoped::map(exec, 9, |i| Ok(i + 1)).unwrap();
    assert_eq!(after, (1..=9).collect::<Vec<_>>(), "{what} at {threads}");
    let after = pool.run(9, |i| i * 2).unwrap();
    assert_eq!(after, (0..9).map(|i| i * 2).collect::<Vec<_>>());
}

fn boom(held: &Arc<Vec<u8>>, i: usize) -> usize {
    assert_eq!(held.len(), 16);
    if i == 5 {
        panic!("{MESSAGE}");
    }
    i
}

#[test]
fn scoped_map_panic_is_backend_inline_and_split() {
    for threads in [1usize, 4] {
        expect_panic_error("scoped::map", threads, |e, held| {
            scoped::map(e, 8, |i| Ok(boom(&held, i))).map(drop)
        });
    }
}

#[test]
fn exec_map_panic_is_backend_inline_and_pooled() {
    for threads in [1usize, 4] {
        // `work < min_work` keeps the op inline whatever the thread count.
        for (what, work) in [("inline", 0usize), ("pooled", 1 << 20)] {
            expect_panic_error(&format!("Exec::map {what}"), threads, |e, held| {
                e.map(8, work, 1 << 10, move |i| boom(&held, i)).map(drop)
            });
        }
    }
}

#[test]
fn pool_run_panic_is_backend_at_every_width() {
    for threads in [1usize, 4] {
        expect_panic_error("Pool::run", threads, |e, held| {
            e.pool.run(8, move |i| boom(&held, i)).map(drop)
        });
    }
}

#[test]
fn attn_fill_parts_panic_is_backend_inline_and_split() {
    for threads in [1usize, 4] {
        for (what, work) in [("inline", 1usize), ("split", 2 * TASK_WORK)] {
            expect_panic_error(&format!("fill_parts {what}"), threads, |e, held| {
                let parts: Vec<usize> = (0..8).collect();
                fill_parts(e, work, parts, |i, _| Ok(boom(&held, i))).map(drop)
            });
        }
    }
}

#[test]
fn scoped_fill_parts_and_chunks_into_n_panic_is_backend() {
    for threads in [1usize, 4] {
        expect_panic_error("scoped::fill_parts", threads, |e, held| {
            let parts: Vec<usize> = (0..8).collect();
            scoped::fill_parts(e, parts, |i, _| Ok(boom(&held, i))).map(drop)
        });
        // Small outputs run on the caller whatever the width; large ones split.
        for (what, len) in [("inline", 64usize), ("split", 1 << 20)] {
            expect_panic_error(&format!("chunks_into {what}"), threads, |e, held| {
                let mut out = vec![0.0f32; len];
                scoped::chunks_into(e, &mut out, len, 1, 8, |range, _| {
                    Ok(boom(&held, usize::from(range.end == len) * 5))
                })
                .map(drop)
            });
        }
    }
}
