//! **K9: the token embedding**, forward (a row gather) and backward (each
//! output row's gradient added into its id's table row), f64. The backward sums
//! a repeated id's rows in position order, the deterministic order the scoping
//! doc fixes for the kernel (`cuda-backend-scoping.md` §3 K9). A port of tessl's
//! in-test reference (`tests/qwen35_bwd.rs:1098-1103`), which adds into a prior.
//!
//! # Validation
//!
//! - **golden**: `tests/fixtures/goldens/embed_*` (torch float64
//!   `F.embedding` and autograd, repeated and out-of-order ids).

fn check_ids(ids: &[usize], vocab: usize) {
    if let Some((r, &id)) = ids.iter().enumerate().find(|(_, &id)| id >= vocab) {
        panic!("embed: ids[{r}] = {id} is not below vocab {vocab}");
    }
}

/// `y [rows, hidden]` = `table[ids]`.
pub fn embed_rows_fwd_f64(table: &[f64], ids: &[usize], vocab: usize, hidden: usize) -> Vec<f64> {
    assert_eq!(
        table.len(),
        vocab * hidden,
        "embed: table must be [vocab, hidden]"
    );
    check_ids(ids, vocab);
    ids.iter()
        .flat_map(|&id| table[id * hidden..(id + 1) * hidden].to_vec())
        .collect()
}

/// `dtable [vocab, hidden]`: `prior` (zeros when `None`) plus, for each row in
/// position order, `dy[row]` added into `dtable[ids[row]]`.
pub fn embed_rows_bwd_f64(
    ids: &[usize],
    dy: &[f64],
    vocab: usize,
    hidden: usize,
    prior: Option<&[f64]>,
) -> Vec<f64> {
    assert_eq!(
        dy.len(),
        ids.len() * hidden,
        "embed: dy must be [rows, hidden]"
    );
    check_ids(ids, vocab);
    let mut dt = match prior {
        Some(p) => {
            assert_eq!(
                p.len(),
                vocab * hidden,
                "embed: prior must be [vocab, hidden]"
            );
            p.to_vec()
        }
        None => vec![0.0; vocab * hidden],
    };
    for (r, &id) in ids.iter().enumerate() {
        for c in 0..hidden {
            dt[id * hidden + c] += dy[r * hidden + c];
        }
    }
    dt
}
