//! Where the header's tree-byte budget actually bites.
//!
//! The reader builds a `JsonValue` tree from the header text, and a header
//! that is almost all shape dimensions is the worst case: `0,` is two input
//! bytes and becomes a number node, so the tree is some multiple of the text.
//! The budget (`8 * input + 64 KiB`) refuses such a header instead of building
//! the tree.
//!
//! This matters because ojas-io is the one reader both ojas and tessl go
//! through now, and tessl's reader had no such bound — it accepted a 10 MB
//! header of five million dimensions and only bounded the single largest
//! allocation. This file records what the bound costs, so the narrowing is a
//! measured fact rather than a guess: a real checkpoint header is nowhere
//! near it, and the headers it refuses are not ones any model writes.

use ojas_io::SafeTensors;

/// A one-tensor header with `dims` zero-sized dimensions.
fn header_with_dims(dims: usize) -> Vec<u8> {
    let zeros = vec!["0"; dims].join(",");
    let h = format!(r#"{{"x":{{"dtype":"U8","shape":[{zeros}],"data_offsets":[0,0]}}}}"#);
    let mut b = (h.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(h.as_bytes());
    b
}

#[test]
fn a_header_of_mostly_dimensions_is_refused_above_a_measured_size() {
    // Found by bisection below, and asserted so a change to node size or to
    // the budget shows up here rather than in a loader.
    let mut lo = 1usize; // accepted
    let mut hi = 5_000_000usize; // refused
    assert!(SafeTensors::parse(&header_with_dims(lo)).is_ok());
    let err = SafeTensors::parse(&header_with_dims(hi)).unwrap_err();
    assert!(
        err.detail().contains("byte budget exceeded"),
        "five million dimensions must be refused by the budget: {}",
        err.detail()
    );
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if SafeTensors::parse(&header_with_dims(mid)).is_ok() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let bytes = header_with_dims(lo).len();
    // The threshold is a property of the node size and the budget, not of
    // this machine. Bracketed rather than pinned exactly so a one-node-byte
    // change does not fail the suite without telling anyone what moved.
    // Measured 2026-11-01: 4,099 dimensions, an 8,257-byte header. The budget
    // is `8 * input + 64 KiB` and a dimension-dense header builds about
    // sixteen tree bytes per input byte, so the two cross at about 8 KiB of
    // text. Bracketed rather than pinned exactly, so a one-byte change in node
    // size reports what moved instead of failing on an exact number.
    assert!(
        (3_000..6_000).contains(&lo),
        "the dimension threshold moved to {lo} ({bytes} header bytes); \
         if that was intended, say so here and in tessl's \
         header_amplification_is_pinned"
    );
    // And the refusal names the budget rather than coming back as a parse
    // error about the text.
    let err = SafeTensors::parse(&header_with_dims(hi)).unwrap_err();
    assert!(
        err.detail().contains("byte budget exceeded"),
        "{}",
        err.detail()
    );
}

/// The header a real model writes is far below the bound: hundreds of tensors
/// with names, dtypes and rank-2 shapes, which is what the budget has to
/// admit.
#[test]
fn a_realistic_checkpoint_header_is_far_below_the_bound() {
    let mut entries = Vec::new();
    let mut at = 0u64;
    for layer in 0..126 {
        for (part, rows, cols) in [
            ("self_attn.q_proj.weight", 4096usize, 2048usize),
            ("self_attn.k_proj.weight", 512, 2048),
            ("mlp.gate_proj.weight", 6144, 2048),
            ("mlp.down_proj.weight", 2048, 6144),
            ("input_layernorm.weight", 1, 2048),
        ] {
            let end = at + (rows * cols * 2) as u64;
            entries.push(format!(
                r#""model.language_model.layers.{layer}.{part}":{{"dtype":"BF16","shape":[{rows},{cols}],"data_offsets":[{at},{end}]}}"#
            ));
            at = end;
        }
    }
    let h = format!("{{{}}}", entries.join(","));
    let mut b = (h.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(h.as_bytes());
    b.resize(b.len() + at as usize, 0);

    let st = SafeTensors::parse(&b).expect("a realistic header must parse");
    // 630 tensors, the shape of the real Qwen3.5-2B header (632).
    assert_eq!(st.names().count(), 126 * 5);
    // Well inside the cap: the point is the margin, not the exact size.
    assert!(h.len() < 100_000, "header is {} bytes", h.len());
}
