//! The KV cache both decoders keep, and the step they run on it.
//!
//! A cache is one key and one value tensor per layer, `[1, slots, n_kv_head,
//! head_dim]`, time-major. It is a ring: position `j` lives in slot
//! `j % slots` (`Backend::kv_cache_write` and
//! `Backend::cached_attention_forward` both index it so). Without a sliding
//! window `slots` is the capacity, so no position ever wraps. With a window
//! `W` smaller than the capacity, a query reads only the last `W`
//! positions, and the ring keeps `2W - 1` slots (or the capacity, if that is
//! fewer): enough for one call of up to `W` new tokens to attend to the `W
//! - 1` positions before it without overwriting any of them.
//!
//! [`Ring`] is that bookkeeping, once for [`crate::DeviceDecoder`] and the
//! host [`crate::KvCache`]; [`run_pieces`] is the step both run.

use ojas_core::{Backend, DType, OjasError, Tensor};
use ojas_model::{block_with, Eval, Graph, ModelParams, ModelSpec, Rope};

/// Where a cache's positions live, and which of them are valid.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ring {
    /// Most positions the cache takes (the RoPE table's length).
    capacity: usize,
    /// Slots per layer tensor.
    slots: usize,
    /// The window attention runs under, `None` when it covers `capacity`.
    window: Option<usize>,
    /// Positions filled; the next token goes at `len`.
    len: usize,
    /// One past the last position any call wrote, failed calls included:
    /// slot `j % slots` holds position `j` only for `j` in `written -
    /// slots..written`.
    written: usize,
}

impl Ring {
    /// A ring for `capacity` positions of a model attending under
    /// `window` ([`ModelSpec::attention_window`]).
    pub(crate) fn new(window: Option<usize>, capacity: usize) -> Self {
        let window = window.filter(|&w| w < capacity);
        let slots = window.map_or(capacity, |w| capacity.min(2 * w - 1));
        Self {
            capacity,
            slots,
            window,
            len: 0,
            written: 0,
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn slots(&self) -> usize {
        self.slots
    }

    pub(crate) fn window(&self) -> Option<usize> {
        self.window
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn remaining(&self) -> usize {
        self.capacity - self.len
    }

    /// Most tokens one piece of a call may forward and stay all or
    /// nothing: the window when the ring is shorter than the capacity
    /// (a longer piece would overwrite positions the state before it
    /// needs), else every free position.
    pub(crate) fn piece(&self) -> usize {
        match self.window {
            Some(w) if self.slots < self.capacity => w,
            _ => self.capacity,
        }
    }

    /// [`OjasError::CapacityExceeded`] for `extra` more positions of
    /// `width` f32s, in bytes.
    pub(crate) fn refusal(&self, width: usize, extra: usize) -> OjasError {
        let width = u64::try_from(width)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .unwrap_or(u64::MAX);
        let live = (self.len as u64).saturating_mul(width);
        OjasError::CapacityExceeded {
            requested: live.saturating_add((extra as u64).saturating_mul(width)),
            cap: (self.capacity as u64).saturating_mul(width),
            live,
        }
    }

    /// Forget every position.
    pub(crate) fn reset(&mut self) {
        self.len = 0;
        self.written = 0;
    }

    /// Keep positions `0..to` and forget the rest. Refused with
    /// [`OjasError::OutOfRange`], changing nothing, when `to` is past
    /// [`Self::len`] (those slots hold nothing valid) or when a position the
    /// next token at `to` attends to has been overwritten in the ring.
    pub(crate) fn truncate(&mut self, to: usize, op: &'static str) -> Result<(), OjasError> {
        if to > self.len {
            return Err(OjasError::OutOfRange {
                op,
                detail: format!("cannot truncate {} filled positions to {to}", self.len),
            });
        }
        if let Some(w) = self.window {
            let oldest_needed = (to + 1).saturating_sub(w);
            let oldest_held = self.written.saturating_sub(self.slots);
            if to > 0 && oldest_needed < oldest_held {
                return Err(OjasError::OutOfRange {
                    op,
                    detail: format!(
                        "position {oldest_needed}, in the window of position {to}, was \
                         overwritten in the {}-slot ring (it holds {oldest_held}..{})",
                        self.slots, self.written
                    ),
                });
            }
        }
        self.len = to;
        Ok(())
    }
}

/// Per-layer key and value rings and their [`Ring`].
pub(crate) struct Kv {
    pub(crate) keys: Vec<Tensor>,
    pub(crate) values: Vec<Tensor>,
    pub(crate) ring: Ring,
}

impl Kv {
    /// `n_layer` key and value tensors `[1, ring.slots(), n_kv_head,
    /// head_dim]` from `alloc`.
    pub(crate) fn new(
        [n_layer, n_kv_head, head_dim]: [usize; 3],
        ring: Ring,
        mut alloc: impl FnMut(&[usize]) -> Result<Tensor, OjasError>,
    ) -> Result<Self, OjasError> {
        let shape = [1, ring.slots(), n_kv_head, head_dim];
        let mut keys = Vec::with_capacity(n_layer);
        let mut values = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            keys.push(alloc(&shape)?);
            values.push(alloc(&shape)?);
        }
        Ok(Self { keys, values, ring })
    }
}

/// Forward `tokens` (non-empty, at most [`Ring::remaining`]) at positions
/// `kv.ring.len()..`, in pieces of at most [`Ring::piece`] tokens, and
/// return the last position's `[1, vocab]` logits on `B` with the last
/// piece's length. `ids` makes a piece's `[1, Tn]` `U32` ids on `B`; `rope`
/// is the table for every position of the cache. Every piece but the last
/// is synced and appended (`len` advances); the last is the caller's to
/// sync and append, so a failure there leaves `len` where that piece
/// began. Only the last position runs the final norm and the head.
pub(crate) fn run_pieces<B: Backend>(
    eval: &mut Eval<B>,
    spec: &ModelSpec,
    params: &ModelParams<Tensor>,
    kv: &mut Kv,
    rope: &Rope,
    tokens: &[u32],
    mut ids: impl FnMut(&Eval<B>, &[u32]) -> Result<Tensor, OjasError>,
) -> Result<(Tensor, usize), OjasError> {
    let mut pieces = tokens.chunks(kv.ring.piece()).peekable();
    while let Some(piece) = pieces.next() {
        let rows = rope.slice(kv.ring.len(), piece.len())?;
        let piece_ids = ids(eval, piece)?;
        let x = blocks(eval, spec, params, kv, &rows, &piece_ids)?;
        if pieces.peek().is_none() {
            return Ok((head(eval, spec, params, &x, piece.len())?, piece.len()));
        }
        eval.backend().sync()?;
        kv.ring.len += piece.len();
    }
    Err(OjasError::Shape {
        op: "decode step",
        detail: "no tokens".into(),
    })
}

/// Every block over `ids` at positions `len..len + Tn` (`rope` holds their
/// rows). Each layer writes its post-RoPE keys and blended values into its
/// ring with `Backend::kv_cache_write`, then attends over the visible
/// positions with [`Graph::cached_attn`]. Returns the hidden state `[1, Tn,
/// n_embd]`.
fn blocks<B: Backend>(
    eval: &mut Eval<B>,
    spec: &ModelSpec,
    params: &ModelParams<Tensor>,
    kv: &mut Kv,
    rope: &Rope,
    ids: &Tensor,
) -> Result<Tensor, OjasError> {
    let (at, tn, window) = (kv.ring.len, rope.len(), kv.ring.window);
    // From here the slots of `at..at + tn` may hold this call's values.
    kv.ring.written = kv.ring.written.max(at + tn);
    let mut x = eval.embedding(&params.tok_emb, ids)?;
    let mut v0: Option<Tensor> = None;
    let layers = params
        .blocks
        .iter()
        .zip(kv.keys.iter_mut().zip(kv.values.iter_mut()));
    for (p, (keys, values)) in layers {
        let attend = |g: &mut Eval<B>, q: &Tensor, k: &Tensor, v: &Tensor| {
            g.kv_cache_write(keys, k, at)?;
            g.kv_cache_write(values, v, at)?;
            g.cached_attn(q, keys, values, at + tn, window)
        };
        let out = block_with(eval, spec, p, &x, v0.as_ref(), rope, 1, attend)?;
        if v0.is_none() {
            v0 = Some(out.raw_v);
        }
        x = out.x;
    }
    Ok(x)
}

/// The final norm and the head on the last of `tn` rows of `x`, a view:
/// nothing is uploaded to pick it. `[1, vocab]` logits.
fn head<B: Backend>(
    eval: &mut Eval<B>,
    spec: &ModelSpec,
    params: &ModelParams<Tensor>,
    x: &Tensor,
    tn: usize,
) -> Result<Tensor, OjasError> {
    let d = spec.n_embd;
    let row_bytes = d * DType::F32.size();
    let last = x.narrow((tn - 1) * row_bytes, &[1, d], &[d, 1])?;
    let h = eval.rms_norm(&last, &params.norm_f, spec.eps())?;
    eval.linear(&h, &params.tok_emb)
}

/// Advance `kv`'s length by the last piece of [`run_pieces`], once the
/// caller has its result.
pub(crate) fn append(kv: &mut Kv, n: usize) {
    kv.ring.len += n;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ring_is_the_capacity_without_a_window_and_2w_minus_1_with_one() {
        let full = Ring::new(None, 100);
        assert_eq!(
            (full.slots(), full.piece(), full.window()),
            (100, 100, None)
        );
        let wide = Ring::new(Some(100), 100);
        assert_eq!((wide.slots(), wide.window()), (100, None));
        let w8 = Ring::new(Some(8), 100);
        assert_eq!((w8.slots(), w8.piece(), w8.window()), (15, 8, Some(8)));
        // A ring as long as the capacity never wraps: any call is one piece.
        let short = Ring::new(Some(8), 12);
        assert_eq!((short.slots(), short.piece()), (12, 12));
    }

    /// W = 4, 7 slots. After 10 positions, slots hold 3..10; the next token
    /// at `to` attends to `to - 3..to`, so `to >= 6` is allowed and below is
    /// refused. A failed call writing 10..14 leaves `len` 10 but moves the
    /// held range to 7..14: rolling back to 7 (needs 4..7) is then refused.
    #[test]
    fn truncate_refuses_a_prefix_whose_window_was_overwritten() {
        let mut r = Ring::new(Some(4), 100);
        r.len = 10;
        r.written = 10;
        let mut ok = r;
        assert!(ok.truncate(6, "t").is_ok());
        assert_eq!(ok.len(), 6);
        let mut bad = r;
        assert!(matches!(
            bad.truncate(5, "t"),
            Err(OjasError::OutOfRange { .. })
        ));
        assert_eq!(bad.len(), 10);
        assert!(matches!(
            r.truncate(11, "t"),
            Err(OjasError::OutOfRange { .. })
        ));

        let mut failed = r;
        failed.written = 14;
        assert!(matches!(
            failed.truncate(7, "t"),
            Err(OjasError::OutOfRange { .. })
        ));
        assert!(failed.truncate(10, "t").is_ok());
        let mut zero = failed;
        assert!(zero.truncate(0, "t").is_ok());
        failed.reset();
        assert_eq!((failed.len(), failed.written), (0, 0));
    }
}
