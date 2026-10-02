//! Blocked GEMM driver, panel packing and the portable micro-kernel.
//!
//! Loop order (GotoBLAS): `jc` over `NC` columns, `pc` over `KC` depth, then
//! `ic` over `MC` rows, then the `NR`-wide and `MR`-tall register tiles. Each
//! micro-tile's accumulators are written back to `C` after a `KC` block and
//! reloaded for the next. An `f32` store and load is exact, so each element's
//! chain is the same for any `KC`, any tile position and any padding.

use std::cell::RefCell;

use crate::layout::Problem;

/// Depth of one packed block.
const KC: usize = 512;
/// Target rows of a packed A block. Rounded down to a multiple of `MR`.
const MC_TARGET: usize = 128;
/// Target columns of a packed B block. Rounded down to a multiple of `NR`.
const NC_TARGET: usize = 960;
/// Largest `MR * NR` of any kernel.
pub(crate) const MAX_TILE: usize = 96;

/// A register-blocked `MR × NR` kernel over packed panels.
pub(crate) trait MicroKernel: Copy {
    const MR: usize;
    const NR: usize;

    /// For `p` in `0..kc` ascending, `acc[r*NR + c] = fma(a[p*MR + r],
    /// b[p*NR + c], acc[r*NR + c])`.
    ///
    /// Panics unless `a.len() >= kc*MR`, `b.len() >= kc*NR` and
    /// `acc.len() == MR*NR`. The driver always satisfies these.
    fn run(self, kc: usize, a: &[f32], b: &[f32], acc: &mut [f32]);
}

/// Safe 8×8 kernel built on `f32::mul_add`.
#[derive(Clone, Copy)]
pub(crate) struct Portable;

impl MicroKernel for Portable {
    const MR: usize = 8;
    const NR: usize = 8;

    fn run(self, kc: usize, a: &[f32], b: &[f32], acc: &mut [f32]) {
        let a = &a[..kc * 8];
        let b = &b[..kc * 8];
        let acc: &mut [f32; 64] = acc.try_into().expect("portable tile is 8x8");
        let mut t = [[0.0f32; 8]; 8];
        for (r, row) in t.iter_mut().enumerate() {
            row.copy_from_slice(&acc[r * 8..r * 8 + 8]);
        }
        for (ap, bp) in a.as_chunks::<8>().0.iter().zip(b.as_chunks::<8>().0) {
            for (row, &ar) in t.iter_mut().zip(ap) {
                for (x, &bc) in row.iter_mut().zip(bp) {
                    *x = ar.mul_add(bc, *x);
                }
            }
        }
        for (r, row) in t.iter().enumerate() {
            acc[r * 8..r * 8 + 8].copy_from_slice(row);
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` on this thread's packing buffer, or on a fresh one if the
/// thread-local buffer is unavailable (thread teardown).
fn with_scratch(mut f: impl FnMut(&mut Vec<f32>)) {
    let ran = SCRATCH
        .try_with(|cell| match cell.try_borrow_mut() {
            Ok(mut v) => {
                f(&mut v);
                true
            }
            Err(_) => false,
        })
        .unwrap_or(false);
    if !ran {
        f(&mut Vec::new());
    }
}

fn round_up(x: usize, to: usize) -> usize {
    x.div_ceil(to) * to
}

/// `C = 0` over the valid region unless accumulating; used when `k == 0`.
pub(crate) fn zero_unless_accumulate(p: &Problem, c: &mut [f32]) {
    if p.accumulate || p.n == 0 {
        return;
    }
    for i in 0..p.m {
        c[i * p.c_rs..i * p.c_rs + p.n].fill(0.0);
    }
}

/// Packs `B[pc..pc+kc, jc..jc+nc]` into `NR`-wide panels, `kc × NR` each,
/// row-major within a panel, zero-padding the last panel's columns.
#[allow(clippy::too_many_arguments)]
fn pack_b(
    b: &[f32],
    rs: usize,
    cs: usize,
    pc: usize,
    kc: usize,
    jc: usize,
    nc: usize,
    nr: usize,
    out: &mut [f32],
) {
    let panels = nc.div_ceil(nr);
    for (jp, panel) in out.chunks_exact_mut(kc * nr).take(panels).enumerate() {
        let j0 = jc + jp * nr;
        let w = nr.min(jc + nc - j0);
        if cs == 1 {
            for (p, dst) in panel.chunks_exact_mut(nr).enumerate() {
                let base = (pc + p) * rs + j0;
                dst[..w].copy_from_slice(&b[base..base + w]);
                dst[w..].fill(0.0);
            }
        } else if rs == 1 {
            for c in 0..w {
                let col = &b[j0 * cs + c * cs + pc..][..kc];
                for (p, &v) in col.iter().enumerate() {
                    panel[p * nr + c] = v;
                }
            }
            for dst in panel.chunks_exact_mut(nr) {
                dst[w..].fill(0.0);
            }
        } else {
            for (p, dst) in panel.chunks_exact_mut(nr).enumerate() {
                let base = (pc + p) * rs;
                for (c, d) in dst[..w].iter_mut().enumerate() {
                    *d = b[base + (j0 + c) * cs];
                }
                dst[w..].fill(0.0);
            }
        }
    }
}

/// Packs `A[ic..ic+mc, pc..pc+kc]` into `MR`-tall panels, `kc × MR` each
/// (`MR` values per `p`), zero-padding the last panel's rows.
#[allow(clippy::too_many_arguments)]
fn pack_a(
    a: &[f32],
    rs: usize,
    cs: usize,
    ic: usize,
    mc: usize,
    pc: usize,
    kc: usize,
    mr: usize,
    out: &mut [f32],
) {
    let panels = mc.div_ceil(mr);
    for (ip, panel) in out.chunks_exact_mut(kc * mr).take(panels).enumerate() {
        let i0 = ic + ip * mr;
        let h = mr.min(ic + mc - i0);
        if cs == 1 {
            for r in 0..h {
                let row = &a[(i0 + r) * rs + pc..][..kc];
                for (p, &v) in row.iter().enumerate() {
                    panel[p * mr + r] = v;
                }
            }
            if h < mr {
                for dst in panel.chunks_exact_mut(mr) {
                    dst[h..].fill(0.0);
                }
            }
        } else if rs == 1 {
            for (p, dst) in panel.chunks_exact_mut(mr).enumerate() {
                let base = (pc + p) * cs + i0;
                dst[..h].copy_from_slice(&a[base..base + h]);
                dst[h..].fill(0.0);
            }
        } else {
            for (p, dst) in panel.chunks_exact_mut(mr).enumerate() {
                let col = (pc + p) * cs;
                for (r, d) in dst[..h].iter_mut().enumerate() {
                    *d = a[(i0 + r) * rs + col];
                }
                dst[h..].fill(0.0);
            }
        }
    }
}

/// Runs a validated problem on kernel `K`.
pub(crate) fn gemm<K: MicroKernel>(kern: K, p: &Problem, a: &[f32], b: &[f32], c: &mut [f32]) {
    let (m, n, k) = (p.m, p.n, p.k);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        zero_unless_accumulate(p, c);
        return;
    }
    let (mr, nr) = (K::MR, K::NR);
    debug_assert!(mr * nr <= MAX_TILE);
    let mc_max = (MC_TARGET / mr).max(1) * mr;
    let nc_max = (NC_TARGET / nr).max(1) * nr;
    let kc_max = KC.min(k);
    let a_len = mc_max.min(round_up(m, mr)) * kc_max;
    let b_len = nc_max.min(round_up(n, nr)) * kc_max;
    let c_rs = p.c_rs;

    with_scratch(|scratch| {
        if scratch.len() < a_len + b_len {
            scratch.resize(a_len + b_len, 0.0);
        }
        let (apack, rest) = scratch.split_at_mut(a_len);
        let bpack = &mut rest[..b_len];
        let mut tile_buf = [0.0f32; MAX_TILE];
        let tile = &mut tile_buf[..mr * nr];

        for jc in (0..n).step_by(nc_max) {
            let nc = nc_max.min(n - jc);
            for pc in (0..k).step_by(kc_max) {
                let kc = kc_max.min(k - pc);
                let load_c = p.accumulate || pc > 0;
                pack_b(b, p.b_rs, p.b_cs, pc, kc, jc, nc, nr, bpack);
                for ic in (0..m).step_by(mc_max) {
                    let mc = mc_max.min(m - ic);
                    pack_a(a, p.a_rs, p.a_cs, ic, mc, pc, kc, mr, apack);
                    for jr in (0..nc).step_by(nr) {
                        let w = nr.min(nc - jr);
                        let bp = &bpack[(jr / nr) * kc * nr..][..kc * nr];
                        for ir in (0..mc).step_by(mr) {
                            let h = mr.min(mc - ir);
                            let ap = &apack[(ir / mr) * kc * mr..][..kc * mr];
                            let (row0, col0) = (ic + ir, jc + jr);
                            if !load_c || h < mr || w < nr {
                                tile.fill(0.0);
                            }
                            if load_c {
                                for r in 0..h {
                                    let src = &c[(row0 + r) * c_rs + col0..][..w];
                                    tile[r * nr..r * nr + w].copy_from_slice(src);
                                }
                            }
                            kern.run(kc, ap, bp, tile);
                            for r in 0..h {
                                c[(row0 + r) * c_rs + col0..][..w]
                                    .copy_from_slice(&tile[r * nr..r * nr + w]);
                            }
                        }
                    }
                }
            }
        }
    });
}
