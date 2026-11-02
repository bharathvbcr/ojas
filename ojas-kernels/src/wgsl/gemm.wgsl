// C[M, N] = A[M, K] @ B[K, N] with element strides, so one kernel covers the
// NN, NT and TN products a linear needs and nothing is transposed on the host.
//
// A(m, k) = ga[m * a_rs + k * a_cs], B(k, n) = gb[k * b_rs + n * b_cs], C is
// row-major. Words: 0 M, 1 N, 2 K, 3 a_rs, 4 a_cs, 5 b_rs, 6 b_cs, then
// 7, 8, 9 the element offsets of A, B and C in their buffers, and 10 = 1 to
// add the product to C (C's row stride is N). Accumulation starts each
// lane's registers from C rather than adding a finished partial, so a
// product split along K into calls with ascending K ranges rounds exactly as
// one call over the whole range would. Unset words are 0: no offsets,
// overwrite.
//
// A 16x16 workgroup owns a 64x64 tile of C; each lane owns a 4x4 block in four
// vec4 accumulators. K advances 16 at a time through shared tiles stored as
// vec4 along M (for A) and N (for B), so the inner loop is two vec4 loads and
// four vec4 multiply-adds. Every output sums k in ascending order. The entry
// point picks which axis consecutive lanes walk when filling the tiles, so the
// global loads stay coalesced for each layout.

@group(0) @binding(2) var<storage, read> ga: array<f32>;
@group(0) @binding(3) var<storage, read> gb: array<f32>;
@group(0) @binding(4) var<storage, read_write> gc: array<f32>;

var<workgroup> tile_a: array<vec4<f32>, 256>;
var<workgroup> tile_b: array<vec4<f32>, 256>;

fn gemm_tile(a_k_fast: bool, b_k_fast: bool, wg: vec3<u32>, lid: vec3<u32>) {
    let m_len = pw(0u);
    let n_len = pw(1u);
    let k_len = pw(2u);
    let a_rs = pw(3u);
    let a_cs = pw(4u);
    let b_rs = pw(5u);
    let b_cs = pw(6u);
    let row0 = wg.y * 64u;
    let col0 = wg.x * 64u;
    let lane = lid.y * 16u + lid.x;
    let col = col0 + lid.x * 4u;
    let row = row0 + lid.y * 4u;
    var acc0 = load_row(row, col, m_len, n_len);
    var acc1 = load_row(row + 1u, col, m_len, n_len);
    var acc2 = load_row(row + 2u, col, m_len, n_len);
    var acc3 = load_row(row + 3u, col, m_len, n_len);
    // Each lane stores one whole vec4 per tile: four rows of A (or four
    // columns of B) at one k. Lanes never write parts of the same vec4; a
    // dynamically indexed component store can lower to a read-modify-write
    // of the vector, which loses neighbouring lanes' values.
    var qa = lane >> 4u;
    var ka = lane & 15u;
    if (!a_k_fast) {
        qa = lane & 15u;
        ka = lane >> 4u;
    }
    var qb = lane >> 4u;
    var kb = lane & 15u;
    if (!b_k_fast) {
        qb = lane & 15u;
        kb = lane >> 4u;
    }
    for (var k0 = 0u; k0 < k_len; k0 = k0 + 16u) {
        let gka = k0 + ka;
        var av = vec4<f32>(0.0);
        if (gka < k_len) {
            let gr = row0 + qa * 4u;
            let base = pw(7u) + gr * a_rs + gka * a_cs;
            if (gr + 3u < m_len) {
                av = vec4<f32>(ga[base], ga[base + a_rs], ga[base + a_rs * 2u], ga[base + a_rs * 3u]);
            } else {
                if (gr < m_len) { av.x = ga[base]; }
                if (gr + 1u < m_len) { av.y = ga[base + a_rs]; }
                if (gr + 2u < m_len) { av.z = ga[base + a_rs * 2u]; }
            }
        }
        tile_a[ka * 16u + qa] = av;
        let gkb = k0 + kb;
        var bv = vec4<f32>(0.0);
        if (gkb < k_len) {
            let gcol = col0 + qb * 4u;
            let base = pw(8u) + gkb * b_rs + gcol * b_cs;
            if (gcol + 3u < n_len) {
                bv = vec4<f32>(gb[base], gb[base + b_cs], gb[base + b_cs * 2u], gb[base + b_cs * 3u]);
            } else {
                if (gcol < n_len) { bv.x = gb[base]; }
                if (gcol + 1u < n_len) { bv.y = gb[base + b_cs]; }
                if (gcol + 2u < n_len) { bv.z = gb[base + b_cs * 2u]; }
            }
        }
        tile_b[kb * 16u + qb] = bv;
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let a = tile_a[kk * 16u + lid.y];
            let b = tile_b[kk * 16u + lid.x];
            acc0 = acc0 + a.x * b;
            acc1 = acc1 + a.y * b;
            acc2 = acc2 + a.z * b;
            acc3 = acc3 + a.w * b;
        }
        workgroupBarrier();
    }
    store_row(row, col, m_len, n_len, acc0);
    store_row(row + 1u, col, m_len, n_len, acc1);
    store_row(row + 2u, col, m_len, n_len, acc2);
    store_row(row + 3u, col, m_len, n_len, acc3);
}

// The starting accumulator: C's values when accumulating, else zero.
fn load_row(row: u32, col: u32, m_len: u32, n_len: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    if (pw(10u) == 1u && row < m_len) {
        let base = pw(9u) + row * n_len + col;
        if (col + 3u < n_len) {
            return vec4<f32>(gc[base], gc[base + 1u], gc[base + 2u], gc[base + 3u]);
        }
        if (col < n_len) { v.x = gc[base]; }
        if (col + 1u < n_len) { v.y = gc[base + 1u]; }
        if (col + 2u < n_len) { v.z = gc[base + 2u]; }
    }
    return v;
}

fn store_row(row: u32, col: u32, m_len: u32, n_len: u32, v: vec4<f32>) {
    if (row >= m_len) {
        return;
    }
    let base = pw(9u) + row * n_len + col;
    if (col + 3u < n_len) {
        report(v.x);
        gc[base] = v.x;
        report(v.y);
        gc[base + 1u] = v.y;
        report(v.z);
        gc[base + 2u] = v.z;
        report(v.w);
        gc[base + 3u] = v.w;
        return;
    }
    if (col < n_len) {
        report(v.x);
        gc[base] = v.x;
    }
    if (col + 1u < n_len) {
        report(v.y);
        gc[base + 1u] = v.y;
    }
    if (col + 2u < n_len) {
        report(v.z);
        gc[base + 2u] = v.z;
    }
}

// x @ W^T: A is k-contiguous, B(k, n) = W[n, k] is k-contiguous.
@compute @workgroup_size(16, 16, 1)
fn gemm_nt(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    gemm_tile(true, true, wg, lid);
}

// gy @ W: A is k-contiguous, B is n-contiguous.
@compute @workgroup_size(16, 16, 1)
fn gemm_nn(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    gemm_tile(true, false, wg, lid);
}

// gy^T @ x: A(m, k) = gy[k, m] is m-contiguous, B is n-contiguous.
@compute @workgroup_size(16, 16, 1)
fn gemm_tn(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    gemm_tile(false, false, wg, lid);
}

// The same product on a 128x128 tile of C for matrices at least that large:
// each lane owns an 8x8 block (rows ty*4.. and 64+ty*4.., columns tx*4..
// and 64+tx*4..) in sixteen vec4 accumulators, so one k step is four vec4
// shared loads for sixteen vec4 multiply-adds instead of two for four. K
// advances 8 at a time through two shared buffers: the next tile is read
// from global memory into registers while the current one is multiplied,
// so a step needs one barrier. Every output still sums k in ascending order.

var<workgroup> big_a: array<vec4<f32>, 512>;
var<workgroup> big_b: array<vec4<f32>, 512>;

// One vec4 of a tile: four consecutive rows of A (or columns of B) at one k.
fn big_load_a(m_len: u32, k_len: u32, a_rs: u32, a_cs: u32, row0: u32, q: u32, k: u32) -> vec4<f32> {
    if (k >= k_len) {
        return vec4<f32>(0.0);
    }
    let gr = row0 + q * 4u;
    let base = pw(7u) + gr * a_rs + k * a_cs;
    if (gr + 3u < m_len) {
        return vec4<f32>(ga[base], ga[base + a_rs], ga[base + a_rs * 2u], ga[base + a_rs * 3u]);
    }
    var v = vec4<f32>(0.0);
    if (gr < m_len) { v.x = ga[base]; }
    if (gr + 1u < m_len) { v.y = ga[base + a_rs]; }
    if (gr + 2u < m_len) { v.z = ga[base + a_rs * 2u]; }
    return v;
}

fn big_load_b(n_len: u32, k_len: u32, b_rs: u32, b_cs: u32, col0: u32, q: u32, k: u32) -> vec4<f32> {
    if (k >= k_len) {
        return vec4<f32>(0.0);
    }
    let gc = col0 + q * 4u;
    let base = pw(8u) + k * b_rs + gc * b_cs;
    if (gc + 3u < n_len) {
        return vec4<f32>(gb[base], gb[base + b_cs], gb[base + b_cs * 2u], gb[base + b_cs * 3u]);
    }
    var v = vec4<f32>(0.0);
    if (gc < n_len) { v.x = gb[base]; }
    if (gc + 1u < n_len) { v.y = gb[base + b_cs]; }
    if (gc + 2u < n_len) { v.z = gb[base + b_cs * 2u]; }
    return v;
}

fn gemm_big(a_k_fast: bool, b_k_fast: bool, wg: vec3<u32>, lid: vec3<u32>) {
    let m_len = pw(0u);
    let n_len = pw(1u);
    let k_len = pw(2u);
    let a_rs = pw(3u);
    let a_cs = pw(4u);
    let b_rs = pw(5u);
    let b_cs = pw(6u);
    let row0 = wg.y * 128u;
    let col0 = wg.x * 128u;
    let lane = lid.y * 16u + lid.x;
    // Each lane loads one vec4 of A and one of B per k step: 32 vec4 along
    // M (or N) times 8 values of k. Consecutive lanes walk the axis that is
    // contiguous in memory.
    var qa = lane >> 3u;
    var ka = lane & 7u;
    if (!a_k_fast) {
        qa = lane & 31u;
        ka = lane >> 5u;
    }
    var qb = lane >> 3u;
    var kb = lane & 7u;
    if (!b_k_fast) {
        qb = lane & 31u;
        kb = lane >> 5u;
    }
    let r_lo = row0 + lid.y * 4u;
    let r_hi = row0 + 64u + lid.y * 4u;
    let c_lo = col0 + lid.x * 4u;
    let c_hi = col0 + 64u + lid.x * 4u;
    var acc0 = load_row(r_lo, c_lo, m_len, n_len);
    var acc1 = load_row(r_lo, c_hi, m_len, n_len);
    var acc2 = load_row(r_lo + 1u, c_lo, m_len, n_len);
    var acc3 = load_row(r_lo + 1u, c_hi, m_len, n_len);
    var acc4 = load_row(r_lo + 2u, c_lo, m_len, n_len);
    var acc5 = load_row(r_lo + 2u, c_hi, m_len, n_len);
    var acc6 = load_row(r_lo + 3u, c_lo, m_len, n_len);
    var acc7 = load_row(r_lo + 3u, c_hi, m_len, n_len);
    var acc8 = load_row(r_hi, c_lo, m_len, n_len);
    var acc9 = load_row(r_hi, c_hi, m_len, n_len);
    var acc10 = load_row(r_hi + 1u, c_lo, m_len, n_len);
    var acc11 = load_row(r_hi + 1u, c_hi, m_len, n_len);
    var acc12 = load_row(r_hi + 2u, c_lo, m_len, n_len);
    var acc13 = load_row(r_hi + 2u, c_hi, m_len, n_len);
    var acc14 = load_row(r_hi + 3u, c_lo, m_len, n_len);
    var acc15 = load_row(r_hi + 3u, c_hi, m_len, n_len);
    big_a[ka * 32u + qa] = big_load_a(m_len, k_len, a_rs, a_cs, row0, qa, ka);
    big_b[kb * 32u + qb] = big_load_b(n_len, k_len, b_rs, b_cs, col0, qb, kb);
    workgroupBarrier();
    var cur = 0u;
    for (var k0 = 0u; k0 < k_len; k0 = k0 + 8u) {
        let more = k0 + 8u < k_len;
        var na = vec4<f32>(0.0);
        var nb = vec4<f32>(0.0);
        if (more) {
            na = big_load_a(m_len, k_len, a_rs, a_cs, row0, qa, k0 + 8u + ka);
            nb = big_load_b(n_len, k_len, b_rs, b_cs, col0, qb, k0 + 8u + kb);
        }
        let base = cur * 256u;
        for (var kk = 0u; kk < 8u; kk = kk + 1u) {
            let a0 = big_a[base + kk * 32u + lid.y];
            let a1 = big_a[base + kk * 32u + 16u + lid.y];
            let b0 = big_b[base + kk * 32u + lid.x];
            let b1 = big_b[base + kk * 32u + 16u + lid.x];
            acc0 = acc0 + a0.x * b0;
            acc1 = acc1 + a0.x * b1;
            acc2 = acc2 + a0.y * b0;
            acc3 = acc3 + a0.y * b1;
            acc4 = acc4 + a0.z * b0;
            acc5 = acc5 + a0.z * b1;
            acc6 = acc6 + a0.w * b0;
            acc7 = acc7 + a0.w * b1;
            acc8 = acc8 + a1.x * b0;
            acc9 = acc9 + a1.x * b1;
            acc10 = acc10 + a1.y * b0;
            acc11 = acc11 + a1.y * b1;
            acc12 = acc12 + a1.z * b0;
            acc13 = acc13 + a1.z * b1;
            acc14 = acc14 + a1.w * b0;
            acc15 = acc15 + a1.w * b1;
        }
        if (more) {
            let nxt = (1u - cur) * 256u;
            big_a[nxt + ka * 32u + qa] = na;
            big_b[nxt + kb * 32u + qb] = nb;
        }
        workgroupBarrier();
        cur = 1u - cur;
    }
    store_row(r_lo, c_lo, m_len, n_len, acc0);
    store_row(r_lo, c_hi, m_len, n_len, acc1);
    store_row(r_lo + 1u, c_lo, m_len, n_len, acc2);
    store_row(r_lo + 1u, c_hi, m_len, n_len, acc3);
    store_row(r_lo + 2u, c_lo, m_len, n_len, acc4);
    store_row(r_lo + 2u, c_hi, m_len, n_len, acc5);
    store_row(r_lo + 3u, c_lo, m_len, n_len, acc6);
    store_row(r_lo + 3u, c_hi, m_len, n_len, acc7);
    store_row(r_hi, c_lo, m_len, n_len, acc8);
    store_row(r_hi, c_hi, m_len, n_len, acc9);
    store_row(r_hi + 1u, c_lo, m_len, n_len, acc10);
    store_row(r_hi + 1u, c_hi, m_len, n_len, acc11);
    store_row(r_hi + 2u, c_lo, m_len, n_len, acc12);
    store_row(r_hi + 2u, c_hi, m_len, n_len, acc13);
    store_row(r_hi + 3u, c_lo, m_len, n_len, acc14);
    store_row(r_hi + 3u, c_hi, m_len, n_len, acc15);
}

@compute @workgroup_size(16, 16, 1)
fn gemm_nt_big(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    gemm_big(true, true, wg, lid);
}

@compute @workgroup_size(16, 16, 1)
fn gemm_nn_big(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    gemm_big(true, false, wg, lid);
}

@compute @workgroup_size(16, 16, 1)
fn gemm_tn_big(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    gemm_big(false, false, wg, lid);
}
