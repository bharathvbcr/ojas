// C[M, N] = A[M, K] @ B[K, N] with element strides, so one kernel covers the
// NN, NT and TN products a linear needs and nothing is transposed on the host.
//
// A(m, k) = ga[m * a_rs + k * a_cs], B(k, n) = gb[k * b_rs + n * b_cs], C is
// row-major. Words: 0 M, 1 N, 2 K, 3 a_rs, 4 a_cs, 5 b_rs, 6 b_cs.
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
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
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
        for (var j = 0u; j < 4u; j = j + 1u) {
            let gr = row0 + qa * 4u + j;
            if (gr < m_len && gka < k_len) {
                av[j] = ga[gr * a_rs + gka * a_cs];
            }
        }
        tile_a[ka * 16u + qa] = av;
        let gkb = k0 + kb;
        var bv = vec4<f32>(0.0);
        for (var j = 0u; j < 4u; j = j + 1u) {
            let gcol = col0 + qb * 4u + j;
            if (gcol < n_len && gkb < k_len) {
                bv[j] = gb[gkb * b_rs + gcol * b_cs];
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
    let col = col0 + lid.x * 4u;
    let row = row0 + lid.y * 4u;
    store_row(row, col, m_len, n_len, acc0);
    store_row(row + 1u, col, m_len, n_len, acc1);
    store_row(row + 2u, col, m_len, n_len, acc2);
    store_row(row + 3u, col, m_len, n_len, acc3);
}

fn store_row(row: u32, col: u32, m_len: u32, n_len: u32, v: vec4<f32>) {
    if (row >= m_len) {
        return;
    }
    for (var j = 0u; j < 4u; j = j + 1u) {
        if (col + j < n_len) {
            let value = v[j];
            report(value);
            gc[row * n_len + col + j] = value;
        }
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
    var v = vec4<f32>(0.0);
    if (k < k_len) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let gr = row0 + q * 4u + j;
            if (gr < m_len) {
                v[j] = ga[gr * a_rs + k * a_cs];
            }
        }
    }
    return v;
}

fn big_load_b(n_len: u32, k_len: u32, b_rs: u32, b_cs: u32, col0: u32, q: u32, k: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    if (k < k_len) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let gc = col0 + q * 4u + j;
            if (gc < n_len) {
                v[j] = gb[k * b_rs + gc * b_cs];
            }
        }
    }
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
    var acc: array<vec4<f32>, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        acc[i] = vec4<f32>(0.0);
    }
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
            acc[0] = acc[0] + a0.x * b0;
            acc[1] = acc[1] + a0.x * b1;
            acc[2] = acc[2] + a0.y * b0;
            acc[3] = acc[3] + a0.y * b1;
            acc[4] = acc[4] + a0.z * b0;
            acc[5] = acc[5] + a0.z * b1;
            acc[6] = acc[6] + a0.w * b0;
            acc[7] = acc[7] + a0.w * b1;
            acc[8] = acc[8] + a1.x * b0;
            acc[9] = acc[9] + a1.x * b1;
            acc[10] = acc[10] + a1.y * b0;
            acc[11] = acc[11] + a1.y * b1;
            acc[12] = acc[12] + a1.z * b0;
            acc[13] = acc[13] + a1.z * b1;
            acc[14] = acc[14] + a1.w * b0;
            acc[15] = acc[15] + a1.w * b1;
        }
        if (more) {
            let nxt = (1u - cur) * 256u;
            big_a[nxt + ka * 32u + qa] = na;
            big_b[nxt + kb * 32u + qb] = nb;
        }
        workgroupBarrier();
        cur = 1u - cur;
    }
    // Rows lid.y*4 + j and 64 + lid.y*4 + j; columns lid.x*4.. and 64 + lid.x*4..
    for (var h = 0u; h < 2u; h = h + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let row = row0 + h * 64u + lid.y * 4u + j;
            let i = h * 8u + j * 2u;
            store_row(row, col0 + lid.x * 4u, m_len, n_len, acc[i]);
            store_row(row, col0 + 64u + lid.x * 4u, m_len, n_len, acc[i + 1u]);
        }
    }
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
