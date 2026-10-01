// Fixed-shape 256-lane tree reductions. The pairing depends only on the lane
// index, so a reduction over the same inputs gives the same bits every run.
// Call from uniform control flow: every lane of the workgroup must arrive.

var<workgroup> red: array<f32, 256>;

fn tree_sum(lane: u32, value: f32) -> f32 {
    red[lane] = value;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            red[lane] = red[lane] + red[lane + s];
        }
        workgroupBarrier();
    }
    let total = red[0];
    workgroupBarrier();
    return total;
}

fn tree_max(lane: u32, value: f32) -> f32 {
    red[lane] = value;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            red[lane] = max(red[lane], red[lane + s]);
        }
        workgroupBarrier();
    }
    let peak = red[0];
    workgroupBarrier();
    return peak;
}
