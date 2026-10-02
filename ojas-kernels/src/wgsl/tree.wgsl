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

// The sum over each aligned segment of `width` lanes (a power of two, at most
// 256), returned to every lane of the segment: 256 / width independent rows
// per workgroup. The pairing is tree_sum's restricted to the segment, so with
// width 256 it is tree_sum, and a narrower width gives the same bits as
// tree_sum would over the segment's values with zeros in the other lanes.
// `width` must be the same for every lane.
fn seg_sum(lane: u32, width: u32, value: f32) -> f32 {
    red[lane] = value;
    workgroupBarrier();
    let at = lane & (width - 1u);
    for (var s = width >> 1u; s > 0u; s = s >> 1u) {
        if (at < s) {
            red[lane] = red[lane] + red[lane + s];
        }
        workgroupBarrier();
    }
    let total = red[lane - at];
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
