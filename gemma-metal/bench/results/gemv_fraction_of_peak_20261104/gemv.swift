// Throwaway probe for ojas ft-5b2a72ee criteria 1-2 (nothing here ships):
//  1. the host's memory-bandwidth ceiling (a float4 read-reduce kernel and a float4 copy kernel);
//  2. tessl's gemv_q4_mlx_simd (the kernel gemma-metal's default E4B decode dispatches), compiled from
//     tessl/kernels into gemv.metallib, timed alone at E4B's shapes, MLX affine Q4, group 64.
// Weights are synthetic. Each timed batch walks distinct copies totalling >= 256 MB, so the SLC
// cannot hold them (decode streams ~2.86 GiB per token). Parity: an f64 CPU reference on copy 0.
// Usage: gemv <metallib> <rounds>
import Foundation
import Metal

let args = CommandLine.arguments
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let gemvLib = try! dev.makeLibrary(URL: URL(fileURLWithPath: args[1]))
let gemv = try! dev.makeComputePipelineState(function: gemvLib.makeFunction(name: "gemv_q4_mlx_simd")!)
let rounds = Int(args[2])!

let ceilSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void read4(device const float4 *a [[buffer(0)]], device float *out [[buffer(1)]],
                  constant uint &n4 [[buffer(2)]],
                  uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]],
                  uint tg [[threadgroup_position_in_grid]], uint sl [[thread_index_in_simdgroup]]) {
    float4 s = 0.0f;
    for (uint i = gid; i < n4; i += gsz) s += a[i];
    float t = simd_sum(s.x + s.y + s.z + s.w);
    if (sl == 0) out[gid / 32] = t;
}
kernel void copy4(device const float4 *a [[buffer(0)]], device float4 *b [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) { b[i] = a[i]; }
"""
let ceilLib = try! dev.makeLibrary(source: ceilSrc, options: nil)
let read4 = try! dev.makeComputePipelineState(function: ceilLib.makeFunction(name: "read4")!)
let copy4 = try! dev.makeComputePipelineState(function: ceilLib.makeFunction(name: "copy4")!)

/// Wall time (encode + commit + wait) of the last gpuMs call, in ms.
var lastWallMs = 0.0
func gpuMs(_ body: (MTLComputeCommandEncoder) -> Void) -> Double {
    let t0 = DispatchTime.now().uptimeNanoseconds
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    body(e)
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    lastWallMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    precondition(cb.status == .completed, "command buffer failed: \(String(describing: cb.error))")
    return (cb.gpuEndTime - cb.gpuStartTime) * 1000
}

func bf16(_ f: Float) -> UInt16 { UInt16(truncatingIfNeeded: f.bitPattern >> 16) }   // truncation, as a fixture
func f32(_ b: UInt16) -> Double { Double(Float(bitPattern: UInt32(b) << 16)) }

// ---- 1. bandwidth ceiling, 512 MiB
let ceilBytes = 512 << 20
let ca = dev.makeBuffer(length: ceilBytes, options: .storageModeShared)!
let cbuf = dev.makeBuffer(length: ceilBytes, options: .storageModeShared)!
memset(ca.contents(), 1, ceilBytes)
let readThreads = 20 * 1024 * 32   // enough simdgroups to fill the GPU; each strides the buffer
let rout = dev.makeBuffer(length: readThreads / 32 * 4, options: .storageModeShared)!
var n4 = UInt32(ceilBytes / 16)
func readRun() -> Double {
    gpuMs { e in
        e.setComputePipelineState(read4); e.setBuffer(ca, offset: 0, index: 0); e.setBuffer(rout, offset: 0, index: 1)
        e.setBytes(&n4, length: 4, index: 2)
        e.dispatchThreads(MTLSize(width: readThreads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
func copyRun() -> Double {
    gpuMs { e in
        e.setComputePipelineState(copy4); e.setBuffer(ca, offset: 0, index: 0); e.setBuffer(cbuf, offset: 0, index: 1)
        e.dispatchThreads(MTLSize(width: ceilBytes / 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

// ---- 2. tessl gemv_q4_mlx_simd
struct Shape { let name: String; let rows: Int; let cols: Int }
let G = 64
let shapes = [
    Shape(name: "mlp gate/up", rows: 10240, cols: 2560),
    Shape(name: "mlp down", rows: 2560, cols: 10240),
    Shape(name: "attn q (sliding)", rows: 2048, cols: 2560),
    Shape(name: "attn q (global)", rows: 4096, cols: 2560),
    Shape(name: "attn k or v (sliding)", rows: 512, cols: 2560),
    Shape(name: "attn o (sliding)", rows: 2560, cols: 2048),
    Shape(name: "lm_head (tied)", rows: 262144, cols: 2560),
]
struct Bank { let s: Shape; let copies: Int; let wBytes: Int; let sbBytes: Int
              let w: MTLBuffer; let sb: MTLBuffer; let x: MTLBuffer; let y: MTLBuffer
              let wHost: UnsafeMutablePointer<UInt8>; let sbHost: UnsafeMutablePointer<UInt16>; let xHost: [UInt16] }
var rng = SystemRandomNumberGenerator()
func makeBank(_ s: Shape) -> Bank {
    let wBytes = s.rows * s.cols / 2, sbBytes = s.rows * (s.cols / G) * 4
    let copies = max(1, Int((256 << 20) / (wBytes + sbBytes)) + 1)
    let w = dev.makeBuffer(length: wBytes * copies, options: .storageModeShared)!
    let sb = dev.makeBuffer(length: sbBytes * copies, options: .storageModeShared)!
    let wp = w.contents().bindMemory(to: UInt8.self, capacity: wBytes * copies)
    let w64 = w.contents().bindMemory(to: UInt64.self, capacity: wBytes * copies / 8)
    var st: UInt64 = 0x9E3779B97F4A7C15 &+ UInt64(s.rows * 31 + s.cols)
    for i in 0..<(wBytes * copies / 8) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; w64[i] = st }
    let sp = sb.contents().bindMemory(to: UInt16.self, capacity: sbBytes / 2 * copies)
    for i in 0..<(sbBytes / 2 * copies) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17
        let u = Float(st >> 40) / Float(1 << 24)   // [0, 1)
        sp[i] = bf16(i % 2 == 0 ? 0.001 + 0.019 * u : -0.1 + 0.2 * u)
    }
    let xh = (0..<s.cols).map { _ in bf16(Float.random(in: -1...1, using: &rng)) }
    let x = dev.makeBuffer(bytes: xh, length: s.cols * 2, options: .storageModeShared)!
    let y = dev.makeBuffer(length: s.rows * 4 * copies, options: .storageModeShared)!
    return Bank(s: s, copies: copies, wBytes: wBytes, sbBytes: sbBytes, w: w, sb: sb, x: x, y: y, wHost: wp, sbHost: sp, xHost: xh)
}
func gemvRun(_ b: Bank) -> Double {
    var rows = UInt32(b.s.rows), cols = UInt32(b.s.cols), gs = UInt32(G)
    let total = gpuMs { e in
        e.setComputePipelineState(gemv)
        e.setBuffer(b.x, offset: 0, index: 3)
        e.setBytes(&rows, length: 4, index: 5); e.setBytes(&cols, length: 4, index: 6); e.setBytes(&gs, length: 4, index: 7)
        for c in 0..<b.copies {
            e.setBuffer(b.w, offset: c * b.wBytes, index: 0)
            e.setBuffer(b.sb, offset: c * b.sbBytes, index: 1)
            e.setBuffer(b.sb, offset: c * b.sbBytes, index: 2)
            e.setBuffer(b.y, offset: c * b.s.rows * 4, index: 4)
            e.dispatchThreadgroups(MTLSize(width: (b.s.rows + 7) / 8, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
    }
    wallPerCall = lastWallMs / Double(b.copies)
    return total / Double(b.copies)
}
var wallPerCall = 0.0
func parity(_ b: Bank) -> Double {
    let y = b.y.contents().bindMemory(to: Float.self, capacity: b.s.rows)
    let gpr = b.s.cols / G
    var worst = 0.0
    for r in stride(from: 0, to: b.s.rows, by: max(1, b.s.rows / 256)) {
        var acc = 0.0
        for g in 0..<gpr {
            let scale = f32(b.sbHost[(r * gpr + g) * 2]), bias = f32(b.sbHost[(r * gpr + g) * 2 + 1])
            var qx = 0.0, xs = 0.0
            for j in 0..<G {
                let k = g * G + j
                let byte = b.wHost[r * b.s.cols / 2 + k / 2]
                let qv = Double(k % 2 == 0 ? byte & 0x0f : byte >> 4)
                let xv = f32(b.xHost[k])
                qx += qv * xv; xs += xv
            }
            acc += scale * qx + bias * xs
        }
        worst = max(worst, abs(Double(y[r]) - acc) / max(1e-3, abs(acc)))
    }
    return worst
}

let banks = shapes.map(makeBank)
_ = readRun(); _ = copyRun(); for b in banks { _ = gemvRun(b) }
var readT: [Double] = [], copyT: [Double] = []
var gemvT = [[Double]](repeating: [], count: banks.count)
var gemvW = [[Double]](repeating: [], count: banks.count)
for _ in 0..<rounds {
    readT.append(readRun()); copyT.append(copyRun())
    for (i, b) in banks.enumerated() { gemvT[i].append(gemvRun(b)); gemvW[i].append(wallPerCall) }
}
func stats(_ v: [Double]) -> (Double, Double, Double) { let s = v.sorted(); return (s[0], s[s.count / 2], s.last!) }
let (rmin, rmed, _) = stats(readT), (cmin, cmed, _) = stats(copyT)
let readGBs = Double(ceilBytes) / (rmin / 1000) / 1e9, copyGBs = 2 * Double(ceilBytes) / (cmin / 1000) / 1e9
print(String(format: "ceiling: read %.1f GB/s (min %.3f ms, median %.3f ms), copy %.1f GB/s read+write (min %.3f ms, median %.3f ms), %d rounds",
             readGBs, rmin, rmed, copyGBs, cmin, cmed, rounds))
print("shape | rows x cols | bytes (w + sb) | copies/batch | GPU min us | GPU median us | GPU max us | GB/s at GPU min | of read ceiling | wall min us | GB/s at wall min | parity max rel")
for (i, b) in banks.enumerated() {
    let (mn, md, mx) = stats(gemvT[i])
    let (wmn, _, _) = stats(gemvW[i])
    let bytes = Double(b.wBytes + b.sbBytes)
    let gbs = bytes / (mn / 1000) / 1e9
    print(String(format: "%@ | %d x %d | %.2f MB | %d | %.1f | %.1f | %.1f | %.1f | %.0f%% | %.1f | %.1f | %.2e",
                 b.s.name, b.s.rows, b.s.cols, bytes / 1e6, b.copies, mn * 1000, md * 1000, mx * 1000, gbs, 100 * gbs / readGBs,
                 wmn * 1000, bytes / (wmn / 1000) / 1e9, parity(b)))
}
