use std::any::Any;
use std::sync::Arc;

use ojas_cluster::cpu::{
    decode_bf16_bytes_to_scratch, decode_bf16_row_checked, decode_bf16_to_scratch, MatrixData,
    MatrixView,
};
use ojas_cluster::{
    assign_cpu_bf16, assign_cpu_f32, assign_cpu_tensor, fit_cpu, lloyd_fit, CpuMath, CpuPool,
    CpuSession, EmeansConfig, LloydConfig, SplitMix64, UNASSIGNED_LABEL,
};
use ojas_core::{
    BackendId, Budget, DType, DeviceBuffer, FenceError, HostLease, HostLeaseMut, MappedBuffer,
    OjasError, Tensor, Timeline, CPU_THREAD_CEILING,
};

#[derive(Debug)]
struct MockTimeline;
impl Timeline for MockTimeline {
    fn submitted(&self) -> Result<u64, FenceError> {
        Ok(1)
    }
    fn completed(&self) -> Result<u64, FenceError> {
        Ok(1)
    }
    fn wait(&self, _value: u64) -> Result<(), FenceError> {
        Ok(())
    }
}

struct MockLease {
    bytes: Vec<u8>,
    f32s: Option<Vec<f32>>,
}

impl HostLease for MockLease {
    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    fn as_f32(&self) -> Result<&[f32], OjasError> {
        self.f32s.as_deref().ok_or_else(|| OjasError::Unsupported {
            op: "MockLease::as_f32",
            detail: "not f32".to_string(),
        })
    }
}

#[derive(Debug)]
struct MockMappedBuffer {
    bytes: Vec<u8>,
    timeline: MockTimeline,
}

impl DeviceBuffer for MockMappedBuffer {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }
    fn byte_len(&self) -> usize {
        self.bytes.len()
    }
    fn read_bytes(&self, _offset: usize, _len: usize) -> Result<Vec<u8>, OjasError> {
        panic!("MappedBuffer must use host_view_window(), never DeviceBuffer::read_bytes!");
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MappedBuffer for MockMappedBuffer {
    fn host_ptr(&self) -> *mut u8 {
        std::ptr::null_mut()
    }
    fn gpu_address(&self) -> u64 {
        0
    }
    fn timeline(&self) -> &dyn Timeline {
        &self.timeline
    }
    fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> {
        let f32s = if self.bytes.len().is_multiple_of(4) {
            Some(
                self.bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_ne_bytes(*b))
                    .collect(),
            )
        } else {
            None
        };
        Ok(Box::new(MockLease {
            bytes: self.bytes.clone(),
            f32s,
        }))
    }
    fn host_write(&self) -> Result<Box<dyn HostLeaseMut>, FenceError> {
        Err(FenceError::Poisoned)
    }
}

#[derive(Debug)]
struct MockDiscreteDevice {
    byte_len: usize,
}

impl DeviceBuffer for MockDiscreteDevice {
    fn backend(&self) -> BackendId {
        BackendId::Cuda
    }
    fn byte_len(&self) -> usize {
        self.byte_len
    }
    fn read_bytes(&self, _offset: usize, _len: usize) -> Result<Vec<u8>, OjasError> {
        Ok(vec![0; self.byte_len])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn f32_to_bf16_rte_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let mag = bits & 0x7fff_ffff;
    if mag > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    if mag < 0x0080_0000 {
        return ((bits >> 16) & 0x8000) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn standard_normal_pair(rng: &mut SplitMix64) -> (f64, f64) {
    let u1_raw = ((rng.next_u64() >> 11) | 1) as f64 * (1.0 / ((1u64 << 53) as f64));
    let u2_raw = (rng.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
    let r = (-2.0 * u1_raw.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2_raw;
    (r * theta.cos(), r * theta.sin())
}

#[test]
fn test_cpu_thread_count_bit_identity_and_ceiling_refusal() {
    // 1. Thread ceiling & zero-thread refusals
    assert!(matches!(
        CpuPool::new(Some(0)),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));
    assert!(matches!(
        CpuPool::new(Some(CPU_THREAD_CEILING as usize + 1)),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));

    let auto_pool = CpuPool::new(None).expect("auto CpuPool");
    assert!(!auto_pool.is_explicit_threads());
    assert!(auto_pool.threads() >= 1 && auto_pool.threads() <= CPU_THREAD_CEILING as usize);

    let pinned_pool = CpuPool::new(Some(2)).expect("pinned CpuPool");
    assert!(pinned_pool.is_explicit_threads());
    assert_eq!(pinned_pool.threads(), 2);

    // 2. Build a 4,608-row (>= 4,096 parallel CSR hist threshold) non-aligned dataset (`d = 37, k = 48`)
    //    with heavy Zipf skew into cluster 0 AND dead initial centroids so split revival also runs.
    let n = 4_608_usize;
    let d = 37_usize;
    let k = 48_usize;
    let budget = Budget::new(128 * 1024 * 1024);

    let mut rng = SplitMix64::new(0x5eed_cafe_1234_5678);
    let mut rows = vec![0.0_f32; n * d];
    for r in 0..n {
        let base_cluster = if r < 2_400 { 0 } else { 1 + (r % 31) };
        for col in 0..d {
            let noise = ((rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32) - 0.5;
            rows[r * d + col] = if col == (base_cluster % d) {
                (base_cluster as f32) * 3.0 + noise * 0.25
            } else {
                noise * 0.25
            };
        }
    }

    // Initial centroids: 32 live clusters + 16 far-away dead clusters at +500.0 to force revival at step 4
    let mut init_c = vec![0.0_f32; k * d];
    for j in 0..k {
        if j < 32 {
            init_c[j * d..(j + 1) * d].copy_from_slice(&rows[(j * 70) * d..(j * 70 + 1) * d]);
        } else {
            init_c[j * d..(j + 1) * d].fill(500.0 + j as f32);
        }
    }

    let x_tensor = Tensor::from_f32(&rows, &[n, d], &budget).expect("x_tensor");
    let init_tensor = Tensor::from_f32(&init_c, &[k, d], &budget).expect("init_tensor");

    let mut cfg = EmeansConfig::new(k);
    cfg.nepochs = 1.6;
    cfg.batch_size = Some(4_608); // chunk_rows == 4,608 >= 4,096 exercises parallel CSR hist!
    cfg.assignment_chunk_size = 4_608;
    cfg.seed = 777;

    // Also run a multi-step config (6 steps >= 5) with chunk_rows >= 4096 on assign and revival on fit
    let mut cfg_revive = EmeansConfig::new(k);
    cfg_revive.nepochs = 2.0;
    cfg_revive.batch_size = Some(768); // 6 steps/epoch -> 12 total steps -> revivals at steps 4 & 9
    cfg_revive.assignment_chunk_size = 4_608;
    cfg_revive.seed = 888;

    let rows_bf16: Vec<u16> = rows.iter().copied().map(f32_to_bf16_rte_bits).collect();

    for math in [CpuMath::DeterministicFma, CpuMath::Exact] {
        let pool_1 = CpuPool::new(Some(1)).expect("pool 1");
        let ref_big_chunk = fit_cpu(&pool_1, &x_tensor, Some(&init_tensor), &cfg, math, &budget)
            .expect("fit_cpu t=1 big chunk");
        let ref_revive = fit_cpu(
            &pool_1,
            &x_tensor,
            Some(&init_tensor),
            &cfg_revive,
            math,
            &budget,
        )
        .expect("fit_cpu t=1 revive");
        assert!(
            ref_revive
                .epoch_stats
                .iter()
                .map(|s| s.revived_clusters)
                .sum::<usize>()
                > 0,
            "dead clusters at +500 must trigger Gumbel split revival ({math:?})"
        );

        let ref_c1 = ref_big_chunk.centroids.f32_slice().unwrap().to_vec();
        let ref_c2 = ref_revive.centroids.f32_slice().unwrap().to_vec();
        let (ref_assign_lbl, ref_assign_obj) = assign_cpu_f32(
            &pool_1, &rows, n, d, &ref_c2, k, 4_608, math, false, &budget,
        )
        .expect("assign_cpu_f32 t=1");
        let (ref_bf16_lbl, ref_bf16_obj) = assign_cpu_bf16(
            &pool_1, &rows_bf16, n, d, &ref_c2, k, 4_608, math, false, &budget,
        )
        .expect("assign_cpu_bf16 t=1");

        let mut lloyd_cfg_1 = LloydConfig::new(k);
        lloyd_cfg_1.max_iters = 4;
        lloyd_cfg_1.tol = 0.0;
        lloyd_cfg_1.threads = Some(1);
        lloyd_cfg_1.math = math;
        lloyd_cfg_1.assignment_chunk_size = 4_608;
        let ref_lloyd =
            lloyd_fit(&x_tensor, Some(&init_tensor), &lloyd_cfg_1, &budget).expect("lloyd_fit t=1");
        let ref_lloyd_c = ref_lloyd.centroids.f32_slice().unwrap().to_vec();

        for threads in [2_usize, 4, 8] {
            let pool_t = CpuPool::new(Some(threads)).expect("pool_t");

            // Reuse the SAME persistent `pool_t` across multiple calls to keep thread-local
            // `sgemm_tile` packing buffers warm and verify zero state bleed!
            for _round in 0..2 {
                let got_big = fit_cpu(&pool_t, &x_tensor, Some(&init_tensor), &cfg, math, &budget)
                    .expect("fit_cpu big chunk");
                let got_c1 = got_big.centroids.f32_slice().unwrap();
                for idx in 0..(k * d) {
                    assert_eq!(
                        got_c1[idx].to_bits(),
                        ref_c1[idx].to_bits(),
                        "big-chunk centroid bit mismatch at {idx} ({math:?}, threads={threads})"
                    );
                }
                assert_eq!(got_big.labels, ref_big_chunk.labels);
                assert_eq!(got_big.cluster_counts, ref_big_chunk.cluster_counts);
                for j in 0..k {
                    assert_eq!(got_big.usage[j].to_bits(), ref_big_chunk.usage[j].to_bits());
                }
                for (es_got, es_ref) in got_big.epoch_stats.iter().zip(&ref_big_chunk.epoch_stats) {
                    assert_eq!(
                        es_got.mean_objective.to_bits(),
                        es_ref.mean_objective.to_bits(),
                        "big-chunk objective bit mismatch ({math:?}, threads={threads})"
                    );
                    assert_eq!(es_got.revived_clusters, es_ref.revived_clusters);
                }

                let got_rev = fit_cpu(
                    &pool_t,
                    &x_tensor,
                    Some(&init_tensor),
                    &cfg_revive,
                    math,
                    &budget,
                )
                .expect("fit_cpu revive");
                let got_c2 = got_rev.centroids.f32_slice().unwrap();
                for idx in 0..(k * d) {
                    assert_eq!(
                        got_c2[idx].to_bits(),
                        ref_c2[idx].to_bits(),
                        "revive centroid bit mismatch at {idx} ({math:?}, threads={threads})"
                    );
                }
                assert_eq!(got_rev.labels, ref_revive.labels);
                assert_eq!(got_rev.cluster_counts, ref_revive.cluster_counts);
                for (es_got, es_ref) in got_rev.epoch_stats.iter().zip(&ref_revive.epoch_stats) {
                    assert_eq!(
                        es_got.mean_objective.to_bits(),
                        es_ref.mean_objective.to_bits(),
                        "revive objective bit mismatch ({math:?}, threads={threads})"
                    );
                    assert_eq!(es_got.revived_clusters, es_ref.revived_clusters);
                }

                let (got_al, got_ao) = assign_cpu_f32(
                    &pool_t, &rows, n, d, &ref_c2, k, 4_608, math, false, &budget,
                )
                .expect("assign_cpu_f32");
                assert_eq!(got_al, ref_assign_lbl);
                assert_eq!(got_ao.to_bits(), ref_assign_obj.to_bits());

                let (got_bl, got_bo) = assign_cpu_bf16(
                    &pool_t, &rows_bf16, n, d, &ref_c2, k, 4_608, math, false, &budget,
                )
                .expect("assign_cpu_bf16");
                assert_eq!(got_bl, ref_bf16_lbl);
                assert_eq!(got_bo.to_bits(), ref_bf16_obj.to_bits());
            }

            let mut lloyd_cfg_t = lloyd_cfg_1.clone();
            lloyd_cfg_t.threads = Some(threads);
            let got_lloyd = lloyd_fit(&x_tensor, Some(&init_tensor), &lloyd_cfg_t, &budget)
                .expect("lloyd_fit threads");
            let got_lloyd_c = got_lloyd.centroids.f32_slice().unwrap();
            for idx in 0..(k * d) {
                assert_eq!(
                    got_lloyd_c[idx].to_bits(),
                    ref_lloyd_c[idx].to_bits(),
                    "lloyd centroid bit mismatch at {idx} ({math:?}, threads={threads})"
                );
            }
            assert_eq!(got_lloyd.labels, ref_lloyd.labels);
            assert_eq!(got_lloyd.cluster_counts, ref_lloyd.cluster_counts);
            for (o_got, o_ref) in got_lloyd
                .iter_objectives
                .iter()
                .zip(&ref_lloyd.iter_objectives)
            {
                assert_eq!(o_got.to_bits(), o_ref.to_bits());
            }
        }
    }

    // Lloyd also refuses invalid thread ceilings
    let mut bad_lloyd = LloydConfig::new(4);
    bad_lloyd.threads = Some(0);
    assert!(matches!(
        lloyd_fit(&x_tensor, None, &bad_lloyd, &budget),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));
    bad_lloyd.threads = Some(CPU_THREAD_CEILING as usize + 1);
    assert!(matches!(
        lloyd_fit(&x_tensor, None, &bad_lloyd, &budget),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));
}

#[test]
fn test_cpu_fast_pinned_threads_and_accelerate_panel() {
    // Exercise widened `[2048, min(k, 4096)]` `sgemm_accelerate` panels (`n = 4_500 > 2 * 2048`)
    // when `threads: None`, AND verify `threads: Some(2)` routes `CpuMath::Fast` through
    // `CpuPool` + `sgemm_tile` (matching `DeterministicFma` bit-for-bit!).
    let n = 4_500_usize;
    let d = 64_usize;
    let k = 96_usize;
    let budget = Budget::new(128 * 1024 * 1024);

    let mut rng = SplitMix64::new(0x1122_3344_5566_7788);
    let mut rows = vec![0.0_f32; n * d];
    for r in 0..n {
        let c = r % k;
        for col in 0..d {
            let u = ((rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32) - 0.5;
            rows[r * d + col] = if col == (c % d) {
                4.0 + u * 0.2
            } else {
                u * 0.2
            };
        }
    }
    let x_f32 = Tensor::from_f32(&rows, &[n, d], &budget).expect("x_f32");
    let bf16_bits: Vec<u16> = rows.iter().copied().map(f32_to_bf16_rte_bits).collect();
    let x_bf16 = Tensor::from_bf16_bits(&bf16_bits, &[n, d], &budget).expect("x_bf16");

    let mut cfg = EmeansConfig::new(k);
    cfg.nepochs = 1.5;
    cfg.assignment_chunk_size = 4_500; // spans 3 row panels: 2048 + 2048 + 404!
    cfg.seed = 1234;

    // 1. Pinned `threads: Some(2)` with `CpuMath::Fast` MUST bypass Accelerate and match
    //    `CpuMath::DeterministicFma` bit-for-bit!
    let pool_pinned = CpuPool::new(Some(2)).expect("pool_pinned");
    assert!(pool_pinned.is_explicit_threads());
    let fit_fast_pinned =
        fit_cpu(&pool_pinned, &x_f32, None, &cfg, CpuMath::Fast, &budget).expect("fast pinned");
    let fit_det = fit_cpu(
        &pool_pinned,
        &x_f32,
        None,
        &cfg,
        CpuMath::DeterministicFma,
        &budget,
    )
    .expect("det pinned");

    let c_fast_pin = fit_fast_pinned.centroids.f32_slice().unwrap();
    let c_det = fit_det.centroids.f32_slice().unwrap();
    for i in 0..(k * d) {
        assert_eq!(
            c_fast_pin[i].to_bits(),
            c_det[i].to_bits(),
            "CpuMath::Fast with explicit Some(2) must route through CpuPool + sgemm_tile"
        );
    }
    assert_eq!(fit_fast_pinned.labels, fit_det.labels);

    // 2. Unpinned `threads: None` with `CpuMath::Fast` (exercises `sgemm_accelerate` widened panel on macOS)
    let pool_auto = CpuPool::new(None).expect("pool_auto");
    assert!(!pool_auto.is_explicit_threads());
    let fit_accel =
        fit_cpu(&pool_auto, &x_f32, None, &cfg, CpuMath::Fast, &budget).expect("fast auto");
    let c_accel = fit_accel.centroids.f32_slice().unwrap();
    assert_eq!(fit_accel.centroids.shape(), &[k, d]);
    assert_eq!(fit_accel.labels.len(), n);
    assert_eq!(fit_accel.cluster_counts.iter().sum::<u64>(), n as u64);

    let (_, mse_det_total) = assign_cpu_f32(
        &pool_pinned,
        &rows,
        n,
        d,
        c_det,
        k,
        4_500,
        CpuMath::Exact,
        false,
        &budget,
    )
    .unwrap();
    let (lbl_accel, mse_accel_total) = assign_cpu_f32(
        &pool_auto,
        &rows,
        n,
        d,
        c_accel,
        k,
        4_500,
        CpuMath::Fast,
        false,
        &budget,
    )
    .unwrap();
    assert_eq!(lbl_accel.len(), n);
    let rel_diff = (mse_accel_total - mse_det_total).abs() / mse_det_total.max(1.0);
    assert!(
        rel_diff < 0.02,
        "Fast auto MSE ({mse_accel_total}) and DeterministicFma MSE ({mse_det_total}) diverged: rel={rel_diff}"
    );

    // 3. `Bf16` input Tensor on `fit_cpu` and `lloyd_fit` + `max_points_per_centroid` + `capture_assignments = false`
    let mut cfg_sub = EmeansConfig::new(16);
    cfg_sub.max_points_per_centroid = Some(16); // effective_n = 256 < 4,500
    cfg_sub.nepochs = 1.0;
    let fit_bf16_sub = fit_cpu(&pool_auto, &x_bf16, None, &cfg_sub, CpuMath::Fast, &budget)
        .expect("fit_cpu bf16 subsampled");
    assert_eq!(fit_bf16_sub.labels.len(), n);
    let assigned_cnt = fit_bf16_sub
        .labels
        .iter()
        .filter(|&&l| l != UNASSIGNED_LABEL)
        .count();
    assert_eq!(assigned_cnt, 256);
    assert_eq!(fit_bf16_sub.cluster_counts.iter().sum::<u64>(), 256);

    cfg_sub.capture_assignments = false;
    let fit_no_lbl =
        fit_cpu(&pool_auto, &x_bf16, None, &cfg_sub, CpuMath::Fast, &budget).expect("no labels");
    assert!(fit_no_lbl.labels.is_empty());

    let mut lloyd_bf16_cfg = LloydConfig::new(16);
    lloyd_bf16_cfg.max_iters = 3;
    let lloyd_bf16 = lloyd_fit(&x_bf16, None, &lloyd_bf16_cfg, &budget).expect("lloyd bf16");
    assert_eq!(lloyd_bf16.labels.len(), n);
    assert_eq!(lloyd_bf16.cluster_counts.iter().sum::<u64>(), n as u64);

    // Verify MatrixView on Host F32 and Host Bf16
    let v_f32 = MatrixView::open(&x_f32, "emeans").expect("view f32");
    assert_eq!((v_f32.rows(), v_f32.cols()), (n, d));
    assert!(matches!(v_f32.data(), MatrixData::F32(s) if s.len() == n * d));

    let v_bf16 = MatrixView::open(&x_bf16, "emeans").expect("view bf16");
    assert_eq!((v_bf16.rows(), v_bf16.cols()), (n, d));
    assert!(matches!(v_bf16.data(), MatrixData::Bf16(s) if s.len() == n * d));
}

#[test]
fn test_spherical_prescaling_zero_rows_and_unassigned_closing_on_cpu() {
    let d = 16_usize;
    let k = 3_usize;
    let budget = Budget::new(16 * 1024 * 1024);
    let pool = CpuPool::new(Some(4)).expect("pool");

    // Extreme centroids: 1e-25 (f32 square underflows to 0 without prescaling),
    // 1e25 (f32 square overflows to Inf without prescaling), and exact [0.0; 16]!
    let mut init_centroids = vec![0.0_f32; k * d];
    for col in 0..d {
        init_centroids[col] = if col % 2 == 0 { 1.0e-25 } else { -1.0e-25 };
        init_centroids[d + col] = 1.0e25;
        init_centroids[2 * d + col] = 0.0;
    }

    // 32 rows aligned with cluster 0, 32 rows aligned with cluster 1, and 2 exact zero rows!
    let mut rows = Vec::with_capacity(66 * d);
    for _ in 0..32 {
        for col in 0..d {
            rows.push(if col % 2 == 0 { 0.25 } else { -0.25 }); // unit norm: sqrt(16 * 1/16) = 1.0
        }
    }
    rows.extend(std::iter::repeat_n(0.25_f32, 32 * d));
    rows.extend(std::iter::repeat_n(0.0_f32, 2 * d));
    let n = 66_usize;

    for math in [CpuMath::Fast, CpuMath::DeterministicFma, CpuMath::Exact] {
        let (labels, obj) = assign_cpu_f32(
            &pool,
            &rows,
            n,
            d,
            &init_centroids,
            k,
            32,
            math,
            true,
            &budget,
        )
        .expect("spherical assign_cpu_f32");
        assert!(obj.is_finite(), "spherical obj must be finite, got {obj}");
        assert!(labels[..32].iter().all(|&l| l == 0));
        assert!(labels[32..64].iter().all(|&l| l == 1));
        assert_eq!(labels[64], 0);
        assert_eq!(labels[65], 0);
        assert!(
            (obj - 4.0).abs() < 1e-4,
            "64 exact unit matches (0.0) + 2 zero rows (2.0 each) must total ~4.0, got {obj}"
        );
    }

    // Fit with `unused_threshold = Some(0.0)` so cluster 2 stays unassigned (`pass_counts[2] == 0`)
    // and retains `[0.0; 16]` through `f64` spherical closing without producing `NaN`!
    let x_tensor = Tensor::from_f32(&rows, &[n, d], &budget).unwrap();
    let init_tensor = Tensor::from_f32(&init_centroids, &[k, d], &budget).unwrap();
    let mut cfg = EmeansConfig::new(k);
    cfg.spherical = true;
    cfg.nepochs = 1.0;
    cfg.batch_size = Some(66);
    cfg.unused_threshold = Some(0.0);

    let fit = fit_cpu(
        &pool,
        &x_tensor,
        Some(&init_tensor),
        &cfg,
        CpuMath::DeterministicFma,
        &budget,
    )
    .expect("spherical fit_cpu");
    assert_eq!(fit.cluster_counts[2], 0);
    let closing = fit.centroids.f32_slice().unwrap();
    assert!(closing.iter().all(|v| v.is_finite()), "closing has NaN/Inf");
    for j in 0..2 {
        let norm = closing[j * d..(j + 1) * d]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "non-zero spherical closing centroid {j} must have unit norm, got {norm}"
        );
    }
    assert!(
        closing[2 * d..3 * d].iter().all(|&v| v == 0.0),
        "unassigned zero spherical centroid must remain exact 0.0"
    );

    // Also test unassigned non-zero centroid in spherical closing (retains normalized direction)
    let mut init_unassigned_nonzero = init_centroids.clone();
    init_unassigned_nonzero[2 * d..3 * d].fill(-1.0e25);
    let init_nz_tensor = Tensor::from_f32(&init_unassigned_nonzero, &[k, d], &budget).unwrap();
    // Make rows positive so cluster 2 at [-0.25; 16] stays unassigned (`pass_counts[2] == 0`)
    let fit_nz = fit_cpu(
        &pool,
        &x_tensor,
        Some(&init_nz_tensor),
        &cfg,
        CpuMath::Exact,
        &budget,
    )
    .expect("spherical unassigned nonzero");
    assert_eq!(fit_nz.cluster_counts[2], 0);
    let c2_norm = fit_nz.centroids.f32_slice().unwrap()[2 * d..3 * d]
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    assert!((c2_norm - 1.0).abs() < 1e-5, "got {c2_norm}");

    // Also test spherical `lloyd_fit` on the same extreme centroids + zero rows + unassigned cluster
    let mut lloyd_sph = LloydConfig::new(k);
    lloyd_sph.spherical = true;
    lloyd_sph.max_iters = 5;
    lloyd_sph.math = CpuMath::DeterministicFma;
    let lfit =
        lloyd_fit(&x_tensor, Some(&init_tensor), &lloyd_sph, &budget).expect("spherical lloyd_fit");
    let lc = lfit.centroids.f32_slice().unwrap();
    assert!(lc.iter().all(|v| v.is_finite()));
    assert!(lc[2 * d..3 * d].iter().all(|&v| v == 0.0));

    // Direct `CpuSession` check for `weight == 1.0` (`alpha == 1.0`) exact override & `alpha == 0.0` with `init_usage == 0.0`
    let mut sess = CpuSession::new(
        2,
        4,
        false,
        0.0,
        1e-4,
        0.0,
        42,
        &[1.25_f32, -2.5, 3.75, 0.5, 10.0, 10.0, 10.0, 10.0],
        &budget,
    )
    .expect("CpuSession::new");
    sess.reset_pass_accumulators(true);
    let batch_rows = [1.0_f32, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0];
    sess.accumulate_f32_chunk(&pool, &batch_rows, 2, CpuMath::Exact, None, &budget)
        .unwrap();
    // alpha = 0.0 with init_usage = 0.0 -> n_new = 0.0, w = 0.0 -> preserves c_old without 0/0 NaN!
    sess.apply_batch_update(0.0, 0, false).unwrap();
    assert!(sess.read_centroids().iter().all(|v| v.is_finite()));
    assert_eq!(&sess.read_centroids()[..4], &[1.25_f32, -2.5, 3.75, 0.5]);

    // alpha = 1.0 -> replace = true, w = 1.0 -> exact batch mean `[2.0, 3.0, 4.0, 5.0]`!
    sess.accumulate_f32_chunk(&pool, &batch_rows, 2, CpuMath::Exact, None, &budget)
        .unwrap();
    sess.apply_batch_update(1.0, 1, false).unwrap();
    assert_eq!(&sess.read_centroids()[..4], &[2.0_f32, 3.0, 4.0, 5.0]);
    assert_eq!(sess.read_usage()[0], 2.0);
    assert_eq!(sess.read_usage()[1], 0.0);
}

#[test]
fn test_cpu_refuses_nonfinite_and_norm_overflow_and_budget_cap() {
    let n = 64_usize;
    let d = 16_usize;
    let k = 4_usize;
    let budget = Budget::new(16 * 1024 * 1024);
    let pool = CpuPool::new(Some(2)).expect("pool");

    let good_rows: Vec<f32> = (0..(n * d))
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();
    let good_cents: Vec<f32> = (0..(k * d)).map(|i| ((i % 7) as f32) * 0.2 - 0.5).collect();
    let good_x = Tensor::from_f32(&good_rows, &[n, d], &budget).unwrap();
    let good_c = Tensor::from_f32(&good_cents, &[k, d], &budget).unwrap();

    // 1. Non-finite and norm-overflow (`2.0e19_f32` -> x^2 = 4.0e38 > f32::MAX) in rows & centroids
    for bad_val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.0e19_f32] {
        let mut bad_rows = good_rows.clone();
        bad_rows[23] = bad_val;
        let bad_x = Tensor::from_f32(&bad_rows, &[n, d], &budget).unwrap();

        let cfg = EmeansConfig::new(k);
        assert!(
            matches!(
                fit_cpu(&pool, &bad_x, Some(&good_c), &cfg, CpuMath::Fast, &budget),
                Err(OjasError::NonFinite { op: "emeans" })
            ),
            "fit_cpu must refuse row bad_val={bad_val:e}"
        );
        // Spherical mode must ALSO refuse 2.0e19 / NaN / Inf in input rows!
        let mut sph_cfg = EmeansConfig::new(k);
        sph_cfg.spherical = true;
        assert!(matches!(
            fit_cpu(
                &pool,
                &bad_x,
                Some(&good_c),
                &sph_cfg,
                CpuMath::Exact,
                &budget
            ),
            Err(OjasError::NonFinite { op: "emeans" })
        ));
        assert!(matches!(
            assign_cpu_f32(
                &pool,
                &bad_rows,
                n,
                d,
                &good_cents,
                k,
                32,
                CpuMath::DeterministicFma,
                false,
                &budget,
            ),
            Err(OjasError::NonFinite { op: "emeans" })
        ));
        assert!(matches!(
            lloyd_fit(&bad_x, Some(&good_c), &LloydConfig::new(k), &budget),
            Err(OjasError::NonFinite { op: "emeans" })
        ));

        // Bad initial centroids (note: 2.0e19 overflows centroid norm2 when spherical = false!)
        let mut bad_cents = good_cents.clone();
        bad_cents[5] = bad_val;
        let bad_c = Tensor::from_f32(&bad_cents, &[k, d], &budget).unwrap();
        assert!(
            matches!(
                fit_cpu(&pool, &good_x, Some(&bad_c), &cfg, CpuMath::Fast, &budget),
                Err(OjasError::NonFinite { op: "emeans" })
            ),
            "fit_cpu must refuse centroid bad_val={bad_val:e}"
        );
        assert!(matches!(
            lloyd_fit(&good_x, Some(&bad_c), &LloydConfig::new(k), &budget),
            Err(OjasError::NonFinite { op: "emeans" })
        ));
    }

    // 2. Non-finite `bf16` (`0x7f80` +Inf, `0xff80` -Inf, `0x7fc0` NaN) & `bf16` norm overflow (`0x5f0f` > 1e19)
    let good_bf16: Vec<u16> = good_rows
        .iter()
        .copied()
        .map(f32_to_bf16_rte_bits)
        .collect();
    for bad_bits in [0x7f80_u16, 0xff80, 0x7fc0, f32_to_bf16_rte_bits(2.0e19_f32)] {
        let mut bad_bf16 = good_bf16.clone();
        bad_bf16[11] = bad_bits;
        let bad_xb = Tensor::from_bf16_bits(&bad_bf16, &[n, d], &budget).unwrap();
        assert!(matches!(
            fit_cpu(
                &pool,
                &bad_xb,
                Some(&good_c),
                &EmeansConfig::new(k),
                CpuMath::Fast,
                &budget
            ),
            Err(OjasError::NonFinite { op: "emeans" })
        ));
        assert!(matches!(
            assign_cpu_bf16(
                &pool,
                &bad_bf16,
                n,
                d,
                &good_cents,
                k,
                32,
                CpuMath::Exact,
                false,
                &budget,
            ),
            Err(OjasError::NonFinite { op: "emeans" })
        ));
    }
    let mut dst_row = [0.0_f32; 4];
    assert!(matches!(
        decode_bf16_row_checked(&[0x3f80, 0x7f80, 0x0000, 0x0000], &mut dst_row),
        Err(OjasError::NonFinite { op: "emeans" })
    ));
    assert!(matches!(
        decode_bf16_to_scratch(&[0x3f80, 0x7fc0], &budget),
        Err(OjasError::NonFinite { op: "emeans" })
    ));

    // 3. `CpuSession` atomic chunk refusal: bad chunk MUST NOT mutate live session accumulators!
    {
        let mut sess =
            CpuSession::new(k, d, false, 0.1, 1e-4, 1.0, 42, &good_cents, &budget).unwrap();
        sess.reset_pass_accumulators(true);
        sess.accumulate_f32_chunk(
            &pool,
            &good_rows[..32 * d],
            32,
            CpuMath::Exact,
            None,
            &budget,
        )
        .unwrap();

        let mut poison_chunk = good_rows[32 * d..].to_vec();
        poison_chunk[d * 10 + 3] = 2.0e19_f32;
        assert!(matches!(
            sess.accumulate_f32_chunk(&pool, &poison_chunk, 32, CpuMath::Exact, None, &budget),
            Err(OjasError::NonFinite { op: "emeans" })
        ));

        // Feed valid second half and verify exact match with uninterrupted 64-row session!
        sess.accumulate_f32_chunk(
            &pool,
            &good_rows[32 * d..],
            32,
            CpuMath::Exact,
            None,
            &budget,
        )
        .unwrap();
        sess.apply_batch_update(0.5, 0, false).unwrap();
        let (obj_recovered, counts_recovered) = sess.finish_epoch();
        let cents_recovered = sess.read_centroids().to_vec();

        let mut clean_sess =
            CpuSession::new(k, d, false, 0.1, 1e-4, 1.0, 42, &good_cents, &budget).unwrap();
        clean_sess.reset_pass_accumulators(true);
        clean_sess
            .accumulate_f32_chunk(&pool, &good_rows, 64, CpuMath::Exact, None, &budget)
            .unwrap();
        clean_sess.apply_batch_update(0.5, 0, false).unwrap();
        let (obj_clean, counts_clean) = clean_sess.finish_epoch();

        assert_eq!(counts_recovered, counts_clean);
        assert_eq!(obj_recovered.to_bits(), obj_clean.to_bits());
        assert_eq!(cents_recovered, clean_sess.read_centroids());
    }

    // 4. Unsupported DType (`U32`) -> `OjasError::Dtype`, pure `Device` -> `OjasError::Placement`,
    //    `Mapped` (`SharedBuf`) `F32`/`Bf16` -> zero-copy `host_view_window()`, and bad rank -> `OjasError::Shape`
    let u32_x = Tensor::from_u32(&vec![1_u32; n * d], &[n, d], &budget).unwrap();
    assert!(matches!(
        fit_cpu(
            &pool,
            &u32_x,
            None,
            &EmeansConfig::new(k),
            CpuMath::Fast,
            &budget
        ),
        Err(OjasError::Dtype {
            op: "emeans",
            expected: DType::F32,
            got: DType::U32
        })
    ));
    let rank1_x = Tensor::from_f32(&good_rows, &[n * d], &budget).unwrap();
    assert!(matches!(
        fit_cpu(
            &pool,
            &rank1_x,
            None,
            &EmeansConfig::new(k),
            CpuMath::Fast,
            &budget
        ),
        Err(OjasError::Shape { op: "emeans", .. })
    ));

    let dev_x = Tensor::from_device(
        Arc::new(MockDiscreteDevice {
            byte_len: n * d * 4,
        }),
        &[n, d],
        DType::F32,
        &budget,
    )
    .unwrap();
    assert!(matches!(
        fit_cpu(
            &pool,
            &dev_x,
            None,
            &EmeansConfig::new(k),
            CpuMath::Fast,
            &budget
        ),
        Err(OjasError::Placement {
            op: "emeans",
            expected: None,
            found: Some(BackendId::Cuda)
        })
    ));
    assert!(matches!(
        assign_cpu_tensor(&pool, &dev_x, &good_c, 32, CpuMath::Exact, false, &budget),
        Err(OjasError::Placement { op: "emeans", .. })
    ));
    assert!(matches!(
        lloyd_fit(&dev_x, None, &LloydConfig::new(k), &budget),
        Err(OjasError::Placement { op: "emeans", .. })
    ));

    let mapped_f32_bytes: Vec<u8> = good_rows.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let mapped_f32 = Tensor::from_mapped(
        Arc::new(MockMappedBuffer {
            bytes: mapped_f32_bytes,
            timeline: MockTimeline,
        }),
        &[n, d],
        DType::F32,
        &budget,
    )
    .unwrap();
    let mapped_bf16_bytes: Vec<u8> = good_bf16.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let mapped_bf16 = Tensor::from_mapped(
        Arc::new(MockMappedBuffer {
            bytes: mapped_bf16_bytes,
            timeline: MockTimeline,
        }),
        &[n, d],
        DType::Bf16,
        &budget,
    )
    .unwrap();
    let mv_bf16 = MatrixView::open(&mapped_bf16, "emeans").expect("mapped bf16 view");
    assert!(matches!(mv_bf16.data(), MatrixData::Bf16Bytes(s) if s.len() == n * d));
    if let MatrixData::Bf16Bytes(pairs) = mv_bf16.data() {
        let dec = decode_bf16_bytes_to_scratch(&pairs[..d], &budget).unwrap();
        assert_eq!(dec.len(), d);
        let mut msess =
            CpuSession::new(k, d, false, 0.1, 1e-4, 1.0, 42, &good_cents, &budget).unwrap();
        msess.reset_pass_accumulators(true);
        msess
            .accumulate_bf16_bytes_chunk(&pool, pairs, n, CpuMath::Exact, None, &budget)
            .unwrap();
    }
    drop(mv_bf16);

    let fit_host = fit_cpu(
        &pool,
        &good_x,
        Some(&good_c),
        &EmeansConfig::new(k),
        CpuMath::DeterministicFma,
        &budget,
    )
    .unwrap();
    let fit_mapped = fit_cpu(
        &pool,
        &mapped_f32,
        Some(&good_c),
        &EmeansConfig::new(k),
        CpuMath::DeterministicFma,
        &budget,
    )
    .unwrap();
    assert_eq!(
        fit_mapped.centroids.f32_slice().unwrap(),
        fit_host.centroids.f32_slice().unwrap()
    );
    let (lbl_mt, obj_mt) = assign_cpu_tensor(
        &pool,
        &mapped_bf16,
        &good_c,
        32,
        CpuMath::DeterministicFma,
        false,
        &budget,
    )
    .unwrap();
    let (lbl_hb, obj_hb) = assign_cpu_bf16(
        &pool,
        &good_bf16,
        n,
        d,
        &good_cents,
        k,
        32,
        CpuMath::DeterministicFma,
        false,
        &budget,
    )
    .unwrap();
    assert_eq!(lbl_mt, lbl_hb);
    assert_eq!(obj_mt.to_bits(), obj_hb.to_bits());

    // 5. Tight `Budget` refuses cleanly with `OjasError::CapacityExceeded` AND leaves `live_bytes() == 0`!
    let exec_budget = Budget::new(512); // 512 bytes: too small for scratch/centroids!
    assert!(matches!(
        fit_cpu(
            &pool,
            &good_x,
            Some(&good_c),
            &EmeansConfig::new(k),
            CpuMath::Fast,
            &exec_budget
        ),
        Err(OjasError::CapacityExceeded { .. })
    ));
    assert_eq!(exec_budget.live_bytes().unwrap(), 0);

    assert!(matches!(
        assign_cpu_f32(
            &pool,
            &good_rows,
            n,
            d,
            &good_cents,
            k,
            32,
            CpuMath::DeterministicFma,
            false,
            &exec_budget,
        ),
        Err(OjasError::CapacityExceeded { .. })
    ));
    assert_eq!(exec_budget.live_bytes().unwrap(), 0);

    assert!(matches!(
        lloyd_fit(&good_x, Some(&good_c), &LloydConfig::new(k), &exec_budget),
        Err(OjasError::CapacityExceeded { .. })
    ));
    assert_eq!(exec_budget.live_bytes().unwrap(), 0);

    // Healthy budget returns to 0 live bytes once `EmeansFit` and `LloydFit` are dropped
    let run_budget = Budget::new(4 * 1024 * 1024);
    {
        let fit_ok = fit_cpu(
            &pool,
            &good_x,
            Some(&good_c),
            &EmeansConfig::new(k),
            CpuMath::DeterministicFma,
            &run_budget,
        )
        .expect("fit_ok");
        assert!(run_budget.live_bytes().unwrap() > 0);
        drop(fit_ok);
    }
    assert_eq!(run_budget.live_bytes().unwrap(), 0);

    {
        let lfit_ok =
            lloyd_fit(&good_x, Some(&good_c), &LloydConfig::new(k), &run_budget).expect("lfit_ok");
        assert!(run_budget.live_bytes().unwrap() > 0);
        drop(lfit_ok);
    }
    assert_eq!(run_budget.live_bytes().unwrap(), 0);
}

#[test]
fn test_emeans_2_6_passes_matches_25_iter_lloyd_mse_on_gaussian_mixture() {
    let n = 8_192_usize;
    let d = 32_usize;
    let k = 64_usize;
    let budget = Budget::new(128 * 1024 * 1024);

    // Multi-cluster Gaussian mixture (`k = 64` components in `d = 32`, `128` points per component).
    // Uniform random seeding (`seed = 42`) leaves colliding/empty components that 2-iter Lloyd
    // cannot untangle, while 2.6-pass `e-means` revives unused clusters and refines over ~84 EMA steps.
    let mut rng = SplitMix64::new(0x2611_0950_1e00_0042);
    let mut true_centers = vec![0.0_f32; k * d];
    for val in &mut true_centers {
        let (z0, _) = standard_normal_pair(&mut rng);
        *val = (z0 * 2.5) as f32;
    }

    let mut rows = vec![0.0_f32; n * d];
    for r in 0..n {
        let comp = (rng.next_bounded_u64(k as u64)) as usize;
        let center = &true_centers[comp * d..(comp + 1) * d];
        let mut col = 0;
        while col < d {
            let (z0, z1) = standard_normal_pair(&mut rng);
            rows[r * d + col] = center[col] + (z0 * 0.35) as f32;
            if col + 1 < d {
                rows[r * d + col + 1] = center[col + 1] + (z1 * 0.35) as f32;
            }
            col += 2;
        }
    }

    let x = Tensor::from_f32(&rows, &[n, d], &budget).expect("Gaussian mixture Tensor");
    let pool = CpuPool::new(Some(4)).expect("pool");

    // 1. 2.6-pass `e-means` (`canonical_v1` defaults, `seed = 42`)
    let mut em_cfg = EmeansConfig::new(k);
    em_cfg.nepochs = 2.6;
    em_cfg.seed = 42;
    let em_fit = fit_cpu(&pool, &x, None, &em_cfg, CpuMath::DeterministicFma, &budget)
        .expect("2.6-pass fit_cpu");

    // 2. 25-iteration full-pass Lloyd's `k`-means (`seed = 42`, `tol = 0.0` so all 25 passes run)
    let mut lloyd25_cfg = LloydConfig::new(k);
    lloyd25_cfg.max_iters = 25;
    lloyd25_cfg.tol = 0.0;
    lloyd25_cfg.seed = 42;
    lloyd25_cfg.threads = Some(4);
    lloyd25_cfg.math = CpuMath::DeterministicFma;
    let lloyd25_fit = lloyd_fit(&x, None, &lloyd25_cfg, &budget).expect("25-iter lloyd_fit");
    assert_eq!(lloyd25_fit.iter_objectives.len(), 25);
    for w in lloyd25_fit.iter_objectives.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-5,
            "Lloyd objective must decrease monotonically: {} -> {}",
            w[0],
            w[1]
        );
    }

    // 3. 2-iteration full-pass Lloyd's `k`-means (`seed = 42`, matching pass budget ~2.0)
    let mut lloyd2_cfg = lloyd25_cfg.clone();
    lloyd2_cfg.max_iters = 2;
    let lloyd2_fit = lloyd_fit(&x, None, &lloyd2_cfg, &budget).expect("2-iter lloyd_fit");
    assert_eq!(lloyd2_fit.iter_objectives.len(), 2);

    // Evaluate exact full-dataset MSE (`1/n * sum_i ||x_i - c_{a_i}||_2^2`) for all three codebooks
    let eval_mse = |centroids_tensor: &Tensor| -> f64 {
        let c = centroids_tensor.f32_slice().unwrap();
        let (_, total_sq) = assign_cpu_f32(
            &pool,
            &rows,
            n,
            d,
            c,
            k,
            4096,
            CpuMath::Exact,
            false,
            &budget,
        )
        .expect("eval_mse assign");
        total_sq / (n as f64)
    };

    let emeans_mse = eval_mse(&em_fit.centroids);
    let lloyd25_mse = eval_mse(&lloyd25_fit.centroids);
    let lloyd2_mse = eval_mse(&lloyd2_fit.centroids);
    eprintln!(
        "Gaussian mixture (n={n}, d={d}, k={k}): emeans_2.6_mse={emeans_mse:.6}, lloyd_25_mse={lloyd25_mse:.6}, lloyd_2_mse={lloyd2_mse:.6}, ratio_2_over_em={:.4}x",
        lloyd2_mse / emeans_mse
    );

    assert!(
        emeans_mse <= lloyd25_mse * 1.02,
        "expected 2.6-pass e-means MSE ({emeans_mse:.6}) <= 1.02 * 25-iter Lloyd MSE ({lloyd25_mse:.6})"
    );
    assert!(
        lloyd2_mse / emeans_mse > 1.2,
        "expected 2.6-pass e-means MSE ({emeans_mse:.6}) to beat 2-iter Lloyd MSE ({lloyd2_mse:.6}) by > 1.2x (ratio = {:.3}x)",
        lloyd2_mse / emeans_mse
    );

    // Also verify `tol > 0.0` early stopping on already-converged centroids
    let mut lloyd_early = LloydConfig::new(k);
    lloyd_early.max_iters = 25;
    lloyd_early.tol = 1.0e-3;
    lloyd_early.threads = Some(4);
    let early_fit =
        lloyd_fit(&x, Some(&em_fit.centroids), &lloyd_early, &budget).expect("early stop lloyd");
    assert!(
        early_fit.iter_objectives.len() < 25,
        "converged initial centroids with tol=1e-3 should stop before 25 iters, ran {}",
        early_fit.iter_objectives.len()
    );
}
