use std::any::Any;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ojas_cluster::{
    emeans_assign, emeans_fit, epoch_permutation, sample_subset_sorted, BatchLaw, ClusterDevice,
    CpuMath, EmeansConfig, EmeansEngine, EmeansFit, EmeansSession, MetalPrecision, SplitMix64,
    UNASSIGNED_LABEL,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use ojas_core::BudgetTrimHandler;
use ojas_core::{
    BackendId, Budget, DType, DeviceBuffer, FenceError, HostLease, HostLeaseMut, MappedBuffer,
    OjasError, Scratch, Tensor, Timeline,
};

#[cfg(all(target_os = "macos", feature = "metal"))]
const ALLOW_NO_GPU: &str = "OJAS_ALLOW_NO_GPU";

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|p| p.into_inner())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn allow_no_gpu() -> bool {
    std::env::var(ALLOW_NO_GPU).ok().as_deref() == Some("1")
}

/// Open a `ClusterDevice::Metal` engine, asserting `OjasError::Unsupported` when compiled
/// without Metal support, skipping cleanly only when `OJAS_ALLOW_NO_GPU=1`, and failing otherwise.
fn open_metal_engine(precision: MetalPrecision, budget: &Budget) -> Option<EmeansEngine> {
    let res = EmeansEngine::new(ClusterDevice::Metal { precision }, budget);
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        match res {
            Err(OjasError::Unsupported { op: "emeans", .. }) => None,
            other => panic!("expected Unsupported without macOS+metal, got {other:?}"),
        }
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        match res {
            Ok(engine) => Some(engine),
            Err(err) if allow_no_gpu() => {
                eprintln!("SKIP ({ALLOW_NO_GPU}=1): Metal engine unavailable: {err}");
                None
            }
            Err(err) => {
                panic!("Metal engine failed to open (set {ALLOW_NO_GPU}=1 to skip): {err:?}")
            }
        }
    }
}

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

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct CountingTrimHandler {
    calls: AtomicUsize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl BudgetTrimHandler for CountingTrimHandler {
    fn trim_memory(&self, _needed_bytes: u64) -> u64 {
        self.calls.fetch_add(1, Ordering::Relaxed);
        0
    }

    fn pool_key(&self) -> Option<usize> {
        Some(0xC0DE_CAFE)
    }
}

#[inline]
fn next_signed_f32(rng: &mut SplitMix64) -> f32 {
    let u = ((rng.next_u64() >> 40) as f64) * (1.0 / ((1_u64 << 24) as f64));
    (2.0 * u - 1.0) as f32
}

/// Deterministic mixture matching the offline `faiss/contrib/e_means` golden generator bit-for-bit.
fn make_golden_mixture(
    n: usize,
    d: usize,
    k: usize,
    seed: u64,
    center_scale: f32,
    noise_scale: f32,
    spherical: bool,
) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut centers = vec![0.0_f32; k * d];
    for val in &mut centers {
        *val = center_scale * next_signed_f32(&mut rng);
    }
    let mut x = vec![0.0_f32; n * d];
    for i in 0..n {
        let cid = i % k;
        let row = &mut x[i * d..(i + 1) * d];
        for col in 0..d {
            let n1 = next_signed_f32(&mut rng);
            let n2 = next_signed_f32(&mut rng);
            row[col] = centers[cid * d + col] + noise_scale * (n1 + n2);
        }
        if spherical {
            let mut sq = 0.0_f64;
            for &v in row.iter() {
                sq += f64::from(v) * f64::from(v);
            }
            let norm = sq.sqrt();
            for v in row.iter_mut() {
                *v = (f64::from(*v) / norm) as f32;
            }
        }
    }
    x
}

fn exact_f64_eval_objective(
    x: &[f32],
    n: usize,
    d: usize,
    c: &[f32],
    k: usize,
    spherical: bool,
) -> f64 {
    let mut total = 0.0_f64;
    for i in 0..n {
        let xr = &x[i * d..(i + 1) * d];
        if spherical {
            let mut best_sim = f64::NEG_INFINITY;
            for j in 0..k {
                let cr = &c[j * d..(j + 1) * d];
                let mut dot = 0.0_f64;
                for col in 0..d {
                    dot += f64::from(xr[col]) * f64::from(cr[col]);
                }
                if dot > best_sim {
                    best_sim = dot;
                }
            }
            total += 2.0 - 2.0 * best_sim;
        } else {
            let mut best_d2 = f64::INFINITY;
            for j in 0..k {
                let cr = &c[j * d..(j + 1) * d];
                let mut d2 = 0.0_f64;
                for col in 0..d {
                    let diff = f64::from(xr[col]) - f64::from(cr[col]);
                    d2 += diff * diff;
                }
                if d2 < best_d2 {
                    best_d2 = d2;
                }
            }
            total += best_d2;
        }
    }
    total / (n as f64)
}

fn f32_to_bf16_rte_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let mag = bits & 0x7fff_ffff;
    if mag > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    if mag < 0x0080_0000 {
        return ((bits >> 16) & 0x8000) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn assert_fits_bit_identical(fit_a: &EmeansFit, fit_b: &EmeansFit, context: &str) {
    assert_eq!(fit_a.recipe, fit_b.recipe, "{context}: recipe mismatch");
    assert_eq!(
        fit_a.cluster_counts, fit_b.cluster_counts,
        "{context}: cluster_counts mismatch"
    );
    assert_eq!(
        fit_a.usage.len(),
        fit_b.usage.len(),
        "{context}: usage len mismatch"
    );
    for (idx, (&ua, &ub)) in fit_a.usage.iter().zip(&fit_b.usage).enumerate() {
        assert_eq!(
            ua.to_bits(),
            ub.to_bits(),
            "{context}: usage[{idx}] bit mismatch ({ua} vs {ub})"
        );
    }

    let ca = fit_a.centroids.f32_slice().expect("centroids a f32");
    let cb = fit_b.centroids.f32_slice().expect("centroids b f32");
    assert_eq!(ca.len(), cb.len(), "{context}: centroids len mismatch");
    for (idx, (&va, &vb)) in ca.iter().zip(cb).enumerate() {
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "{context}: centroid[{idx}] bit mismatch ({va} vs {vb})"
        );
    }

    assert_eq!(
        fit_a.epoch_stats.len(),
        fit_b.epoch_stats.len(),
        "{context}: epoch_stats len mismatch"
    );
    for (ep, (sa, sb)) in fit_a.epoch_stats.iter().zip(&fit_b.epoch_stats).enumerate() {
        assert_eq!(sa.epoch, sb.epoch, "{context}: epoch[{ep}] index");
        assert_eq!(
            sa.rows_seen, sb.rows_seen,
            "{context}: epoch[{ep}] rows_seen"
        );
        assert_eq!(
            sa.batch_updates, sb.batch_updates,
            "{context}: epoch[{ep}] batch_updates"
        );
        assert_eq!(
            sa.revived_clusters, sb.revived_clusters,
            "{context}: epoch[{ep}] revived_clusters"
        );
        assert_eq!(
            sa.mean_objective.to_bits(),
            sb.mean_objective.to_bits(),
            "{context}: epoch[{ep}] mean_objective bit mismatch ({} vs {})",
            sa.mean_objective,
            sb.mean_objective
        );
    }
}

fn verify_streaming_parity_for_engine(
    engine: &EmeansEngine,
    x_data: &[f32],
    n: usize,
    d: usize,
    cfg: &EmeansConfig,
    label: &str,
) {
    let io_budget = Budget::new(64 << 20);
    let x_tensor = Tensor::from_f32(x_data, &[n, d], &io_budget).expect("x_tensor");
    let init_tensor =
        Tensor::from_f32(&x_data[..cfg.k * d], &[cfg.k, d], &io_budget).expect("init_tensor");

    let fit_resident = engine
        .fit_with_init(&x_tensor, &init_tensor, cfg)
        .expect("fit_with_init");

    let mut session = EmeansSession::begin(engine, n, &init_tensor, cfg).expect("session begin");
    assert_eq!(session.epoch(), 0);
    assert_eq!(session.step(), 0);
    assert_eq!(session.recipe(), &fit_resident.recipe);

    let recipe = session.recipe().clone();
    let num_epochs = recipe.epoch_rows.len();
    let mut unpermuted_labels = vec![UNASSIGNED_LABEL; n];
    let mut streamed_closing_labels = Vec::new();
    let mut expected_global_step = 0_usize;

    for (epoch, &rows_this_epoch) in recipe.epoch_rows.iter().enumerate() {
        assert_eq!(session.epoch(), epoch);
        assert_eq!(session.remaining_in_epoch(), rows_this_epoch);
        let final_epoch = epoch + 1 == num_epochs;

        let order = epoch_permutation(n, recipe.seed, epoch, &io_budget).expect("epoch perm");
        let order_s = order.as_slice();
        let mut pos = 0_usize;

        while pos < rows_this_epoch {
            let b_len = (rows_this_epoch - pos).min(recipe.batch_size);
            assert_eq!(session.remaining_in_batch(), b_len);
            let mut sub_off = 0_usize;

            while sub_off < b_len {
                let c_len = (b_len - sub_off).min(recipe.assignment_chunk_size);
                let chunk_indices = &order_s[pos + sub_off..pos + sub_off + c_len];
                let mut chunk_f32 = Vec::with_capacity(c_len * d);
                for &row_idx in chunk_indices {
                    let r = row_idx as usize;
                    chunk_f32.extend_from_slice(&x_data[r * d..(r + 1) * d]);
                }
                let chunk_t =
                    Tensor::from_f32(&chunk_f32, &[c_len, d], &io_budget).expect("chunk tensor");

                let (acc, maybe_labels) = session.accumulate(&chunk_t).expect("accumulate");
                assert_eq!(acc.rows_accepted, c_len);
                assert_eq!(acc.remaining_in_batch, b_len - (sub_off + c_len));
                assert_eq!(
                    acc.remaining_in_epoch,
                    rows_this_epoch - (pos + sub_off + c_len)
                );
                assert_eq!(acc.batch_ready, sub_off + c_len == b_len);

                if final_epoch && cfg.capture_assignments {
                    let chunk_lbls = maybe_labels.expect("closing epoch must return labels");
                    assert_eq!(chunk_lbls.len(), c_len);
                    streamed_closing_labels.extend_from_slice(&chunk_lbls);
                    for (&row_idx, &lbl) in chunk_indices.iter().zip(&chunk_lbls) {
                        unpermuted_labels[row_idx as usize] = lbl;
                    }
                } else {
                    assert!(
                        maybe_labels.is_none(),
                        "{label}: non-closing epoch must return None labels"
                    );
                }

                sub_off += c_len;
            }

            assert_eq!(session.remaining_in_batch(), 0);
            let _revived = session.update().expect("session update");
            expected_global_step += 1;
            assert_eq!(session.step(), expected_global_step);
            pos += b_len;
        }

        assert_eq!(session.remaining_in_epoch(), 0);
        assert_eq!(session.remaining_in_batch(), 0);

        // Snapshot current_centroids mid-session and verify shape / finiteness / spherical unit norm.
        let snap = session.current_centroids().expect("current_centroids");
        assert_eq!(snap.shape(), &[cfg.k, d]);
        let snap_s = snap.f32_slice().expect("snap f32");
        assert!(snap_s.iter().all(|v| v.is_finite()));
        if cfg.spherical {
            for j in 0..cfg.k {
                let row = &snap_s[j * d..(j + 1) * d];
                let norm2: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
                if norm2 > 0.0 {
                    assert!(
                        (norm2.sqrt() - 1.0).abs() < 1.0e-5,
                        "{label}: spherical current_centroids row {j} norm2={norm2}"
                    );
                }
            }
        }

        let ep_stat = session.end_epoch().expect("end_epoch");
        assert_eq!(ep_stat.epoch, epoch);
        assert_eq!(ep_stat.rows_seen, rows_this_epoch);
    }

    assert_eq!(session.epoch(), num_epochs);
    let fit_stream = session.finalize().expect("session finalize");

    assert_fits_bit_identical(&fit_resident, &fit_stream, label);
    assert_eq!(
        fit_stream.labels, streamed_closing_labels,
        "{label}: finalize labels must equal closing-epoch streamed labels"
    );
    if cfg.capture_assignments {
        assert_eq!(
            unpermuted_labels, fit_resident.labels,
            "{label}: unpermuted streaming labels must match fit_with_init bit-for-bit"
        );
    }
}

#[test]
fn test_streaming_session_is_bit_identical_to_emeans_fit_on_cpu_and_metal() {
    let _guard = serial();

    let n = 320;
    let d = 24;
    let k = 12;
    let x_euc = make_golden_mixture(n, d, k, 0x1020_3040_5060_7080, 2.5, 0.35, false);
    let x_sph = make_golden_mixture(n, d, k, 0x8070_6050_4030_2010, 2.5, 0.30, true);

    let mut cfg_euc = EmeansConfig::new(k);
    cfg_euc.nepochs = 2.4;
    cfg_euc.batch_size = Some(80);
    cfg_euc.assignment_chunk_size = 33; // multi-chunk irregular sub-batches inside 80-row batches
    cfg_euc.seed = 99;

    let mut cfg_sph = EmeansConfig::new(k);
    cfg_sph.nepochs = 1.6;
    cfg_sph.spherical = true;
    cfg_sph.batch_size = Some(72);
    cfg_sph.assignment_chunk_size = 29;
    cfg_sph.seed = 123;

    // Single-epoch fractional schedule (`nepochs < 1.0`) verifying epoch-0 `final_epoch = true`.
    let mut cfg_frac = EmeansConfig::new(k);
    cfg_frac.nepochs = 0.75;
    cfg_frac.batch_size = Some(64);
    cfg_frac.assignment_chunk_size = 25;
    cfg_frac.seed = 55;

    let cpu_budget = Budget::new(64 << 20);
    let cpu_engine = EmeansEngine::new(
        ClusterDevice::Cpu {
            threads: Some(4),
            math: CpuMath::DeterministicFma,
        },
        &cpu_budget,
    )
    .expect("CPU engine");
    assert_eq!(
        cpu_engine.device(),
        ClusterDevice::Cpu {
            threads: Some(4),
            math: CpuMath::DeterministicFma,
        }
    );
    assert_eq!(cpu_engine.trim_pool(), 0);

    verify_streaming_parity_for_engine(&cpu_engine, &x_euc, n, d, &cfg_euc, "cpu-euc");
    verify_streaming_parity_for_engine(&cpu_engine, &x_sph, n, d, &cfg_sph, "cpu-sph");
    verify_streaming_parity_for_engine(&cpu_engine, &x_euc, n, d, &cfg_frac, "cpu-frac");
    drop(cpu_engine);
    assert_eq!(cpu_budget.live_bytes().unwrap(), 0);

    for precision in [
        MetalPrecision::Bf16,
        MetalPrecision::Bf16AssignF32Stats,
        MetalPrecision::F32,
    ] {
        let metal_budget = Budget::new(128 << 20);
        let Some(metal_engine) = open_metal_engine(precision, &metal_budget) else {
            break;
        };
        assert_eq!(metal_engine.device(), ClusterDevice::Metal { precision });
        let tag = format!("metal-{precision:?}");
        verify_streaming_parity_for_engine(
            &metal_engine,
            &x_euc,
            n,
            d,
            &cfg_euc,
            &format!("{tag}-euc"),
        );
        verify_streaming_parity_for_engine(
            &metal_engine,
            &x_sph,
            n,
            d,
            &cfg_sph,
            &format!("{tag}-sph"),
        );
        verify_streaming_parity_for_engine(
            &metal_engine,
            &x_euc,
            n,
            d,
            &cfg_frac,
            &format!("{tag}-frac"),
        );
        drop(metal_engine);
        assert_eq!(metal_budget.live_bytes().unwrap(), 0);
    }
}

#[test]
fn test_metal_k1024_multi_epoch_revival_repeat_and_streaming_bit_identity() {
    let _guard = serial();

    // `k = 1024` (> 256 multi-tile donor radix sort + 1024-threadgroup `emeans_apply_ema`),
    // `d = 320` (> 256 multi-simdgroup + `col += 256` loop + 5 x 64 cooperative TensorOps K-tiles),
    // `n = 4096` drawn from 192 tight modes so Forgy `init_centroids` (`x[..k*d]`) starves hundreds
    // of clusters per 1024-row mini-batch and drives heavy Gumbel dead-cluster split revival across
    // a 3-pass fractional schedule (`nepochs = 2.5` -> `[4096, 2048, 4096]`).
    let n = 4_096;
    let d = 320;
    let k = 1_024;
    let active_modes = 192;
    let x_euc = make_golden_mixture(n, d, active_modes, 0x2026_1111_1024_0320, 2.5, 0.25, false);
    let x_sph = make_golden_mixture(n, d, active_modes, 0x2026_1111_1024_5150, 2.5, 0.25, true);

    let mut cfg_euc = EmeansConfig::new(k);
    cfg_euc.nepochs = 2.5;
    cfg_euc.batch_size = Some(1_024);
    cfg_euc.assignment_chunk_size = 384; // 384 + 384 + 256 = 1024 irregular sub-batches
    cfg_euc.seed = 0x5150_1024;

    let mut cfg_sph = EmeansConfig::new(k);
    cfg_sph.nepochs = 2.5;
    cfg_sph.spherical = true;
    cfg_sph.batch_size = Some(1_024);
    cfg_sph.assignment_chunk_size = 384;
    cfg_sph.seed = 0x5150_2048;

    let io_budget = Budget::new(64 << 20);
    let x_euc_t = Tensor::from_f32(&x_euc, &[n, d], &io_budget).expect("x_euc tensor");
    let init_euc_t =
        Tensor::from_f32(&x_euc[..k * d], &[k, d], &io_budget).expect("init_euc tensor");
    let x_sph_t = Tensor::from_f32(&x_sph, &[n, d], &io_budget).expect("x_sph tensor");
    let init_sph_t =
        Tensor::from_f32(&x_sph[..k * d], &[k, d], &io_budget).expect("init_sph tensor");

    for precision in [
        MetalPrecision::Bf16,
        MetalPrecision::Bf16AssignF32Stats,
        MetalPrecision::F32,
    ] {
        let metal_budget = Budget::new(256 << 20);
        let Some(metal_engine) = open_metal_engine(precision, &metal_budget) else {
            break;
        };
        assert_eq!(metal_engine.device(), ClusterDevice::Metal { precision });

        for (mode_tag, x_slice, x_t, init_t, cfg) in [
            ("euc", x_euc.as_slice(), &x_euc_t, &init_euc_t, &cfg_euc),
            ("sph", x_sph.as_slice(), &x_sph_t, &init_sph_t, &cfg_sph),
        ] {
            let ctx = format!("metal-k1024-{precision:?}-{mode_tag}");
            let fit_1 = metal_engine
                .fit_with_init(x_t, init_t, cfg)
                .expect("fit_with_init run 1");
            assert_eq!(
                fit_1.epoch_stats.len(),
                3,
                "{ctx}: expected 3 passes for nepochs=2.5"
            );
            let total_revived: usize = fit_1.epoch_stats.iter().map(|s| s.revived_clusters).sum();
            assert!(
                total_revived >= 128,
                "{ctx}: expected heavy Gumbel dead-cluster revival at k=1024, got {total_revived}"
            );

            let fit_2 = metal_engine
                .fit_with_init(x_t, init_t, cfg)
                .expect("fit_with_init run 2");
            let fit_3 = metal_engine
                .fit_with_init(x_t, init_t, cfg)
                .expect("fit_with_init run 3");

            assert_fits_bit_identical(&fit_1, &fit_2, &format!("{ctx}-repeat-1v2"));
            assert_eq!(
                fit_1.labels, fit_2.labels,
                "{ctx}: repeat 1v2 labels mismatch"
            );
            assert_fits_bit_identical(&fit_1, &fit_3, &format!("{ctx}-repeat-1v3"));
            assert_eq!(
                fit_1.labels, fit_3.labels,
                "{ctx}: repeat 1v3 labels mismatch"
            );
            drop(fit_1);
            drop(fit_2);
            drop(fit_3);

            verify_streaming_parity_for_engine(
                &metal_engine,
                x_slice,
                n,
                d,
                cfg,
                &format!("{ctx}-stream"),
            );
        }

        drop(metal_engine);
        assert_eq!(metal_budget.live_bytes().unwrap(), 0);
    }
}

fn run_early_drop_and_200_reuse_check(device: ClusterDevice) -> bool {
    let io_budget = Budget::new(64 << 20);
    let n = 192;
    let d = 16;
    let k = 8;
    let x_vec = make_golden_mixture(n, d, k, 0xABCD_EF01_2345_6789, 2.0, 0.35, false);
    let x = Tensor::from_f32(&x_vec, &[n, d], &io_budget).expect("x");
    let init_c = Tensor::from_f32(&x_vec[..k * d], &[k, d], &io_budget).expect("init_c");
    let chunk_24 = Tensor::from_f32(&x_vec[..24 * d], &[24, d], &io_budget).expect("chunk_24");
    let chunk_48 = Tensor::from_f32(&x_vec[..48 * d], &[48, d], &io_budget).expect("chunk_48");

    let mut cfg = EmeansConfig::new(k);
    cfg.nepochs = 2.0;
    cfg.batch_size = Some(48);
    cfg.assignment_chunk_size = 24;
    cfg.seed = 7;

    let engine_budget = Budget::new(128 << 20);
    let engine = match device {
        ClusterDevice::Cpu { .. } => EmeansEngine::new(device, &engine_budget).expect("cpu engine"),
        ClusterDevice::Metal { precision } => match open_metal_engine(precision, &engine_budget) {
            Some(eng) => eng,
            None => return false,
        },
    };
    assert_eq!(engine_budget.live_bytes().unwrap(), 0);

    // Drop 5 unfinished sessions at 5 distinct stages while `engine` stays live,
    // asserting `engine_budget.live_bytes()? == 0` immediately after each `drop(session)`.
    for stage in 0..5 {
        let mut session = EmeansSession::begin(&engine, n, &init_c, &cfg).expect("begin");
        assert!(
            engine_budget.live_bytes().unwrap() > 0,
            "stage {stage}: live session must hold budget bytes"
        );
        match stage {
            0 => {
                // Drop right after `begin`.
            }
            1 => {
                // Drop mid-batch after partial `accumulate`.
                session.accumulate(&chunk_24).expect("partial accum");
            }
            2 => {
                // Drop with a completed batch before `update`.
                session.accumulate(&chunk_48).expect("full batch accum");
            }
            3 => {
                // Drop mid-epoch after one `update`.
                session.accumulate(&chunk_48).expect("batch 0 accum");
                session.update().expect("batch 0 update");
                session.accumulate(&chunk_24).expect("batch 1 partial");
            }
            4 => {
                // Drop after completing epoch 0 of a 2-epoch session.
                for _ in 0..4 {
                    session.accumulate(&chunk_48).expect("ep0 batch");
                    session.update().expect("ep0 update");
                }
                session.end_epoch().expect("ep0 end");
                session.accumulate(&chunk_24).expect("ep1 partial");
            }
            _ => unreachable!(),
        }
        drop(session);
        assert_eq!(
            engine_budget.live_bytes().unwrap(),
            0,
            "stage {stage} ({device:?}): budget.live_bytes() must be 0 immediately after drop(session)"
        );
    }

    // Run 200 sequential `engine.fit(...)` calls on the live engine.
    let mut fit_cfg = EmeansConfig::new(k);
    fit_cfg.nepochs = 1.5;
    fit_cfg.seed = 21;

    for iter in 0..200 {
        let fit = engine.fit(&x, &fit_cfg).expect("sequential fit");
        assert_eq!(fit.centroids.shape(), &[k, d]);
        drop(fit);

        if iter == 99 && matches!(device, ClusterDevice::Metal { .. }) {
            let freed = engine.trim_pool();
            assert!(
                freed > 0,
                "Metal engine pool should have cached bytes to trim"
            );
            assert_eq!(
                engine_budget.live_bytes().unwrap(),
                0,
                "after trim_pool(), budget.live_bytes() must be 0"
            );
        }
    }

    drop(engine);
    assert_eq!(
        engine_budget.live_bytes().unwrap(),
        0,
        "after 200 fits and drop(engine) ({device:?}), budget.live_bytes() must be 0"
    );
    true
}

#[test]
fn test_session_early_drop_and_engine_200_reuse_leak_zero_bytes() {
    let _guard = serial();

    assert!(run_early_drop_and_200_reuse_check(ClusterDevice::Cpu {
        threads: Some(2),
        math: CpuMath::Fast,
    }));
    let _ = run_early_drop_and_200_reuse_check(ClusterDevice::Metal {
        precision: MetalPrecision::Bf16,
    });
}

#[test]
fn test_mapped_shared_buf_cross_runtime_safety_and_budget_trim_handler() {
    let _guard = serial();

    let n = 256;
    let d = 16;
    let k = 8;
    let x_vec = make_golden_mixture(n, d, k, 0x55AA_55AA_1234_5678, 2.5, 0.35, false);

    // 1. Cross-platform MockMappedBuffer (`F32` and `Bf16`) on `ClusterDevice::Cpu`
    //    verifying `host_view_window()` is taken instead of `DeviceBuffer::read_bytes`.
    {
        let budget = Budget::new(32 << 20);
        let mut bytes_f32 = Vec::with_capacity(x_vec.len() * 4);
        for &v in &x_vec {
            bytes_f32.extend_from_slice(&v.to_ne_bytes());
        }
        let mapped_f32 = Tensor::from_mapped(
            Arc::new(MockMappedBuffer {
                bytes: bytes_f32,
                timeline: MockTimeline,
            }),
            &[n, d],
            DType::F32,
            &budget,
        )
        .expect("mapped f32");

        let mut bytes_bf16 = Vec::with_capacity(x_vec.len() * 2);
        for &v in &x_vec {
            bytes_bf16.extend_from_slice(&f32_to_bf16_rte_bits(v).to_ne_bytes());
        }
        let mapped_bf16 = Tensor::from_mapped(
            Arc::new(MockMappedBuffer {
                bytes: bytes_bf16,
                timeline: MockTimeline,
            }),
            &[n, d],
            DType::Bf16,
            &budget,
        )
        .expect("mapped bf16");

        let cpu_eng = EmeansEngine::new(
            ClusterDevice::Cpu {
                threads: Some(2),
                math: CpuMath::DeterministicFma,
            },
            &budget,
        )
        .expect("cpu engine");
        let cfg = EmeansConfig::new(k);
        let fit_f32 = cpu_eng.fit(&mapped_f32, &cfg).expect("cpu fit mapped f32");
        let fit_bf16 = cpu_eng
            .fit(&mapped_bf16, &cfg)
            .expect("cpu fit mapped bf16");
        let (lbls, obj) = cpu_eng
            .assign(&mapped_bf16, &fit_f32.centroids, false, 64)
            .expect("cpu assign mapped bf16");
        assert_eq!(lbls.len(), n);
        assert!(obj.is_finite() && obj > 0.0);
        drop(fit_f32);
        drop(fit_bf16);
        drop(cpu_eng);
        drop(mapped_f32);
        drop(mapped_bf16);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    // 2. Real cross-runtime `tessl::SharedAlloc` (`F32` and `Bf16`) -> `ClusterDevice::Metal`
    //    AND shared tight `Budget` trim-handler eviction + `ON_EMEANS_WORKER` deadlock-freedom!
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let metal_probe_budget = Budget::new(128 << 20);
        let Some(metal_eng) =
            open_metal_engine(MetalPrecision::Bf16AssignF32Stats, &metal_probe_budget)
        else {
            return;
        };

        let producer_rt = tessl::GpuRuntime::new_inference().expect("producer GpuRuntime");
        let producer_budget = Budget::new(64 << 20);

        // Allocate `F32` SharedAlloc on foreign `producer_rt`.
        let alloc_f32 = tessl::shared_buf::SharedAlloc::new(&producer_rt, x_vec.len() * 4)
            .expect("SharedAlloc f32");
        let sbuf_f32 = alloc_f32.shared_buf();
        {
            let mut w = sbuf_f32.host_write().expect("host_write f32");
            w.as_f32_mut().expect("as_f32_mut").copy_from_slice(&x_vec);
        }
        let mapped_f32 = Tensor::from_mapped(
            Arc::new(sbuf_f32.clone()),
            &[n, d],
            DType::F32,
            &producer_budget,
        )
        .expect("Tensor::from_mapped f32");

        // Allocate `Bf16` SharedAlloc on foreign `producer_rt`.
        let alloc_bf16 = tessl::shared_buf::SharedAlloc::new(&producer_rt, x_vec.len() * 2)
            .expect("SharedAlloc bf16");
        let sbuf_bf16 = alloc_bf16.shared_buf();
        {
            let mut w = sbuf_bf16.host_write().expect("host_write bf16");
            let dst = w.as_bytes_mut();
            for (i, &v) in x_vec.iter().enumerate() {
                let b = f32_to_bf16_rte_bits(v).to_ne_bytes();
                dst[2 * i] = b[0];
                dst[2 * i + 1] = b[1];
            }
        }
        let mapped_bf16 = Tensor::from_mapped(
            Arc::new(sbuf_bf16.clone()),
            &[n, d],
            DType::Bf16,
            &producer_budget,
        )
        .expect("Tensor::from_mapped bf16");

        let mut cfg = EmeansConfig::new(k);
        cfg.nepochs = 2.0;
        cfg.batch_size = Some(64);
        cfg.assignment_chunk_size = 32;

        // Exercise `fit` (including subsampled `max_points_per_centroid`), `assign`, and `EmeansSession`
        // across foreign `SharedBuf` tensors without poisoning either runtime!
        let fit_a = metal_eng
            .fit(&mapped_f32, &cfg)
            .expect("metal fit mapped f32");
        let fit_b = metal_eng
            .fit(&mapped_bf16, &cfg)
            .expect("metal fit mapped bf16");
        let mut sub_cfg = cfg.clone();
        sub_cfg.max_points_per_centroid = Some(16);
        let fit_sub = metal_eng
            .fit(&mapped_bf16, &sub_cfg)
            .expect("metal subsampled fit mapped bf16");
        assert_eq!(
            fit_sub
                .labels
                .iter()
                .filter(|&&l| l == UNASSIGNED_LABEL)
                .count(),
            n - 16 * k
        );

        let (assign_lbls, assign_obj) = metal_eng
            .assign(&mapped_bf16, &fit_a.centroids, false, 64)
            .expect("metal assign mapped bf16");
        assert_eq!(assign_lbls.len(), n);
        assert!(assign_obj.is_finite() && assign_obj > 0.0);

        // Stream foreign `Mapped` chunks through `EmeansSession` on Metal.
        let init_bf16 = Tensor::from_bf16_bits(
            &x_vec[..k * d]
                .iter()
                .map(|&v| f32_to_bf16_rte_bits(v))
                .collect::<Vec<_>>(),
            &[k, d],
            &producer_budget,
        )
        .expect("init_bf16");
        let mut one_ep_cfg = cfg.clone();
        one_ep_cfg.nepochs = 1.0;
        one_ep_cfg.batch_size = Some(n);
        let mut sess =
            EmeansSession::begin(&metal_eng, n, &init_bf16, &one_ep_cfg).expect("metal sess");
        sess.accumulate(&mapped_bf16)
            .expect("metal sess accum mapped bf16");
        sess.update().expect("metal sess update");
        sess.end_epoch().expect("metal sess end_epoch");
        let _ = sess.finalize().expect("metal sess finalize");

        // Verify zero leaked `HostLease`s (`host_write` requires `lease == 0`) and healthy `producer_rt`.
        assert!(
            sbuf_f32.host_write().is_ok(),
            "f32 SharedBuf read lease must be released before fit/assign returns"
        );
        assert!(
            sbuf_bf16.host_write().is_ok(),
            "bf16 SharedBuf read lease must be released before fit/assign returns"
        );
        producer_rt
            .synchronize()
            .expect("producer_rt must remain unpoisoned");

        drop(fit_a);
        drop(fit_b);
        drop(fit_sub);
        drop(metal_eng);
        assert_eq!(metal_probe_budget.live_bytes().unwrap(), 0);

        // Shared tight `Budget` test:
        // 1) Pre-register a sibling `CountingTrimHandler` BEFORE `EmeansEngine::new` to prove
        //    `EmeansEngine` uses `add_trim_handler` (preserving prior handlers, never `set_trim_handler`).
        // 2) Warm GPU pool free-list cache (`cache_cap = Cap / 4 = 2 MiB` on an `8 MiB` budget).
        // 3) Allocate a `7 MiB` host `Tensor::zeros` on `shared_budget`, which only fits after
        //    `MetalWorkerTrimHandler` synchronously trims the cached GPU buffers!
        // 4) Trigger GPU allocation pressure on `"ojas-cluster-metal"` itself while host holds `7 MiB`
        //    so `ON_EMEANS_WORKER` guard proves zero worker self-deadlock!
        let shared_cap: u64 = 8 << 20;
        let shared_budget = Budget::new(shared_cap);
        let sibling_handler = Arc::new(CountingTrimHandler {
            calls: AtomicUsize::new(0),
        });
        shared_budget.add_trim_handler(sibling_handler.clone());

        let tight_eng = EmeansEngine::new(
            ClusterDevice::Metal {
                precision: MetalPrecision::F32,
            },
            &shared_budget,
        )
        .expect("tight Metal engine");

        // Fit a dataset (~512 KiB) so GPU buffers populate the `2 MiB` pool cache after `drop(fit)`.
        let big_n = 4_096;
        let big_d = 32;
        let big_vec = make_golden_mixture(big_n, big_d, 16, 0x9999_1111, 2.0, 0.3, false);
        let big_x = Tensor::from_f32(&big_vec, &[big_n, big_d], &producer_budget).expect("big_x");
        let warm_fit = tight_eng
            .fit(&big_x, &EmeansConfig::new(16))
            .expect("warm_fit");
        drop(warm_fit);

        let cached_before = shared_budget.live_bytes().unwrap();
        assert!(
            cached_before > 0,
            "GPU pool free-list cache should hold > 0 bytes before host pressure"
        );

        // Request `shared_cap - cached_before / 2` bytes on host so it CANNOT fit without trimming `tight_eng`!
        let host_need_elems = ((shared_cap - (cached_before / 2)) as usize) / 4;
        let host_big = Tensor::zeros(&[host_need_elems], DType::F32, &shared_budget)
            .expect("host Tensor::zeros must succeed via MetalWorkerTrimHandler eviction");
        assert!(
            sibling_handler.calls.load(Ordering::Relaxed) >= 1,
            "sibling trim handler registered before EmeansEngine::new must be preserved (add_trim_handler)"
        );

        // While `host_big` holds almost the entire `8 MiB` budget, run a GPU `fit` that exceeds remaining
        // headroom on `"ojas-cluster-metal"` — proving `ON_EMEANS_WORKER` returns `0` without deadlocking
        // and surfaces a clean `OjasError::CapacityExceeded`!
        let gpu_over = tight_eng.fit(&big_x, &EmeansConfig::new(16));
        assert!(
            matches!(gpu_over, Err(OjasError::CapacityExceeded { .. })),
            "expected clean CapacityExceeded without worker self-deadlock, got {gpu_over:?}"
        );

        drop(host_big);
        // After releasing `host_big`, `tight_eng` remains completely healthy for subsequent fits!
        let recovered_fit = tight_eng
            .fit(&big_x, &EmeansConfig::new(16))
            .expect("engine must recover cleanly after CapacityExceeded");
        drop(recovered_fit);
        drop(tight_eng);
        assert_eq!(shared_budget.live_bytes().unwrap(), 0);
    }
}

fn check_mid_session_error_recovery(device: ClusterDevice) {
    let io_budget = Budget::new(32 << 20);
    let n = 128;
    let d = 16;
    let k = 8;
    let x_vec = make_golden_mixture(n, d, k, 0x4242_4242, 2.0, 0.3, false);
    let init_c = Tensor::from_f32(&x_vec[..k * d], &[k, d], &io_budget).expect("init_c");
    let valid_32 = Tensor::from_f32(&x_vec[..32 * d], &[32, d], &io_budget).expect("valid_32");

    let mut nan_vec = x_vec[..32 * d].to_vec();
    nan_vec[17 * d + 3] = f32::NAN;
    let nan_chunk = Tensor::from_f32(&nan_vec, &[32, d], &io_budget).expect("nan_chunk");

    let mut inf_vec = x_vec[..32 * d].to_vec();
    inf_vec[5 * d + 1] = f32::NEG_INFINITY;
    let inf_chunk = Tensor::from_f32(&inf_vec, &[32, d], &io_budget).expect("inf_chunk");

    let mut norm_ovf_vec = x_vec[..32 * d].to_vec();
    norm_ovf_vec[9 * d] = 2.0e19; // finite f32 & bf16, but `v * v = 4e38 > f32::MAX` overflows row_norm2!
    let norm_ovf_chunk =
        Tensor::from_f32(&norm_ovf_vec, &[32, d], &io_budget).expect("norm_ovf_chunk");

    let mut cfg = EmeansConfig::new(k);
    cfg.nepochs = 1.0;
    cfg.batch_size = Some(64);
    cfg.assignment_chunk_size = 32;

    let session_budget = Budget::new(8 << 20);
    let engine = match device {
        ClusterDevice::Cpu { .. } => EmeansEngine::new(device, &session_budget).expect("cpu eng"),
        ClusterDevice::Metal { precision } => match open_metal_engine(precision, &session_budget) {
            Some(eng) => eng,
            None => return,
        },
    };

    let mut session = EmeansSession::begin(&engine, n, &init_c, &cfg).expect("session begin");

    // Accumulate first valid half-batch (32 of 64 rows).
    let (s0, l0) = session.accumulate(&valid_32).expect("first half-batch");
    assert_eq!(s0.remaining_in_batch, 32);
    assert!(l0.is_some());

    // 1. Feed NaN, Inf, and row-norm-overflow chunks: each MUST return `OjasError::NonFinite`
    //    WITHOUT corrupting `remaining_in_batch` or latching GPU faults!
    for bad in [&nan_chunk, &inf_chunk, &norm_ovf_chunk] {
        let err = session
            .accumulate(bad)
            .expect_err("must refuse non-finite/overflow");
        assert!(
            matches!(err, OjasError::NonFinite { op: "emeans" }),
            "expected NonFinite on {device:?}, got {err:?}"
        );
        assert_eq!(session.remaining_in_batch(), 32);
        assert_eq!(session.remaining_in_epoch(), 96);
    }

    // 2. Saturate `session_budget` with an external `Scratch<u8>` reservation so `accumulate`
    //    cannot stage/allocate and refuses with `OjasError::CapacityExceeded`, then release it!
    //    (On Metal, first trim any free-list cache so the budget headroom is completely pinned.)
    let _ = engine.trim_pool();
    let live_now = session_budget.live_bytes().unwrap();
    let headroom = session_budget.cap_bytes().saturating_sub(live_now);
    let blocker = Scratch::<u8>::try_alloc(headroom as usize, &session_budget).expect("blocker");
    let cap_err = session
        .accumulate(&valid_32)
        .expect_err("must refuse CapacityExceeded when budget is full");
    assert!(
        matches!(cap_err, OjasError::CapacityExceeded { .. }),
        "expected CapacityExceeded on {device:?}, got {cap_err:?}"
    );
    assert_eq!(session.remaining_in_batch(), 32);
    assert_eq!(session.remaining_in_epoch(), 96);
    drop(blocker);

    // 3. Recover cleanly on the very same `session` and drive it to `finalize()`!
    let (s1, l1) = session
        .accumulate(&valid_32)
        .expect("session must recover after NonFinite and CapacityExceeded");
    assert!(s1.batch_ready);
    assert!(l1.is_some());
    session.update().expect("batch 0 update");

    // Batch 1 (remaining 64 rows).
    session.accumulate(&valid_32).expect("b1 part 1");
    session.accumulate(&valid_32).expect("b1 part 2");
    session.update().expect("batch 1 update");
    let stat = session.end_epoch().expect("end_epoch");
    assert!(stat.mean_objective.is_finite());

    let fit = session.finalize().expect("finalize after recovery");
    assert_eq!(fit.labels.len(), n);
    assert!(fit
        .centroids
        .f32_slice()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
    drop(fit);
    drop(engine);
    assert_eq!(session_budget.live_bytes().unwrap(), 0);
}

#[test]
fn test_dtype_and_session_state_machine_refusals() {
    let _guard = serial();

    let budget = Budget::new(32 << 20);
    let n = 64;
    let d = 8;
    let k = 4;
    let x_vec = make_golden_mixture(n, d, k, 0x7777_8888, 2.0, 0.25, false);
    let x_valid = Tensor::from_f32(&x_vec, &[n, d], &budget).expect("x_valid");
    let c_valid = Tensor::from_f32(&x_vec[..k * d], &[k, d], &budget).expect("c_valid");

    let u32_tensor = Tensor::from_u32(&vec![1_u32; n * d], &[n, d], &budget).expect("u32_tensor");
    let dev_tensor = Tensor::from_device(
        Arc::new(MockDiscreteDevice {
            byte_len: n * d * 4,
        }),
        &[n, d],
        DType::F32,
        &budget,
    )
    .expect("dev_tensor");
    let rank1_tensor = Tensor::from_f32(&x_vec[..d], &[d], &budget).expect("rank1");
    let wrong_d_chunk =
        Tensor::from_f32(&vec![0.5_f32; 16 * (d + 2)], &[16, d + 2], &budget).expect("wrong_d");
    let empty_chunk = Tensor::from_f32(&[], &[0, d], &budget).expect("empty_chunk");

    let mut cfg = EmeansConfig::new(k);
    cfg.nepochs = 2.0;
    cfg.batch_size = Some(32);
    cfg.assignment_chunk_size = 16;

    let eng = EmeansEngine::new(
        ClusterDevice::Cpu {
            threads: Some(2),
            math: CpuMath::DeterministicFma,
        },
        &budget,
    )
    .expect("cpu engine");

    // 1. DType::U32 -> OjasError::Dtype
    assert!(matches!(
        eng.fit(&u32_tensor, &cfg),
        Err(OjasError::Dtype { op: "emeans", .. })
    ));
    assert!(matches!(
        eng.assign(&u32_tensor, &c_valid, false, 16),
        Err(OjasError::Dtype { op: "emeans", .. })
    ));
    assert!(matches!(
        EmeansSession::begin(&eng, n, &u32_tensor, &cfg),
        Err(OjasError::Dtype { op: "emeans", .. })
    ));

    // 2. DeviceBuffer tensor -> OjasError::Placement
    assert!(matches!(
        eng.fit(&dev_tensor, &cfg),
        Err(OjasError::Placement { op: "emeans", .. })
    ));
    assert!(matches!(
        eng.assign(&dev_tensor, &c_valid, false, 16),
        Err(OjasError::Placement { op: "emeans", .. })
    ));
    assert!(matches!(
        EmeansSession::begin(&eng, n, &dev_tensor, &cfg),
        Err(OjasError::Placement { op: "emeans", .. })
    ));

    // 3. Shape & chunk_size = 0 refusals
    assert!(matches!(
        eng.fit(&rank1_tensor, &cfg),
        Err(OjasError::Shape { op: "emeans", .. })
    ));
    assert!(matches!(
        eng.assign(&x_valid, &c_valid, false, 0),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));

    // 4. Session state machine boundary refusals
    let mut sess = EmeansSession::begin(&eng, n, &c_valid, &cfg).expect("begin");
    assert!(matches!(
        sess.accumulate(&u32_tensor),
        Err(OjasError::Dtype { op: "emeans", .. })
    ));
    assert!(matches!(
        sess.accumulate(&dev_tensor),
        Err(OjasError::Placement { op: "emeans", .. })
    ));
    assert!(matches!(
        sess.accumulate(&wrong_d_chunk),
        Err(OjasError::Shape { op: "emeans", .. })
    ));
    assert!(matches!(
        sess.accumulate(&empty_chunk),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));
    // Oversized chunk (`64 > remaining_in_batch = 32`)
    assert!(matches!(
        sess.accumulate(&x_valid),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));
    // Premature `update()` and `end_epoch()` at 0 rows
    assert!(matches!(
        sess.update(),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));
    assert!(matches!(
        sess.end_epoch(),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));

    let chunk_16 = Tensor::from_f32(&x_vec[..16 * d], &[16, d], &budget).expect("chunk_16");
    let chunk_32 = Tensor::from_f32(&x_vec[..32 * d], &[32, d], &budget).expect("chunk_32");

    // Partial batch (`16 < 32`) -> premature `update()` still refused without corrupting state!
    sess.accumulate(&chunk_16).expect("part 1");
    assert!(matches!(
        sess.update(),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));
    sess.accumulate(&chunk_16).expect("part 2");

    // Batch full (`remaining_in_batch == 0`) -> extra `accumulate` before `update()` refused!
    assert!(matches!(
        sess.accumulate(&chunk_16),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));
    sess.update().expect("batch 0 update");

    // Mid-epoch (`32 of 64` rows committed) -> premature `end_epoch()` refused!
    assert!(matches!(
        sess.end_epoch(),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));

    // Accumulate second batch (`remaining_in_epoch == 0` NOW, but `update()` not called yet!) ->
    // `end_epoch()` MUST refuse because the pending batch hasn't been applied!
    sess.accumulate(&chunk_32).expect("batch 1 accum");
    assert_eq!(sess.remaining_in_epoch(), 0);
    assert!(matches!(
        sess.end_epoch(),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));
    sess.update().expect("batch 1 update");
    // Extra `update()` when epoch is already complete -> refused!
    assert!(matches!(
        sess.update(),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));
    sess.end_epoch().expect("epoch 0 end");

    // Premature `finalize()` after only 1 of 2 epochs -> refuses with `OutOfRange` and releases memory!
    assert!(matches!(
        sess.finalize(),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));

    // 5. Mid-session recovery from `NonFinite` and `CapacityExceeded` on BOTH CPU and Metal!
    check_mid_session_error_recovery(ClusterDevice::Cpu {
        threads: Some(2),
        math: CpuMath::DeterministicFma,
    });
    check_mid_session_error_recovery(ClusterDevice::Metal {
        precision: MetalPrecision::Bf16,
    });
    check_mid_session_error_recovery(ClusterDevice::Metal {
        precision: MetalPrecision::F32,
    });
}

#[test]
fn test_upstream_numpy_emeans_golden_parity() {
    let _guard = serial();

    // Offline-computed golden constants from upstream `facebookresearch/faiss/contrib/e_means`
    // (`NumpyEmeans` driven over identical `SplitMix64` mixtures and epoch permutations).
    const GOLDEN_EUC_POST_FIT_MSE: f64 = 4.302_574_338_523_97;
    const GOLDEN_EUC_EPOCH_MEAN_OBJ: [f64; 3] = [4.670754883438, 4.427717731870, 4.373659264296];

    const GOLDEN_SPH_POST_FIT_OBJ: f64 = 0.036604895803129;
    const GOLDEN_SPH_EPOCH_MEAN_OBJ: [f64; 3] = [0.039904661593, 0.037409894635, 0.037214846117];

    const GOLDEN_SUB_POST_FIT_MSE: f64 = 2.675833845001627;
    const GOLDEN_SUB_UPSTREAM_PCG_TRAIN_MSE: f64 = 2.665907041873373;
    const GOLDEN_REV_POST_FIT_MSE: f64 = 9.964473660;

    let budget = Budget::new(128 << 20);

    let n_main = 2_048;
    let d_main = 32;
    let k_main = 16;
    let x_euc_vec = make_golden_mixture(
        n_main,
        d_main,
        k_main,
        0xCAFE_BABE_1234_5678,
        2.5,
        0.45,
        false,
    );
    let x_euc = Tensor::from_f32(&x_euc_vec, &[n_main, d_main], &budget).expect("x_euc");
    let init_euc = Tensor::from_f32(&x_euc_vec[..k_main * d_main], &[k_main, d_main], &budget)
        .expect("init_euc");
    let x_euc_bf16 = Tensor::from_bf16_bits(
        &x_euc_vec
            .iter()
            .map(|&v| f32_to_bf16_rte_bits(v))
            .collect::<Vec<_>>(),
        &[n_main, d_main],
        &budget,
    )
    .expect("x_euc_bf16");

    let x_sph_vec = make_golden_mixture(
        n_main,
        d_main,
        k_main,
        0xDEAD_BEEF_8765_4321,
        2.5,
        0.35,
        true,
    );
    let x_sph = Tensor::from_f32(&x_sph_vec, &[n_main, d_main], &budget).expect("x_sph");
    let init_sph = Tensor::from_f32(&x_sph_vec[..k_main * d_main], &[k_main, d_main], &budget)
        .expect("init_sph");

    let n_sub = 1_024;
    let d_sub = 24;
    let k_sub = 16;
    let x_sub_vec =
        make_golden_mixture(n_sub, d_sub, k_sub, 0x1122_3344_5566_7788, 2.5, 0.40, false);
    let x_sub = Tensor::from_f32(&x_sub_vec, &[n_sub, d_sub], &budget).expect("x_sub");
    let init_sub =
        Tensor::from_f32(&x_sub_vec[..k_sub * d_sub], &[k_sub, d_sub], &budget).expect("init_sub");
    let expected_keep = sample_subset_sorted(n_sub, 256, 77, &budget).expect("expected_keep");

    // Adversarial init with 8 live centroids + 8 far-away dead centroids (`50.0`) to force Gumbel split revival.
    let mut bad_init_vec = vec![50.0_f32; k_main * d_main];
    bad_init_vec[..8 * d_main].copy_from_slice(&x_euc_vec[..8 * d_main]);
    let bad_init = Tensor::from_f32(&bad_init_vec, &[k_main, d_main], &budget).expect("bad_init");

    let mut devices = vec![
        ClusterDevice::Cpu {
            threads: None,
            math: CpuMath::Fast,
        },
        ClusterDevice::Cpu {
            threads: Some(4),
            math: CpuMath::DeterministicFma,
        },
        ClusterDevice::Cpu {
            threads: Some(2),
            math: CpuMath::Exact,
        },
    ];
    if open_metal_engine(MetalPrecision::Bf16, &budget).is_some() {
        devices.push(ClusterDevice::Metal {
            precision: MetalPrecision::Bf16,
        });
        devices.push(ClusterDevice::Metal {
            precision: MetalPrecision::Bf16AssignF32Stats,
        });
        devices.push(ClusterDevice::Metal {
            precision: MetalPrecision::F32,
        });
    }

    for dev in devices {
        let is_bf16_lane = matches!(
            dev,
            ClusterDevice::Metal {
                precision: MetalPrecision::Bf16 | MetalPrecision::Bf16AssignF32Stats
            }
        );
        let rel_tol = if is_bf16_lane { 1.5e-2 } else { 2.0e-4 };

        let engine = EmeansEngine::new(dev, &budget).expect("engine");

        // 1. Euclidean golden trajectory (`nepochs = 2.6, seed = 42, batch_law = Auto -> Throughput`)
        let mut cfg_euc = EmeansConfig::new(k_main);
        cfg_euc.nepochs = 2.6;
        cfg_euc.seed = 42;
        let fit_euc = engine
            .fit_with_init(&x_euc, &init_euc, &cfg_euc)
            .expect("fit_euc");
        assert_eq!(fit_euc.recipe.resolved_batch_law, BatchLaw::Throughput);
        assert_eq!(fit_euc.recipe.batch_size, 64);
        assert_eq!(fit_euc.recipe.epoch_rows, vec![2_048, 1_229, 2_048]);
        assert_eq!(fit_euc.labels.len(), n_main);
        assert!(fit_euc.labels.iter().all(|&l| (l as usize) < k_main));
        assert_eq!(fit_euc.cluster_counts.iter().sum::<u64>(), n_main as u64);

        let c_euc = fit_euc.centroids.f32_slice().expect("c_euc");
        let mse_euc = exact_f64_eval_objective(&x_euc_vec, n_main, d_main, c_euc, k_main, false);
        let rel_euc = (mse_euc - GOLDEN_EUC_POST_FIT_MSE).abs() / GOLDEN_EUC_POST_FIT_MSE;
        assert!(
            rel_euc <= rel_tol,
            "{dev:?} Euclidean MSE {mse_euc:.9} drifted from golden {GOLDEN_EUC_POST_FIT_MSE:.9} (rel={rel_euc:.3e})"
        );
        for (ep, &g_ep) in GOLDEN_EUC_EPOCH_MEAN_OBJ.iter().enumerate() {
            let got = fit_euc.epoch_stats[ep].mean_objective;
            let rel = (got - g_ep).abs() / g_ep;
            assert!(
                rel <= rel_tol,
                "{dev:?} Euclidean epoch {ep} obj {got:.9} vs golden {g_ep:.9} (rel={rel:.3e})"
            );
        }

        // Verify standalone `emeans_assign` matches `exact_f64_eval_objective`.
        let (assign_lbls, assign_total_obj) =
            emeans_assign(&x_euc, &fit_euc.centroids, false, dev, &budget).expect("emeans_assign");
        assert_eq!(assign_lbls.len(), n_main);
        let assign_mse = assign_total_obj / (n_main as f64);
        assert!(
            ((assign_mse - mse_euc).abs() / mse_euc) <= rel_tol,
            "{dev:?} emeans_assign MSE {assign_mse:.9} vs exact {mse_euc:.9}"
        );

        // Verify `DType::Bf16` input tensor via one-shot `emeans_fit`.
        let fit_bf16_in = emeans_fit(&x_euc_bf16, &cfg_euc, dev, &budget).expect("emeans_fit bf16");
        let mse_bf16_in = exact_f64_eval_objective(
            &x_euc_vec,
            n_main,
            d_main,
            fit_bf16_in.centroids.f32_slice().unwrap(),
            k_main,
            false,
        );
        assert!(
            mse_bf16_in < 10.0,
            "{dev:?} Bf16 input emeans_fit MSE {mse_bf16_in} unexpectedly high"
        );

        // 2. Spherical golden trajectory (`nepochs = 2.6, seed = 42, spherical = true`)
        let mut cfg_sph = EmeansConfig::new(k_main);
        cfg_sph.nepochs = 2.6;
        cfg_sph.spherical = true;
        cfg_sph.seed = 42;
        let fit_sph = engine
            .fit_with_init(&x_sph, &init_sph, &cfg_sph)
            .expect("fit_sph");
        let c_sph = fit_sph.centroids.f32_slice().expect("c_sph");
        for j in 0..k_main {
            let row = &c_sph[j * d_main..(j + 1) * d_main];
            let norm2: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            assert!(
                (norm2.sqrt() - 1.0).abs() < 1.0e-5,
                "{dev:?} closing spherical centroid {j} not unit norm ({norm2})"
            );
        }
        let obj_sph = exact_f64_eval_objective(&x_sph_vec, n_main, d_main, c_sph, k_main, true);
        let rel_sph = (obj_sph - GOLDEN_SPH_POST_FIT_OBJ).abs() / GOLDEN_SPH_POST_FIT_OBJ;
        let sph_tol = if is_bf16_lane { 2.5e-2 } else { 5.0e-4 };
        assert!(
            rel_sph <= sph_tol,
            "{dev:?} Spherical obj {obj_sph:.9} drifted from golden {GOLDEN_SPH_POST_FIT_OBJ:.9} (rel={rel_sph:.3e})"
        );
        for (ep, &g_ep) in GOLDEN_SPH_EPOCH_MEAN_OBJ.iter().enumerate() {
            let got = fit_sph.epoch_stats[ep].mean_objective;
            let rel = (got - g_ep).abs() / g_ep;
            assert!(
                rel <= sph_tol,
                "{dev:?} Spherical epoch {ep} obj {got:.9} vs golden {g_ep:.9} (rel={rel:.3e})"
            );
        }

        // 3. `max_points_per_centroid = Some(16)` subsampling golden (`effective_n = 256` of `1024`)
        let mut cfg_sub = EmeansConfig::new(k_sub);
        cfg_sub.nepochs = 3.0;
        cfg_sub.seed = 77;
        cfg_sub.max_points_per_centroid = Some(16);
        let fit_sub = engine
            .fit_with_init(&x_sub, &init_sub, &cfg_sub)
            .expect("fit_sub");
        assert_eq!(fit_sub.recipe.global_n, n_sub);
        assert_eq!(fit_sub.recipe.effective_n, 256);
        assert_eq!(fit_sub.labels.len(), n_sub);
        let unassigned_cnt = fit_sub
            .labels
            .iter()
            .filter(|&&l| l == UNASSIGNED_LABEL)
            .count();
        assert_eq!(unassigned_cnt, n_sub - 256);
        for &kept_row in expected_keep.as_slice() {
            assert!(
                (fit_sub.labels[kept_row as usize] as usize) < k_sub,
                "{dev:?} sampled row {kept_row} must have valid cluster label"
            );
        }
        assert_eq!(fit_sub.cluster_counts.iter().sum::<u64>(), 256);
        let mse_sub = exact_f64_eval_objective(
            &x_sub_vec,
            n_sub,
            d_sub,
            fit_sub.centroids.f32_slice().unwrap(),
            k_sub,
            false,
        );
        let rel_sub = (mse_sub - GOLDEN_SUB_POST_FIT_MSE).abs() / GOLDEN_SUB_POST_FIT_MSE;
        assert!(
            rel_sub <= rel_tol,
            "{dev:?} Subsampled MSE {mse_sub:.9} drifted from golden {GOLDEN_SUB_POST_FIT_MSE:.9} (rel={rel_sub:.3e})"
        );
        let rel_sub_pcg =
            (mse_sub - GOLDEN_SUB_UPSTREAM_PCG_TRAIN_MSE).abs() / GOLDEN_SUB_UPSTREAM_PCG_TRAIN_MSE;
        assert!(
            rel_sub_pcg <= 1.5e-2,
            "{dev:?} Subsampled MSE {mse_sub:.9} drifted from upstream PCG64 train {GOLDEN_SUB_UPSTREAM_PCG_TRAIN_MSE:.9} (rel={rel_sub_pcg:.3e})"
        );

        // 4. Dead-cluster Gumbel split revival (`bad_init` with 8 dead centroids at `50.0`)
        let mut cfg_rev = EmeansConfig::new(k_main);
        cfg_rev.nepochs = 4.0;
        cfg_rev.seed = 42;
        let fit_rev = engine
            .fit_with_init(&x_euc, &bad_init, &cfg_rev)
            .expect("fit_rev");
        let total_revived: usize = fit_rev.epoch_stats.iter().map(|s| s.revived_clusters).sum();
        assert!(
            total_revived >= 8,
            "{dev:?} expected >= 8 dead clusters revived from bad_init, got {total_revived}"
        );
        assert!(
            fit_rev.cluster_counts.iter().all(|&cnt| cnt > 0),
            "{dev:?} all 16 clusters should be occupied after revival, got {:?}",
            fit_rev.cluster_counts
        );
        let init_bad_mse =
            exact_f64_eval_objective(&x_euc_vec, n_main, d_main, &bad_init_vec, k_main, false);
        let mse_rev = exact_f64_eval_objective(
            &x_euc_vec,
            n_main,
            d_main,
            fit_rev.centroids.f32_slice().unwrap(),
            k_main,
            false,
        );
        assert!(
            mse_rev < init_bad_mse * 0.25,
            "{dev:?} revived MSE {mse_rev:.6} failed to improve substantially over bad_init ({init_bad_mse:.6})"
        );
        let rel_rev = (mse_rev - GOLDEN_REV_POST_FIT_MSE).abs() / GOLDEN_REV_POST_FIT_MSE;
        assert!(
            rel_rev <= rel_tol,
            "{dev:?} revived MSE {mse_rev:.9} drifted across substrates from {GOLDEN_REV_POST_FIT_MSE:.9} (rel={rel_rev:.3e})"
        );
    }
}
