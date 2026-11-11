use ojas_cluster::{
    choose_splits, epoch_permutation, gumbel_key, prescaled_unit_normalize_f32,
    prescaled_unit_normalize_f64, resolve_recipe, round_power_of_two, sample_subset_sorted,
    scheduled_alpha, separate_collapsed_split_bf16, separate_collapsed_split_f32, BatchLaw,
    ClusterDevice, CpuMath, EmeansConfig, MetalPrecision, SplitMix64, UNASSIGNED_LABEL,
};
use ojas_core::{Budget, OjasError};

#[test]
fn test_canonical_v1_scaling_laws_match_upstream_python() {
    assert_eq!(UNASSIGNED_LABEL, u32::MAX);
    assert_eq!(MetalPrecision::default(), MetalPrecision::Bf16);
    assert_eq!(CpuMath::default(), CpuMath::Fast);
    assert_eq!(BatchLaw::default(), BatchLaw::Auto);
    assert_eq!(
        ClusterDevice::default(),
        ClusterDevice::Cpu {
            threads: None,
            math: CpuMath::Fast,
        }
    );

    let def_cfg = EmeansConfig::default();
    assert_eq!(def_cfg.k, 1);
    assert_eq!(def_cfg.nepochs, 2.6);
    assert_eq!(def_cfg.batch_law, BatchLaw::Auto);
    assert_eq!(def_cfg.batch_size, None);
    assert_eq!(def_cfg.alpha0, None);
    assert!(!def_cfg.spherical);
    assert_eq!(def_cfg.seed, 42);
    assert_eq!(def_cfg.max_points_per_centroid, None);
    assert_eq!(def_cfg.assignment_chunk_size, 16_384);
    assert_eq!(def_cfg.unused_threshold, None);
    assert_eq!(def_cfg.split_epsilon, 1.0e-4);
    assert_eq!(def_cfg.init_usage, 1.0 / 65_536.0);
    assert!(def_cfg.capture_assignments);

    // Log-space power-of-two rounding (`faiss/contrib/e_means/base.py`)
    assert_eq!(round_power_of_two(-5.0), 1);
    assert_eq!(round_power_of_two(0.0), 1);
    assert_eq!(round_power_of_two(1.0), 1);
    assert_eq!(round_power_of_two(1.41), 1);
    assert_eq!(round_power_of_two(2.0_f64.sqrt()), 2);
    assert_eq!(round_power_of_two(2.0), 2);
    assert_eq!(round_power_of_two(2.82), 2);
    assert_eq!(round_power_of_two(2.83), 4);
    assert_eq!(round_power_of_two(4690.0), 4096);
    assert_eq!(round_power_of_two(f64::MAX), 1usize << (usize::BITS - 1));

    // 1. Throughput law under Auto when d > 16 (`n = 100_000, d = 128, k = 1024, nepochs = 2.6`)
    let cfg = EmeansConfig::new(1024);
    let r_tp = resolve_recipe(100_000, 128, &cfg).expect("resolve throughput recipe");
    assert_eq!(r_tp.global_n, 100_000);
    assert_eq!(r_tp.effective_n, 100_000);
    assert_eq!(r_tp.k, 1024);
    assert_eq!(r_tp.d, 128);
    assert_eq!(r_tp.resolved_batch_law, BatchLaw::Throughput);
    assert_eq!(r_tp.batch_size, 4096);
    assert_eq!(r_tp.unused_threshold, 0.1);
    assert_eq!(r_tp.epoch_rows, vec![100_000, 60_000, 100_000]);
    assert_eq!(r_tp.steps_per_pass, 25);
    assert_eq!(r_tp.epoch_steps, vec![25, 15, 25]);
    assert_eq!(r_tp.total_steps, 65);
    let want_alpha_tp = (0.8244_f64
        * 1024.0_f64.powf(-0.4617)
        * 4096.0_f64.powf(0.9003)
        * 100_000.0_f64.powf(-0.4498)
        * 2.6_f64.powf(-0.1057))
    .clamp(1e-4, 1.0) as f32;
    assert_eq!(r_tp.alpha0.to_bits(), want_alpha_tp.to_bits());

    // 2. Quality law under Auto when d <= 16 (`d = 16`)
    let r_qual = resolve_recipe(100_000, 16, &cfg).expect("resolve quality recipe");
    assert_eq!(r_qual.resolved_batch_law, BatchLaw::Quality);
    assert_eq!(r_qual.batch_size, 512); // C_B_qual = C_B_tp / 8 => 4096 / 8 = 512
    assert_eq!(r_qual.unused_threshold, 0.1);

    // 3. Quality vs Throughput half-n cap behavior (`n = 100, k = 64, nepochs = 500.0`)
    let mut cfg_cap = EmeansConfig::new(64);
    cfg_cap.nepochs = 500.0;
    let r_qual_cap = resolve_recipe(100, 8, &cfg_cap).expect("quality cap");
    assert_eq!(r_qual_cap.batch_size, 50); // b0 = 64 capped at effective_n / 2 = 50
    cfg_cap.batch_law = BatchLaw::Throughput;
    let r_tp_cap = resolve_recipe(100, 8, &cfg_cap).expect("throughput cap");
    assert_eq!(r_tp_cap.batch_size, 64); // b0 = 512 capped at round_power_of_two(50.0) = 64

    // 4. `unused_threshold = 0.0` when `d <= 2` (disabled in 1D/2D per upstream `base.py`)
    let r_d1 = resolve_recipe(1_000, 1, &EmeansConfig::new(16)).unwrap();
    let r_d2 = resolve_recipe(1_000, 2, &EmeansConfig::new(16)).unwrap();
    let r_d3 = resolve_recipe(1_000, 3, &EmeansConfig::new(16)).unwrap();
    assert_eq!(r_d1.unused_threshold, 0.0);
    assert_eq!(r_d2.unused_threshold, 0.0);
    assert_eq!(r_d3.unused_threshold, 0.1);

    // Explicit override of `unused_threshold` at `d = 2` is honored
    let mut cfg_ut = EmeansConfig::new(16);
    cfg_ut.unused_threshold = Some(0.25);
    assert_eq!(
        resolve_recipe(1_000, 2, &cfg_ut).unwrap().unused_threshold,
        0.25
    );

    // 5. `max_points_per_centroid` subsampling cap
    let mut cfg_sub = EmeansConfig::new(10);
    cfg_sub.max_points_per_centroid = Some(16);
    let r_sub = resolve_recipe(1_000, 32, &cfg_sub).unwrap();
    assert_eq!(r_sub.global_n, 1_000);
    assert_eq!(r_sub.effective_n, 160);

    // 6. Fractional `round_ties_even()` `epoch_rows` and closing full-pass placement
    let mut cfg_frac = EmeansConfig::new(2);
    cfg_frac.nepochs = 1.0;
    assert_eq!(
        resolve_recipe(100, 4, &cfg_frac).unwrap().epoch_rows,
        vec![100]
    );

    cfg_frac.nepochs = 0.25; // 0.25 * 10 = 2.5 -> round_ties_even = 2
    assert_eq!(
        resolve_recipe(10, 4, &cfg_frac).unwrap().epoch_rows,
        vec![2]
    );

    cfg_frac.nepochs = 0.75; // 0.75 * 10 = 7.5 -> round_ties_even = 8
    assert_eq!(
        resolve_recipe(10, 4, &cfg_frac).unwrap().epoch_rows,
        vec![8]
    );

    cfg_frac.nepochs = 1.25; // whole = 1, 0.25 * 2 = 0.5 -> round_ties_even = 0 -> omitted!
    assert_eq!(resolve_recipe(2, 4, &cfg_frac).unwrap().epoch_rows, vec![2]);

    cfg_frac.nepochs = 1.75; // whole = 1, 0.75 * 2 = 1.5 -> round_ties_even = 2 -> [2, 2]
    assert_eq!(
        resolve_recipe(2, 4, &cfg_frac).unwrap().epoch_rows,
        vec![2, 2]
    );

    cfg_frac.nepochs = 3.4; // 2 full passes, 40-row fractional pass, 1 closing full pass
    assert_eq!(
        resolve_recipe(100, 4, &cfg_frac).unwrap().epoch_rows,
        vec![100, 100, 40, 100]
    );

    // 7. Cosine decay `scheduled_alpha`
    let a0 = 0.8_f32;
    assert_eq!(scheduled_alpha(a0, 0, 100).to_bits(), a0.to_bits());
    let mid = scheduled_alpha(a0, 50, 100);
    assert!((mid - 0.55 * a0).abs() < 1e-6, "got {mid}");
    let end = scheduled_alpha(a0, 100, 100);
    assert!((end - 0.1 * a0).abs() < 1e-6, "got {end}");
    let past_end = scheduled_alpha(a0, 250, 100);
    assert_eq!(past_end.to_bits(), end.to_bits());
    assert_eq!(scheduled_alpha(a0, 0, 0).to_bits(), a0.to_bits());
}

#[test]
fn test_recipe_refuses_zero_rounded_fractional_epoch_and_invalid_bounds() {
    let assert_emeans_out_of_range = |res: Result<_, OjasError>, label: &str| {
        match res {
        Err(OjasError::OutOfRange { op: "emeans", .. }) => {}
        other => panic!("expected Err(OjasError::OutOfRange {{ op: \"emeans\", .. }}) for {label}, got {other:?}"),
    }
    };

    // Zero-rounded fractional epoch: nepochs = 1e-6 on n = 100 -> round_ties_even(1e-4) == 0
    let mut cfg = EmeansConfig::new(4);
    cfg.nepochs = 1.0e-6;
    assert_emeans_out_of_range(resolve_recipe(100, 8, &cfg), "zero-rounded nepochs = 1e-6");

    // Non-positive, NaN, infinite, or absurdly huge nepochs
    for bad_nepochs in [
        0.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
    ] {
        let mut c = EmeansConfig::new(4);
        c.nepochs = bad_nepochs;
        assert_emeans_out_of_range(
            resolve_recipe(100, 8, &c),
            &format!("nepochs = {bad_nepochs}"),
        );
    }

    // k == 0, d == 0, global_n < k
    assert_emeans_out_of_range(resolve_recipe(100, 8, &EmeansConfig::new(0)), "k == 0");
    assert_emeans_out_of_range(resolve_recipe(100, 0, &EmeansConfig::new(4)), "d == 0");
    assert_emeans_out_of_range(resolve_recipe(3, 8, &EmeansConfig::new(4)), "global_n < k");

    // max_points_per_centroid == Some(0), batch_size == Some(0), assignment_chunk_size == 0
    let mut c_mpc = EmeansConfig::new(4);
    c_mpc.max_points_per_centroid = Some(0);
    assert_emeans_out_of_range(
        resolve_recipe(100, 8, &c_mpc),
        "max_points_per_centroid == 0",
    );

    let mut c_bs = EmeansConfig::new(4);
    c_bs.batch_size = Some(0);
    assert_emeans_out_of_range(resolve_recipe(100, 8, &c_bs), "batch_size == 0");

    let mut c_chunk = EmeansConfig::new(4);
    c_chunk.assignment_chunk_size = 0;
    assert_emeans_out_of_range(
        resolve_recipe(100, 8, &c_chunk),
        "assignment_chunk_size == 0",
    );

    // alpha0 outside (0.0, 1.0] or non-finite
    for bad_a in [
        0.0_f32,
        -0.5,
        1.0001,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let mut c = EmeansConfig::new(4);
        c.alpha0 = Some(bad_a);
        assert_emeans_out_of_range(resolve_recipe(100, 8, &c), &format!("alpha0 = {bad_a}"));
    }
    let mut ok_a1 = EmeansConfig::new(4);
    ok_a1.alpha0 = Some(1.0);
    assert_eq!(resolve_recipe(100, 8, &ok_a1).unwrap().alpha0, 1.0);

    // unused_threshold < 0.0 or non-finite
    for bad_ut in [-0.001_f32, -1.0, f32::NAN, f32::INFINITY] {
        let mut c = EmeansConfig::new(4);
        c.unused_threshold = Some(bad_ut);
        assert_emeans_out_of_range(
            resolve_recipe(100, 8, &c),
            &format!("unused_threshold = {bad_ut}"),
        );
    }

    // split_epsilon and init_usage < 0.0 or non-finite
    for bad_v in [-1.0e-4_f32, f32::NAN, f32::INFINITY] {
        let mut c_eps = EmeansConfig::new(4);
        c_eps.split_epsilon = bad_v;
        assert_emeans_out_of_range(resolve_recipe(100, 8, &c_eps), "bad split_epsilon");

        let mut c_init = EmeansConfig::new(4);
        c_init.init_usage = bad_v;
        assert_emeans_out_of_range(resolve_recipe(100, 8, &c_init), "bad init_usage");
    }
}

#[test]
fn test_split_separation_handles_neg_zero_subnormals_and_max_finite() {
    // 1. [-0.0; 4], [0.0; 4], [f32::from_bits(1); 4], and [-f32::from_bits(1); 4]
    //    must step child[0] to 2^-62 (`0x2080_0000`) with zero NaNs in both f32 and bf16!
    for base in [
        [-0.0_f32; 4],
        [0.0_f32; 4],
        [f32::from_bits(1); 4],
        [f32::from_bits(0x8000_0001); 4],
    ] {
        let eps = 1.0e-4_f32;
        let mut donor = [0.0_f32; 4];
        let mut child_f32 = [0.0_f32; 4];
        for col in 0..4 {
            let off = if col % 2 == 0 { eps } else { -eps };
            donor[col] = base[col] * (1.0 - off);
            child_f32[col] = base[col] * (1.0 + off);
        }
        let mut child_bf16 = child_f32;

        separate_collapsed_split_f32(&donor, &mut child_f32);
        assert!(
            child_f32.iter().all(|v| v.is_finite()),
            "NaN/Inf in f32: {child_f32:?}"
        );
        assert_eq!(child_f32[0].to_bits(), 0x2080_0000);

        separate_collapsed_split_bf16(&donor, &mut child_bf16);
        assert!(
            child_bf16.iter().all(|v| v.is_finite()),
            "NaN/Inf in bf16: {child_bf16:?}"
        );
        assert_eq!(child_bf16[0].to_bits(), 0x2080_0000);
    }

    // 2. [1.0; 4] with +/-1e-4 collapses in bf16 (`0x3f80`) and must step child[0] to `0x3f81_0000`
    let eps = 1.0e-4_f32;
    let donor_one = [
        1.0 * (1.0 - eps),
        1.0 * (1.0 + eps),
        1.0 * (1.0 - eps),
        1.0 * (1.0 + eps),
    ];
    let mut child_one = [
        1.0 * (1.0 + eps),
        1.0 * (1.0 - eps),
        1.0 * (1.0 + eps),
        1.0 * (1.0 - eps),
    ];
    separate_collapsed_split_bf16(&donor_one, &mut child_one);
    assert_eq!(child_one[0].to_bits(), 0x3f81_0000);

    // Also direct [1.0; 4] == [1.0; 4]
    let mut direct_bf16 = [1.0_f32; 4];
    separate_collapsed_split_bf16(&[1.0_f32; 4], &mut direct_bf16);
    assert_eq!(direct_bf16[0].to_bits(), 0x3f81_0000);

    // 3. [f32::MAX; 4] and [-f32::MAX; 4] step inward without overflowing to Inf
    let mut child_max_f32 = [f32::MAX; 4];
    separate_collapsed_split_f32(&[f32::MAX; 4], &mut child_max_f32);
    assert!(child_max_f32[0].is_finite());
    assert_eq!(child_max_f32[0].to_bits(), 0x7f7f_fffe);
    assert!(child_max_f32[0] < f32::MAX);

    let mut child_nmax_f32 = [-f32::MAX; 4];
    separate_collapsed_split_f32(&[-f32::MAX; 4], &mut child_nmax_f32);
    assert!(child_nmax_f32[0].is_finite());
    assert_eq!(child_nmax_f32[0].to_bits(), 0xff7f_fffe);
    assert!(child_nmax_f32[0] > -f32::MAX);

    // Max finite bf16 (`0x7f7f_0000`, `-0x7f7f_0000`) and `[f32::MAX; 4]` / `[-f32::MAX; 4]` in bf16
    let bf16_max = f32::from_bits(0x7f7f_0000);
    let mut child_bmax = [bf16_max; 4];
    separate_collapsed_split_bf16(&[bf16_max; 4], &mut child_bmax);
    assert!(child_bmax[0].is_finite());
    assert_eq!(child_bmax[0].to_bits(), 0x7f7e_0000);

    let mut child_nbmax = [-bf16_max; 4];
    separate_collapsed_split_bf16(&[-bf16_max; 4], &mut child_nbmax);
    assert!(child_nbmax[0].is_finite());
    assert_eq!(child_nbmax[0].to_bits(), 0xff7e_0000);

    let mut child_fmax_in_bf16 = [f32::MAX; 4];
    separate_collapsed_split_bf16(&[f32::MAX; 4], &mut child_fmax_in_bf16);
    assert!(child_fmax_in_bf16[0].is_finite());

    let mut child_nfmax_in_bf16 = [-f32::MAX; 4];
    separate_collapsed_split_bf16(&[-f32::MAX; 4], &mut child_nfmax_in_bf16);
    assert!(child_nfmax_in_bf16[0].is_finite());

    // 4. `-min_normal` (`0x8080_0000`) steps to `+0.0f32` (`0x0000_0000`), never a subnormal
    let neg_min_norm = [-f32::MIN_POSITIVE; 4];
    let mut child_nmn_f32 = neg_min_norm;
    separate_collapsed_split_f32(&neg_min_norm, &mut child_nmn_f32);
    assert_eq!(child_nmn_f32[0].to_bits(), 0x0000_0000);

    let mut child_nmn_bf16 = neg_min_norm;
    separate_collapsed_split_bf16(&neg_min_norm, &mut child_nmn_bf16);
    assert_eq!(child_nmn_bf16[0].to_bits(), 0x0000_0000);
}

#[test]
fn test_rng_permutations_gumbel_and_spherical_prescale() {
    // 1. Budget-charged `sample_subset_sorted`
    let budget = Budget::new(64 * 1024);
    let sub1 = sample_subset_sorted(500, 64, 42, &budget).expect("sample_subset_sorted");
    let sub2 = sample_subset_sorted(500, 64, 42, &budget).expect("repeat sample_subset_sorted");
    let sub_other = sample_subset_sorted(500, 64, 43, &budget).expect("other seed");
    assert_eq!(sub1.len(), 64);
    assert_eq!(sub1.as_slice(), sub2.as_slice());
    assert_ne!(sub1.as_slice(), sub_other.as_slice());
    assert!(
        sub1.as_slice().windows(2).all(|w| w[0] < w[1]),
        "must be strictly ascending"
    );
    assert!(sub1.as_slice().iter().all(|&idx| idx < 500));
    assert_eq!(budget.live_bytes().unwrap(), 3 * 64 * 4);
    drop(sub1);
    drop(sub2);
    drop(sub_other);
    assert_eq!(budget.live_bytes().unwrap(), 0);

    // Tiny budget refuses `sample_subset_sorted` without leaking bytes
    let tiny = Budget::new(64);
    let err_sub = sample_subset_sorted(1_000, 8, 42, &tiny).unwrap_err();
    assert!(matches!(err_sub, OjasError::CapacityExceeded { .. }));
    assert_eq!(tiny.live_bytes().unwrap(), 0);

    // Full subset (`target_n == global_n`) and out-of-range (`target_n > global_n`)
    let full_sub = sample_subset_sorted(16, 16, 42, &budget).unwrap();
    assert_eq!(full_sub.as_slice(), &(0..16u32).collect::<Vec<_>>());
    drop(full_sub);
    assert!(matches!(
        sample_subset_sorted(16, 17, 42, &budget),
        Err(OjasError::OutOfRange { op: "emeans", .. })
    ));
    assert_eq!(budget.live_bytes().unwrap(), 0);

    // 2. Budget-charged `epoch_permutation`
    let perm_e0_a = epoch_permutation(257, 99, 0, &budget).expect("perm epoch 0");
    let perm_e0_b = epoch_permutation(257, 99, 0, &budget).expect("perm epoch 0 repeat");
    let perm_e1 = epoch_permutation(257, 99, 1, &budget).expect("perm epoch 1");
    assert_eq!(perm_e0_a.as_slice(), perm_e0_b.as_slice());
    assert_ne!(perm_e0_a.as_slice(), perm_e1.as_slice());
    assert_eq!(budget.live_bytes().unwrap(), 3 * 257 * 4);

    let mut sorted = perm_e0_a.as_slice().to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..257u32).collect::<Vec<_>>());
    drop(perm_e0_a);
    drop(perm_e0_b);
    drop(perm_e1);
    assert_eq!(budget.live_bytes().unwrap(), 0);

    let err_perm = epoch_permutation(1_000, 99, 0, &tiny).unwrap_err();
    assert!(matches!(err_perm, OjasError::CapacityExceeded { .. }));
    assert_eq!(tiny.live_bytes().unwrap(), 0);

    // 3. `gumbel_key` SplitMix64 formula + Metal `revival_gumbel_key` bit-identity + `choose_splits`
    let mut sm = SplitMix64::new(0x1234_5678_9abc_def0);
    assert_ne!(sm.next_u64(), sm.next_u64());
    for step in [0_u64, 4, 9, 1024] {
        for cid in [0_u32, 1, 7, 511, 65_535] {
            let g_lo = gumbel_key(2.0, 0xfeed_face_cafe_babe, step, cid);
            let g_hi = gumbel_key(4.0, 0xfeed_face_cafe_babe, step, cid);
            assert!(g_lo.is_finite() && g_hi.is_finite());
            assert!((g_hi - g_lo - std::f32::consts::LN_2).abs() < 1e-6);

            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                let gpu_g =
                    tessl::emeans::revival_gumbel_key(2.0, 0xfeed_face_cafe_babe, step, cid);
                assert_eq!(
                    g_lo.to_bits(),
                    gpu_g.to_bits(),
                    "CPU gumbel_key must match tessl::emeans::revival_gumbel_key bit-for-bit"
                );
            }
        }
    }

    let usage = [8.0_f32, 0.01, 12.0, 16.0, 0.0, 20.0];
    // mean = 56.01 / 6 = 9.335, thresh = 0.2 * 9.335 = 1.867 -> unused = [1, 4], donors = [0, 2, 3, 5]
    let seed = 0xfeed_face_cafe_babe_u64;
    let step = 9_u64;
    let splits = choose_splits(&usage, 0.2, seed, step);
    assert_eq!(splits.len(), 2);
    assert_eq!(splits[0].1, 1);
    assert_eq!(splits[1].1, 4);
    let mut want_donors: Vec<(f32, u32)> = [0_u32, 2, 3, 5]
        .into_iter()
        .map(|j| (gumbel_key(usage[j as usize], seed, step, j), j))
        .collect();
    want_donors.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    assert_eq!(splits[0].0, want_donors[0].1);
    assert_eq!(splits[1].0, want_donors[1].1);
    assert!(choose_splits(&usage, 0.0, seed, step).is_empty());

    // 4. `prescaled_unit_normalize_f64` (and `f32`) on `1e-300`, `1e300`, subnormal, and `0.0`
    let mut dst = [0.0_f32; 4];
    prescaled_unit_normalize_f64(&[1.0e-300, -1.0e-300, 1.0e-300, -1.0e-300], &mut dst);
    assert_eq!(dst, [0.5, -0.5, 0.5, -0.5]);

    prescaled_unit_normalize_f64(&[1.0e300, 1.0e300, -1.0e300, -1.0e300], &mut dst);
    assert_eq!(dst, [0.5, 0.5, -0.5, -0.5]);

    prescaled_unit_normalize_f64(&[0.0, -0.0, 0.0, 0.0], &mut dst);
    assert_eq!(dst, [0.0, 0.0, 0.0, 0.0]);
    assert!(dst.iter().all(|v| v.is_finite()));

    let mut dst_sub = [0.0_f32; 2];
    prescaled_unit_normalize_f64(&[f64::from_bits(1), 0.0], &mut dst_sub);
    assert_eq!(dst_sub, [1.0, 0.0]);

    prescaled_unit_normalize_f32(&[1.0e-30, -1.0e-30, 1.0e-30, -1.0e-30], &mut dst);
    assert_eq!(dst, [0.5, -0.5, 0.5, -0.5]);

    // Non-finite corruption (`[1.0, NaN]`, `[0.0, NaN]`, `[1.0, Inf]`, `[1.0, -Inf]`) must
    // propagate `NaN` across `dst[..d]` rather than being laundered into finite `0.0`s.
    for bad64 in [
        [1.0_f64, f64::NAN],
        [f64::NAN, 1.0_f64],
        [0.0_f64, f64::NAN],
        [1.0_f64, f64::INFINITY],
        [1.0_f64, f64::NEG_INFINITY],
    ] {
        let mut out2 = [1.0_f32; 2];
        prescaled_unit_normalize_f64(&bad64, &mut out2);
        assert!(
            out2.iter().all(|v| v.is_nan()),
            "expected all-NaN propagation for f64 {bad64:?}, got {out2:?}"
        );
    }

    for bad32 in [
        [1.0_f32, f32::NAN],
        [f32::NAN, 1.0_f32],
        [0.0_f32, f32::NAN],
        [1.0_f32, f32::INFINITY],
        [1.0_f32, f32::NEG_INFINITY],
    ] {
        let mut out2 = [1.0_f32; 2];
        prescaled_unit_normalize_f32(&bad32, &mut out2);
        assert!(
            out2.iter().all(|v| v.is_nan()),
            "expected all-NaN propagation for f32 {bad32:?}, got {out2:?}"
        );
    }
}

#[test]
fn test_resolve_recipe_rejects_unassigned_label_k() {
    let sentinel_k = UNASSIGNED_LABEL as usize;

    // 1. Even when `global_n >= k`, `k == UNASSIGNED_LABEL as usize` (and `k > UNASSIGNED_LABEL`)
    //    must be rejected so `UNASSIGNED_LABEL` (`u32::MAX`) can never collide with a cluster ID.
    let err_large_n = resolve_recipe(
        sentinel_k.saturating_add(16),
        1,
        &EmeansConfig::new(sentinel_k),
    )
    .expect_err("resolve_recipe must reject k == UNASSIGNED_LABEL even when global_n >= k");
    assert!(
        matches!(
            &err_large_n,
            OjasError::OutOfRange { op: "emeans", detail } if detail.contains("UNASSIGNED_LABEL")
        ),
        "expected OutOfRange mentioning UNASSIGNED_LABEL, got {err_large_n:?}"
    );

    // 2. Upfront entry check: with tiny `global_n = 1, d = 0`, `k >= UNASSIGNED_LABEL` is rejected
    //    at function entry before `d == 0` or `global_n < k`.
    let err_tiny = resolve_recipe(1, 0, &EmeansConfig::new(sentinel_k))
        .expect_err("resolve_recipe must reject k == UNASSIGNED_LABEL upfront");
    assert!(
        matches!(
            &err_tiny,
            OjasError::OutOfRange { op: "emeans", detail } if detail.contains("UNASSIGNED_LABEL")
        ),
        "expected upfront OutOfRange mentioning UNASSIGNED_LABEL, got {err_tiny:?}"
    );

    if let Some(above_sentinel) = sentinel_k.checked_add(1) {
        let err_above = resolve_recipe(
            above_sentinel.saturating_add(16),
            1,
            &EmeansConfig::new(above_sentinel),
        )
        .expect_err("resolve_recipe must reject k > UNASSIGNED_LABEL");
        assert!(matches!(
            err_above,
            OjasError::OutOfRange { op: "emeans", .. }
        ));
    }
}
