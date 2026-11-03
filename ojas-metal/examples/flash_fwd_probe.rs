//! Flash forward probe (bench/results/2026-11-03-flash-fwd-specialize).
//!
//! One fixed source, built into every binary of the A/B, so only the library
//! differs between them. Two modes:
//!
//! - `flash_fwd_probe hash`: one line per case with an FNV-1a 64 hash over the
//!   f32 bit patterns of the output, then the log-sum-exp where the call
//!   returns one. The cases are SpikeTrack's memory cross-attention call
//!   (`attention_forward` at q `[1,8,5184,32]` x k,v `[1,8,36400,32]`) and small
//!   forced-flash, biased, causal, windowed and bf16-accumulation calls with
//!   ragged query tails. Every binary of the A/B must print the same hashes.
//! - `flash_fwd_probe time <warmups> <runs> <tag>`: times three arms, each call
//!   with a device sync on both sides, odd rounds in reverse order:
//!   `x36400` (the SpikeTrack call), `x15600` (the same at 3 memory frames) and
//!   `ctl` (`attention_forward` at q,k,v `[1,8,5184,32]`, which routes to
//!   tessl's `encoder_attn` and is untouched by the change: the control).
//!   It also prints the flash pipelines each arm encoded.

#[cfg(target_os = "macos")]
mod probe {
    use std::process::Command;
    use std::time::Instant;

    use ojas_core::{AttentionOpts, Backend, Budget, Tensor};
    use ojas_metal::MetalBackend;

    /// The flash forward's pipelines at head dim 32 (the bf16 entry exists only
    /// after the change; an unknown name counts 0).
    const FLASH_D32: [&str; 2] = ["ojas_attn_fwd_d32", "ojas_attn_fwd_bf16"];

    fn load() -> String {
        let out = Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .expect("sysctl vm.loadavg");
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .trim_matches(|c| c == '{' || c == '}')
            .trim()
            .to_string()
    }

    fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 40) as f32 / (1u64 << 24) as f32;
                (2.0 * u - 1.0) * scale
            })
            .collect()
    }

    fn fnv(h: &mut u64, xs: &[f32]) {
        for x in xs {
            for byte in x.to_bits().to_le_bytes() {
                *h ^= u64::from(byte);
                *h = h.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
    }

    fn pct(sorted: &[f64], p: f64) -> f64 {
        let i = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[i]
    }

    struct Ctx {
        budget: Budget,
        metal: MetalBackend,
    }

    impl Ctx {
        fn up(&self, shape: &[usize], seed: u64, scale: f32) -> Tensor {
            let n = shape.iter().product();
            let t = Tensor::from_f32(&values(n, seed, scale), shape, &self.budget).expect("host");
            self.metal.upload(&t).expect("upload")
        }

        fn hash(&self, ts: &[&Tensor]) -> u64 {
            self.metal.sync().expect("sync");
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for t in ts {
                let v = t
                    .to_host(&self.budget)
                    .expect("host")
                    .to_f32_vec()
                    .expect("f32");
                fnv(&mut h, &v);
            }
            h
        }

        fn flash_counts(&self) -> Vec<u64> {
            FLASH_D32
                .iter()
                .map(|n| self.metal.kernel_dispatches(n))
                .collect()
        }
    }

    fn cross(c: &Ctx, name: &str, qs: [usize; 4], tk: usize, bias: bool) {
        let [b, h, tq, d] = qs;
        let q = c.up(&qs, 11, 1.0);
        let k = c.up(&[b, h, tk, d], 12, 1.0);
        let v = c.up(&[b, h, tk, d], 13, 1.0);
        let bt = bias.then(|| c.up(&[b, h, tq, tk], 14, 0.5));
        let opts = AttentionOpts {
            causal: false,
            bias: bt.as_ref(),
            ..Default::default()
        };
        let before = c.flash_counts();
        let (o, lse) = c
            .metal
            .attention_forward_saving_flash(&q, &k, &v, opts)
            .expect("flash");
        let h = c.hash(&[&o, &lse]);
        println!(
            "hash\t{name}\t{h:016x}\tflash_d32 {:?} -> {:?}",
            before,
            c.flash_counts()
        );
    }

    fn causal(c: &Ctx, name: &str, s: [usize; 4], window: Option<usize>) {
        let q = c.up(&s, 21, 1.0);
        let k = c.up(&s, 22, 1.0);
        let v = c.up(&s, 23, 1.0);
        let (o, lse) = c
            .metal
            .causal_sdpa_forward(&q, &k, &v, window)
            .expect("causal");
        let h = c.hash(&[&o, &lse]);
        println!("hash\t{name}\t{h:016x}");
    }

    fn hashes(c: &Ctx) {
        let (b, h, tq, tk, d) = (1usize, 8usize, 5184usize, 36400usize, 32usize);
        let q = c.up(&[b, h, tq, d], 101, 1.0);
        let k = c.up(&[b, h, tk, d], 102, 1.0);
        let v = c.up(&[b, h, tk, d], 103, 1.0);
        let opts = || AttentionOpts {
            causal: false,
            ..Default::default()
        };
        let before = c.flash_counts();
        let o = c.metal.attention_forward(&q, &k, &v, opts()).expect("fwd");
        let ho = c.hash(&[&o]);
        let after = c.flash_counts();
        let (o2, lse2) = c
            .metal
            .attention_forward_saving(&q, &k, &v, opts())
            .expect("saving");
        let hs = c.hash(&[&o2, &lse2]);
        println!("hash\tspiketrack_x36400_fwd\t{ho:016x}\tflash_d32 {before:?} -> {after:?}");
        println!("hash\tspiketrack_x36400_saving\t{hs:016x}");
        drop((q, k, v, o, o2, lse2));

        cross(c, "x_d32_q100_k333", [1, 2, 100, 32], 333, false);
        cross(c, "x_d32_q5184_k15600", [1, 8, 5184, 32], 15600, false);
        cross(c, "x_d64_q65_k129_b2h3", [2, 3, 65, 64], 129, false);
        cross(c, "x_d128_q40_k70", [1, 2, 40, 128], 70, false);
        cross(c, "x_d32_q100_k333_bias", [1, 2, 100, 32], 333, true);
        causal(c, "causal_d32_t200", [1, 2, 200, 32], None);
        causal(c, "causal_d64_t150_w37", [1, 2, 150, 64], Some(37));
        c.metal.set_bf16_accumulation(true);
        cross(c, "bf16_x_d32_q100_k333", [1, 2, 100, 32], 333, false);
        cross(c, "bf16_x_d32_q100_k333_bias", [1, 2, 100, 32], 333, true);
        causal(c, "bf16_causal_d32_t200", [1, 2, 200, 32], None);
        c.metal.set_bf16_accumulation(false);
    }

    type Arm<'a> = (&'static str, Box<dyn Fn() + 'a>);

    fn time(c: &Ctx, warmups: usize, runs: usize, tag: &str) {
        let load_start = load();
        let (h, tq, d) = (8usize, 5184usize, 32usize);
        let q = c.up(&[1, h, tq, d], 101, 1.0);
        let k = c.up(&[1, h, 36400, d], 102, 1.0);
        let v = c.up(&[1, h, 36400, d], 103, 1.0);
        let k3 = c.up(&[1, h, 15600, d], 104, 1.0);
        let v3 = c.up(&[1, h, 15600, d], 105, 1.0);
        let ks = c.up(&[1, h, tq, d], 106, 1.0);
        let vs = c.up(&[1, h, tq, d], 107, 1.0);
        let opts = || AttentionOpts {
            causal: false,
            ..Default::default()
        };
        let m = &c.metal;
        let arms: Vec<Arm<'_>> = vec![
            (
                "x36400",
                Box::new(|| drop(m.attention_forward(&q, &k, &v, opts()).expect("x36400"))),
            ),
            (
                "x15600",
                Box::new(|| drop(m.attention_forward(&q, &k3, &v3, opts()).expect("x15600"))),
            ),
            (
                "ctl",
                Box::new(|| drop(m.attention_forward(&q, &ks, &vs, opts()).expect("ctl"))),
            ),
        ];
        let mut ms: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        let mut ran: Vec<Vec<u64>> = vec![vec![0; FLASH_D32.len()]; arms.len()];
        for round in 0..warmups + runs {
            for step in 0..arms.len() {
                let i = if round % 2 == 0 {
                    step
                } else {
                    arms.len() - 1 - step
                };
                m.sync().expect("sync");
                let before = c.flash_counts();
                let t = Instant::now();
                (arms[i].1)();
                m.sync().expect("sync");
                let dt = t.elapsed().as_secs_f64() * 1e3;
                for (r, (a, b)) in ran[i].iter_mut().zip(c.flash_counts().iter().zip(&before)) {
                    *r += a - b;
                }
                if round >= warmups {
                    ms[i].push(dt);
                }
            }
        }
        let load_end = load();
        for (i, arm) in arms.iter().enumerate() {
            let mut s = ms[i].clone();
            s.sort_by(f64::total_cmp);
            println!(
                "probe\t{tag}\t{}\tmin {:.3}\tp10 {:.3}\tmedian {:.3}\tp90 {:.3}\tn {}\tflash_d32 {:?}\tload_start {}\tload_end {}",
                arm.0,
                s[0],
                pct(&s, 0.1),
                pct(&s, 0.5),
                pct(&s, 0.9),
                s.len(),
                ran[i],
                load_start,
                load_end
            );
        }
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let budget = Budget::new(24 << 30);
        let metal = MetalBackend::new(budget.clone()).expect("Metal device");
        let c = Ctx { budget, metal };
        match args.first().map(String::as_str) {
            Some("hash") => hashes(&c),
            Some("time") => {
                let num =
                    |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
                let tag = args.get(3).cloned().unwrap_or_default();
                time(&c, num(1, 3), num(2, 15), &tag);
            }
            _ => {
                eprintln!("usage: flash_fwd_probe hash | time <warmups> <runs> <tag>");
                std::process::exit(2);
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    probe::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("flash_fwd_probe needs macOS and Metal");
}
