//! Dump the window starts that `ojas_data::BatchSampler` draws, so the torch
//! oracle (`ojas-oracle/python/batches.py`) trains on the same batches.
//!
//! ```text
//! cargo run --release -p ojas-oracle --example dump_batch_starts -- \
//!     --bin tokens.bin --seed 1337 --steps 40 --batch 2 --accum 2 --seq 32 \
//!     --out batch_starts.json --rows
//! ```
//!
//! nanolab's `Batcher` samples starts with replacement from torch's RNG; ojas
//! samples without replacement through a keyed Feistel permutation. Neither
//! reproduces the other, so the ojas side is the source of truth and torch
//! replays it.
//!
//! Output (`ojas-batch-starts-v1`, one JSON object, numbers only in arrays):
//! - `starts`: flat, `steps * accum * batch` token offsets, ordered
//!   `(step, micro, row)`; the row `r` of micro-batch `m` of step `s` is
//!   `starts[(s * accum + m) * batch + r]`.
//! - `rows` (with `--rows`): the `seq_len + 1` tokens of every window in the
//!   same order, exactly as `BatchSampler::next_batch` returned them
//!   (`x` = first `seq_len`, `y` = last `seq_len`).
//! - `bin_fnv1a64`: FNV-1a 64 of the whole bin file, as 16 hex digits, so a
//!   dump cannot be replayed against a different bin unnoticed.
//!
//! Every micro-batch is checked before it is written: the starts walked with
//! `window_start` must give, through `TokenBin::read_into`, exactly the
//! tokens `next_batch` returned.

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ojas_data::{BatchSampler, SamplerConfig, TokenBin, FINEWEB_HEADER_BYTES};

/// Largest integer every JSON reader holds exactly (an f64 mantissa).
const MAX_EXACT_JSON_INT: u64 = 1 << 53;
/// A dump is a test artifact; refuse anything that would not fit in memory
/// comfortably or in a fixture.
const MAX_WINDOWS: u64 = 1 << 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinFormat {
    Headerless,
    Fineweb,
}

impl BinFormat {
    fn name(self) -> &'static str {
        match self {
            BinFormat::Headerless => "headerless-u16le",
            BinFormat::Fineweb => "fineweb-u16le",
        }
    }

    fn header_bytes(self) -> u64 {
        match self {
            BinFormat::Headerless => 0,
            BinFormat::Fineweb => FINEWEB_HEADER_BYTES,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Args {
    bin: PathBuf,
    format: BinFormat,
    seed: u64,
    steps: u64,
    batch: usize,
    accum: usize,
    seq_len: usize,
    out: PathBuf,
    rows: bool,
}

const USAGE: &str = "usage: dump_batch_starts --bin PATH [--format headerless|fineweb] --seed N \
--steps N --batch B --accum K --seq T --out PATH [--rows]";

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut bin = None;
    let mut format = BinFormat::Headerless;
    let mut seed = None;
    let mut steps = None;
    let mut batch = None;
    let mut accum = None;
    let mut seq_len = None;
    let mut out = None;
    let mut rows = false;
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        if flag == "--rows" {
            rows = true;
            continue;
        }
        let value = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))?;
        let num = |v: &str| -> Result<u64, String> {
            v.parse::<u64>()
                .map_err(|_| format!("{flag}: {v:?} is not a non-negative integer"))
        };
        match flag.as_str() {
            "--bin" => bin = Some(PathBuf::from(value)),
            "--format" => {
                format = match value.as_str() {
                    "headerless" => BinFormat::Headerless,
                    "fineweb" => BinFormat::Fineweb,
                    other => return Err(format!("--format: unknown {other:?}")),
                }
            }
            "--seed" => seed = Some(num(value)?),
            "--steps" => steps = Some(num(value)?),
            "--batch" => batch = Some(num(value)?),
            "--accum" => accum = Some(num(value)?),
            "--seq" => seq_len = Some(num(value)?),
            "--out" => out = Some(PathBuf::from(value)),
            other => return Err(format!("unknown flag {other:?}\n{USAGE}")),
        }
    }
    let need = |v: Option<u64>, name: &str| v.ok_or_else(|| format!("missing {name}\n{USAGE}"));
    let to_usize = |v: u64, name: &str| {
        usize::try_from(v).map_err(|_| format!("{name} {v} does not fit in usize"))
    };
    let args = Args {
        bin: bin.ok_or_else(|| format!("missing --bin\n{USAGE}"))?,
        format,
        seed: need(seed, "--seed")?,
        steps: need(steps, "--steps")?,
        batch: to_usize(need(batch, "--batch")?, "--batch")?,
        accum: to_usize(need(accum, "--accum")?, "--accum")?,
        seq_len: to_usize(need(seq_len, "--seq")?, "--seq")?,
        out: out.ok_or_else(|| format!("missing --out\n{USAGE}"))?,
        rows,
    };
    if args.seed >= MAX_EXACT_JSON_INT {
        return Err(format!(
            "--seed {} is not exactly representable in JSON readers (>= 2^53)",
            args.seed
        ));
    }
    if args.steps == 0 || args.batch == 0 || args.accum == 0 || args.seq_len == 0 {
        return Err("steps, batch, accum and seq must be non-zero".to_string());
    }
    let windows = args
        .steps
        .checked_mul(args.accum as u64)
        .and_then(|n| n.checked_mul(args.batch as u64))
        .ok_or_else(|| "steps * accum * batch overflows".to_string())?;
    if windows > MAX_WINDOWS {
        return Err(format!(
            "{windows} windows exceed the {MAX_WINDOWS}-window dump cap"
        ));
    }
    Ok(args)
}

/// The dump, before serialization.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Dump {
    bin_tokens: u64,
    bin_fnv1a64: u64,
    windows_per_epoch: u64,
    cursor_after: (u64, u64),
    starts: Vec<u64>,
    rows: Option<Vec<u16>>,
}

fn fnv1a64_file(path: &Path) -> Result<u64, String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            return Ok(hash);
        }
        for &byte in &buf[..n] {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn dump(args: &Args) -> Result<Dump, String> {
    let bin = match args.format {
        BinFormat::Headerless => TokenBin::open_headerless(&args.bin),
        BinFormat::Fineweb => TokenBin::open_fineweb(&args.bin),
    }
    .map_err(|e| e.to_string())?;
    let cfg = SamplerConfig {
        seq_len: args.seq_len,
        batch: args.batch,
        seed: args.seed,
    };
    let mut sampler = BatchSampler::new(&bin, cfg).map_err(|e| e.to_string())?;
    let windows = sampler.windows_per_epoch();
    let t = args.seq_len;
    let mut starts = Vec::new();
    let mut rows = if args.rows { Some(Vec::new()) } else { None };
    let mut row = vec![0u16; t + 1];
    for _step in 0..args.steps {
        for _micro in 0..args.accum {
            let cursor = sampler.cursor();
            let (mut epoch, mut ordinal) = (cursor.shard, cursor.token_index);
            let first = starts.len();
            for _ in 0..args.batch {
                let start = sampler
                    .window_start(epoch, ordinal)
                    .map_err(|e| e.to_string())?;
                if start >= MAX_EXACT_JSON_INT {
                    return Err(format!("start {start} is >= 2^53"));
                }
                starts.push(start);
                ordinal += 1;
                if ordinal == windows {
                    ordinal = 0;
                    epoch = epoch
                        .checked_add(1)
                        .ok_or_else(|| "epoch counter overflows".to_string())?;
                }
            }
            let batch = sampler.next_batch().map_err(|e| e.to_string())?;
            for (r, &start) in starts[first..].iter().enumerate() {
                bin.read_into(start, &mut row).map_err(|e| e.to_string())?;
                let x = &batch.x[r * t..(r + 1) * t];
                let y = &batch.y[r * t..(r + 1) * t];
                let row_x = row[..t].iter().map(|&v| u32::from(v));
                let row_y = row[1..].iter().map(|&v| u32::from(v));
                if !row_x.eq(x.iter().copied()) || !row_y.eq(y.iter().copied()) {
                    return Err(format!(
                        "window start {start} does not reproduce next_batch row {r}"
                    ));
                }
                if let Some(rows) = rows.as_mut() {
                    rows.extend_from_slice(&row);
                }
            }
            let after = sampler.cursor();
            if (after.shard, after.token_index) != (epoch, ordinal) {
                return Err("cursor walk disagrees with next_batch".to_string());
            }
        }
    }
    let cursor = sampler.cursor();
    Ok(Dump {
        bin_tokens: bin.len(),
        bin_fnv1a64: fnv1a64_file(&args.bin)?,
        windows_per_epoch: windows,
        cursor_after: (cursor.shard, cursor.token_index),
        starts,
        rows,
    })
}

fn push_array<T: std::fmt::Display>(out: &mut String, name: &str, values: &[T]) {
    let items: Vec<String> = values.iter().map(ToString::to_string).collect();
    out.push_str(&format!(",\n  \"{name}\": [{}]", items.join(",")));
}

/// A JSON string for text from a fixed safe alphabet (no escaping needed).
fn quoted(text: &str) -> String {
    debug_assert!(!text.contains(['"', '\\']));
    format!("\"{text}\"")
}

fn render(args: &Args, dump: &Dump) -> String {
    let file_name = args
        .bin
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    // File names go into a JSON string; keep them to a safe alphabet so the
    // writer needs no escaping.
    let file_name: String = file_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    let mut out = String::from("{\n  \"format\": \"ojas-batch-starts-v1\"");
    let fields: [(&str, String); 13] = [
        (
            "generator",
            quoted("ojas-oracle/examples/dump_batch_starts"),
        ),
        ("sampler", quoted("ojas_data::BatchSampler")),
        ("bin", quoted(&file_name)),
        ("bin_format", quoted(args.format.name())),
        ("bin_header_bytes", args.format.header_bytes().to_string()),
        ("bin_tokens", dump.bin_tokens.to_string()),
        ("bin_fnv1a64", quoted(&format!("{:016x}", dump.bin_fnv1a64))),
        ("seed", args.seed.to_string()),
        ("seq_len", args.seq_len.to_string()),
        ("batch", args.batch.to_string()),
        ("accum", args.accum.to_string()),
        ("steps", args.steps.to_string()),
        ("windows_per_epoch", dump.windows_per_epoch.to_string()),
    ];
    for (name, value) in fields {
        out.push_str(&format!(",\n  \"{name}\": {value}"));
    }
    push_array(
        &mut out,
        "cursor_after",
        &[dump.cursor_after.0, dump.cursor_after.1],
    );
    push_array(&mut out, "starts", &dump.starts);
    if let Some(rows) = &dump.rows {
        push_array(&mut out, "rows", rows);
    }
    out.push_str("\n}\n");
    out
}

fn write_atomic(path: &Path, text: &str) -> Result<(), String> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut file = File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    file.write_all(text.as_bytes())
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

fn run(argv: &[String]) -> Result<(), String> {
    let args = parse_args(argv)?;
    let dump = dump(&args)?;
    write_atomic(&args.out, &render(&args, &dump))
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&argv) {
        eprintln!("dump_batch_starts: {e}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TmpDir(PathBuf);

    impl TmpDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("ojas-batch-starts-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            TmpDir(dir)
        }
    }

    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_bin(dir: &Path, tokens: &[u16]) -> PathBuf {
        let path = dir.join("tokens.bin");
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn argv(bin: &Path, out: &Path, extra: &[&str]) -> Vec<String> {
        let mut v: Vec<String> = [
            "--bin",
            bin.to_str().unwrap(),
            "--seed",
            "1337",
            "--steps",
            "3",
            "--batch",
            "2",
            "--accum",
            "2",
            "--seq",
            "4",
            "--out",
            out.to_str().unwrap(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    }

    #[test]
    fn starts_reproduce_the_sampler_and_cross_epochs() {
        let dir = TmpDir::new("epochs");
        // 21 tokens, T=4: 5 windows per epoch; 12 windows span 3 epochs.
        let tokens: Vec<u16> = (0..21u16).map(|i| i * 7 + 3).collect();
        let bin = write_bin(&dir.0, &tokens);
        let args = parse_args(&argv(&bin, &dir.0.join("o.json"), &["--rows"])).unwrap();
        let d = dump(&args).unwrap();
        assert_eq!(d.windows_per_epoch, 5);
        assert_eq!(d.starts.len(), 12);
        // Each epoch is a permutation of the window starts {0,4,8,12,16}.
        for epoch in d.starts.chunks(5).take(2) {
            let mut e = epoch.to_vec();
            e.sort_unstable();
            assert_eq!(e, vec![0, 4, 8, 12, 16]);
        }
        assert_eq!(d.cursor_after, (2, 2));
        let rows = d.rows.unwrap();
        for (i, &s) in d.starts.iter().enumerate() {
            let s = s as usize;
            assert_eq!(&rows[i * 5..(i + 1) * 5], &tokens[s..s + 5]);
        }
    }

    #[test]
    fn the_dump_matches_an_independent_sampler_run() {
        let dir = TmpDir::new("independent");
        let tokens: Vec<u16> = (0..400u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 7) as u16)
            .collect();
        let bin_path = write_bin(&dir.0, &tokens);
        let args = parse_args(&argv(&bin_path, &dir.0.join("o.json"), &[])).unwrap();
        let d = dump(&args).unwrap();
        let bin = TokenBin::open_headerless(&bin_path).unwrap();
        let mut s = BatchSampler::new(
            &bin,
            SamplerConfig {
                seq_len: 4,
                batch: 2,
                seed: 1337,
            },
        )
        .unwrap();
        for (i, micro) in d.starts.chunks(2).enumerate() {
            let b = s.next_batch().unwrap();
            for (r, &start) in micro.iter().enumerate() {
                let start = start as usize;
                let want: Vec<u32> = tokens[start..start + 4]
                    .iter()
                    .map(|&v| u32::from(v))
                    .collect();
                assert_eq!(&b.x[r * 4..(r + 1) * 4], &want[..], "micro {i} row {r}");
            }
        }
    }

    #[test]
    fn render_writes_the_documented_fields_and_hash() {
        let dir = TmpDir::new("render");
        let bin = write_bin(&dir.0, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let out = dir.0.join("o.json");
        let args = parse_args(&argv(&bin, &out, &["--rows"])).unwrap();
        run(&argv(&bin, &out, &["--rows"])).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        for key in [
            "\"format\": \"ojas-batch-starts-v1\"",
            "\"bin\": \"tokens.bin\"",
            "\"bin_header_bytes\": 0",
            "\"bin_tokens\": 9",
            "\"seed\": 1337",
            "\"windows_per_epoch\": 2",
            "\"starts\": [",
            "\"rows\": [",
        ] {
            assert!(text.contains(key), "missing {key} in {text}");
        }
        // FNV-1a 64 of the 18 bytes, computed independently of fnv1a64_file.
        let mut h: u64 = 0xcbf29ce484222325;
        for t in 1u16..=9 {
            for b in t.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
        assert!(text.contains(&format!("\"bin_fnv1a64\": \"{h:016x}\"")));
        assert_eq!(dump(&args).unwrap().bin_fnv1a64, h);
    }

    #[test]
    fn bad_arguments_are_refused() {
        let ok = |extra: &[&str]| {
            let mut v = argv(Path::new("b"), Path::new("o"), &[]);
            v.extend(extra.iter().map(|s| s.to_string()));
            parse_args(&v)
        };
        assert!(ok(&[]).is_ok());
        assert!(ok(&["--seed", "9007199254740992"]).is_err(), "2^53 seed");
        assert!(ok(&["--seed", "-1"]).is_err());
        assert!(ok(&["--steps", "0"]).is_err());
        assert!(ok(&["--format", "npy"]).is_err());
        assert!(ok(&["--bogus", "1"]).is_err());
        assert!(ok(&["--steps", "16777217", "--batch", "1", "--accum", "1"]).is_err());
        assert!(parse_args(&["--bin".to_string()]).is_err());
    }

    #[test]
    fn a_bin_shorter_than_one_window_is_an_error() {
        let dir = TmpDir::new("short");
        let bin = write_bin(&dir.0, &[1, 2, 3]);
        let args = parse_args(&argv(&bin, &dir.0.join("o.json"), &[])).unwrap();
        assert!(dump(&args).is_err());
    }
}
