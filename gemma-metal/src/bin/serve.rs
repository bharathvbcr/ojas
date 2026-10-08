//! Phase 6: minimal OpenAI-compatible HTTP serve stub.
//!
//! `--preset e4b|31b` selects config/MTP metadata. When mlx 31B Q4 is cached,
//! `--preset 31b` attempts Hot upload + real generate; otherwise falls back to
//! synthetic mini (documented in gates.md).
//!
//! ```bash
//! cargo run -p gemma-metal --release --bin serve -- --port 8787 --preset e4b
//! cargo run -p gemma-metal --release --bin serve -- --port 8787 --preset 31b
//! ```

use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use gemma_metal::config::Gemma4TextConfig;
use gemma_metal::diag;
use gemma_metal::forward::{greedy_decode_host, SyntheticE4bGraph};
use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::mtp::{b31_assistant_preset, e4b_assistant_preset};
use gemma_metal::quant::QuantScheme;
use gemma_metal::weights::{
    load_from_hf_dir, resolve_default_31b_mlx_cache, LoadOptions,
};

enum Backend {
    Synthetic(SyntheticE4bGraph),
    Hot(GpuDecodeSession),
}

struct AppState {
    preset: String,
    model_id: String,
    text: Gemma4TextConfig,
    backend: Backend,
    hot_note: String,
}

fn main() {
    diag::init();
    let argv: Vec<String> = env::args().collect();
    diag::log(
        "serve",
        format_args!("argv={argv:?} version={}", gemma_metal::version()),
    );

    let mut port = 8787u16;
    let mut preset = "e4b".to_string();
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--port" => {
                port = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(port);
            }
            "--preset" => {
                preset = args.next().unwrap_or_else(|| "e4b".into());
            }
            "-h" | "--help" => {
                eprintln!(
                    "serve [--port 8787] [--preset e4b|31b]\n\
                     OpenAI-compatible stub: GET /health, GET /v1/models, POST /v1/chat/completions\n\
                     Logs: GEMMA_METAL_LOG=1 (default ON) | GEMMA_METAL_LOG=0 silence\n\
                     Infer: GEMMA_METAL_INFER_LOG=1 (default ON) | =0 silence"
                );
                return;
            }
            _ => {
                diag::log("serve", format_args!("ignoring unknown arg {a}"));
            }
        }
    }
    diag::log("serve", format_args!("preset={preset} port={port}"));

    let (text, model_id) = match preset.as_str() {
        "31b" | "b31" => {
            preset = "31b".into();
            (Gemma4TextConfig::b31_preset(), "gemma-metal-31b".to_string())
        }
        _ => {
            preset = "e4b".into();
            (Gemma4TextConfig::e4b_preset(), "gemma-metal-e4b".to_string())
        }
    };

    #[allow(unused_assignments)]
    let mut hot_note = String::new();
    let backend = if preset == "31b" {
        match try_load_31b_hot() {
            Ok(sess) => {
                hot_note = "Hot Q4 loaded from mlx-community/gemma-4-31b-it-4bit".into();
                Backend::Hot(sess)
            }
            Err(e) => {
                hot_note = format!(
                    "31B Hot load unavailable ({e}); synthetic mini fallback — Ollama gemma4:31b-mlx ~12.3 tok/s Phase-0"
                );
                Backend::Synthetic(SyntheticE4bGraph::mini_parity().expect("mini graph"))
            }
        }
    } else {
        hot_note = "E4B serve uses synthetic mini unless client loads via bench --e4b".into();
        Backend::Synthetic(SyntheticE4bGraph::mini_parity().expect("mini graph"))
    };

    println!(
        "gemma-metal serve ({}) preset={preset} model_id={model_id}",
        gemma_metal::version()
    );
    println!(
        "  text shapes: hidden={} layers={} k_eq_v={} ple={} kv_share={} head_dim={}/{}",
        text.hidden_size,
        text.num_hidden_layers,
        text.attention_k_eq_v,
        text.has_ple(),
        text.num_kv_shared_layers,
        text.head_dim,
        text.global_head_dim
    );
    if preset == "31b" {
        let a = b31_assistant_preset();
        println!(
            "  31B MTP assistant stub: backbone={} draft_hidden={} centroids={}",
            a.backbone_hidden_size, a.text_config.hidden_size, a.num_centroids
        );
    } else {
        let a = e4b_assistant_preset();
        println!(
            "  E4B MTP assistant stub: backbone={} draft_hidden={} centroids={}",
            a.backbone_hidden_size, a.text_config.hidden_size, a.num_centroids
        );
    }
    println!("  backend: {hot_note}");

    let state = Mutex::new(AppState {
        preset: preset.clone(),
        model_id,
        text,
        backend,
        hot_note,
    });

    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("bind {addr}: {e}");
        std::process::exit(1);
    });
    println!("listening on http://{addr}");

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if let Err(e) = handle_client(s, &state) {
                    eprintln!("client error: {e}");
                }
            }
            Err(e) => eprintln!("accept: {e}"),
        }
    }
}

fn try_load_31b_hot() -> Result<GpuDecodeSession, String> {
    diag::log("serve", format_args!("▶ try_load_31b_hot"));
    let dir = match resolve_default_31b_mlx_cache() {
        Some(d) => {
            diag::log("serve", format_args!("31b cache={}", d.display()));
            d
        }
        None => {
            let msg = "no HF cache for mlx-community/gemma-4-31b-it-4bit".to_string();
            diag::err_msg("serve", "31b resolve", &msg);
            return Err(msg);
        }
    };
    let load_t0 = std::time::Instant::now();
    let banks = load_from_hf_dir(
        &dir,
        LoadOptions {
            scheme: QuantScheme::q4_mlx_default(),
            max_seq: 256,
            ..LoadOptions::default()
        },
    )
    .map_err(|e| {
        diag::err_msg("serve", "31b load_from_hf_dir", &e);
        e.to_string()
    })?;
    diag::log(
        "serve",
        format_args!(
            "31b loaded matrices={} hot={} in {:.1}s",
            banks.matrices.len(),
            diag::fmt_bytes(banks.total_hot_bytes() as u64),
            load_t0.elapsed().as_secs_f64()
        ),
    );
    // Host banks dropped inside from_host_banks before session KV/scratch alloc.
    let model = GpuSynthModel::from_host_banks(banks).map_err(|e| {
        diag::err_msg("serve", "31b Hot upload", &e);
        e.to_string()
    })?;
    GpuDecodeSession::new(model).map_err(|e| {
        diag::err_msg("serve", "31b session", &e);
        e.to_string()
    })
}

fn handle_client(mut stream: TcpStream, state: &Mutex<AppState>) -> std::io::Result<()> {
    let mut buf = [0u8; 8192];
    let n = stream.read(&mut buf)?;
    let req = String::from_utf8_lossy(&buf[..n]);
    let first = req.lines().next().unwrap_or("");
    let mut st = state.lock().unwrap();

    if first.starts_with("GET /health") {
        let body = format!(
            "{{\"ok\":true,\"version\":\"{}\",\"preset\":\"{}\",\"model\":\"{}\",\
             \"hot\":\"{}\"}}",
            gemma_metal::version(),
            st.preset,
            st.model_id,
            st.hot_note.replace('"', "'")
        );
        write_json(&mut stream, 200, &body)?;
    } else if first.starts_with("GET /v1/models") {
        write_json(
            &mut stream,
            200,
            &format!(
                "{{\"object\":\"list\",\"data\":[\
             {{\"id\":\"{}\",\"object\":\"model\",\"owned_by\":\"gemma-metal\"}},\
             {{\"id\":\"gemma-metal-31b\",\"object\":\"model\",\"owned_by\":\"gemma-metal\"}}\
             ]}}",
                st.model_id
            ),
        )?;
    } else if first.starts_with("POST /v1/chat/completions") {
        let prompt = [1u32, 2, 3, 4];
        let toks = match &mut st.backend {
            Backend::Synthetic(m) => {
                greedy_decode_host(m, &prompt, 8).unwrap_or_else(|_| prompt.to_vec())
            }
            Backend::Hot(sess) => sess
                .generate(&prompt, 8)
                .unwrap_or_else(|_| prompt.to_vec()),
        };
        let content = format!("tokens={toks:?}");
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let body = format!(
            "{{\"id\":\"chatcmpl-gemma\",\"object\":\"chat.completion\",\"created\":{created},\
             \"model\":\"{}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\
             \"content\":{}}},\"finish_reason\":\"stop\"}}]}}",
            st.model_id,
            serde_json::to_string(&content).unwrap_or_else(|_| "\"ok\"".into())
        );
        write_json(&mut stream, 200, &body)?;
        let _ = &st.text;
    } else {
        write_json(&mut stream, 404, "{\"error\":\"not found\"}")?;
    }
    Ok(())
}

fn write_json(stream: &mut TcpStream, code: u16, body: &str) -> std::io::Result<()> {
    let status = match code {
        200 => "200 OK",
        404 => "404 Not Found",
        _ => "500 Internal Server Error",
    };
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(resp.as_bytes())
}
