# gemma-metal Phase 0 bench harness

Local baseline ladder for Gemma 4 on the M5 Pro host. Inspired by
[`/Users/bharath/Code/Benchmark`](/Users/bharath/Code/Benchmark), extended for
multi-runtime compare (Ollama, mlx-lm, LiteRT-LM, BaseRT) with pinned KV /
max tokens.

Living gates: [`../docs/gates.md`](../docs/gates.md).

## Quick start

```bash
cd gemma-metal/bench
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # optional; stdlib works for Ollama-only

# Probe which runtimes / models are present
python3 bench.py probe

# Ollama 31B (already pulled as gemma4:31b-mlx)
python3 bench.py ollama --model gemma4:31b-mlx --num-ctx 4096 --max-tokens 128

# mlx-lm (downloads HF repo on first use)
python3 bench.py mlx --model mlx-community/gemma-4-e4b-it-4bit \
  --prompt-tokens 128 --generation-tokens 128

# Write/refresh gates.md measurement section from latest JSON
python3 bench.py summarize
```

## Pins (honest lane)

| Knob | Default | Why |
|------|---------|-----|
| `num_ctx` / max KV | 4096 | Cap KV traffic; comparable across runs |
| `max_tokens` / generation | 128 | Stable decode tok/s (short gens inflate TTFT share) |
| temperature | 0 | Greedy |
| thinking | off (Ollama `think:false`) | Don't mix reasoning tokens into decode rate |

## Outputs

- `results/run_YYYYMMDD_HHMMSS.json` — raw metrics
- `results/latest.json` — symlink/copy of most recent run
