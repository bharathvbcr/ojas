#!/usr/bin/env python3
"""OpenAI-compatible server for the fastest local Gemma-4-31B config on this M5 Pro.

MLX 0.32 + DFlash spec-decode (block=5) + 4-bit draft ≈ 34-37 tok/s decode
(vs ~12.7 plain). Exact verify → outputs identical to greedy 31B.

Multi-turn: reuses target/draft prompt caches via common-prefix trim so turn-2+
TTFT skips the cached prefix (often −50–90%).

Run:
    ~/.venvs/dflash32/bin/python serve_dflash.py --port 8788

Use (OpenAI-compatible):
    curl -N http://localhost:8788/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true,"max_tokens":256}'

Stdlib-only HTTP (single-user local serving); model loads once at startup.
"""
import argparse, json, time, threading, queue
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.nn as nn
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from dflash.model_mlx import load_draft, stream_generate

TARGET = "mlx-community/gemma-4-31b-it-4bit"
DRAFT = "z-lab/gemma-4-31B-it-DFlash"
BLOCK_SIZE = 5

MODEL = TOK = DRAFTM = None
GEN_LOCK = threading.Lock()  # one generation at a time (single GPU)

# Sticky single-conversation prompt-cache (local single-user server).
_CONV = {
    "tokens": [],          # full prompt+completion token ids last turned
    "prompt_cache": None,  # target KV
    "draft_cache": None,
}


def load_all():
    global MODEL, TOK, DRAFTM
    t0 = time.perf_counter()
    MODEL, TOK = mlx_load(TARGET)
    DRAFTM = load_draft(DRAFT)
    nn.quantize(DRAFTM, group_size=64, bits=4,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
    DRAFTM.bind(MODEL)
    print(f"[serve] loaded target+draft in {time.perf_counter()-t0:.1f}s "
          f"(block={BLOCK_SIZE}, q4 draft, mlx0.32, prompt-cache ON)", flush=True)


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _prepare_cache(prompt_ids):
    """Trim sticky caches to the shared prefix; return (cache, draft, prior, rest)."""
    prior = _CONV["tokens"]
    cache = _CONV["prompt_cache"]
    draft = _CONV["draft_cache"]
    prefix = _common_prefix_len(prior, prompt_ids) if cache is not None else 0

    if cache is None:
        cache = make_prompt_cache(MODEL)
        draft = make_prompt_cache(DRAFTM)
        prior_use, rest = [], prompt_ids
        return cache, draft, prior_use, rest

    cached_n = int(getattr(cache[0], "offset", 0) or 0)
    # Shrink sticky cache if common prefix is shorter than cached tokens.
    if cached_n > prefix:
        trim_prompt_cache(cache, cached_n - prefix)
        d_off = int(getattr(draft[0], "offset", 0) or 0) if draft else 0
        if draft is not None and d_off > prefix:
            trim_prompt_cache(draft, d_off - prefix)
        cached_n = prefix

    if cached_n < prefix:
        # Cache drifted behind the token list — rebuild.
        cache = make_prompt_cache(MODEL)
        draft = make_prompt_cache(DRAFTM)
        prior_use, rest = [], prompt_ids
        return cache, draft, prior_use, rest

    prior_use = prompt_ids[:cached_n]
    rest = prompt_ids[cached_n:]
    if not rest:
        # Degenerate: nothing new — force one-token re-feed of last id so logits refresh.
        if prior_use:
            rest = [prior_use[-1]]
            prior_use = prior_use[:-1]
            if cached_n > 0:
                trim_prompt_cache(cache, 1)
                if draft is not None and int(getattr(draft[0], "offset", 0) or 0) > 0:
                    trim_prompt_cache(draft, 1)
        else:
            rest = prompt_ids
            prior_use = []
    return cache, draft, prior_use, rest


def _run_generate(prompt_text, max_tokens, on_response=None):
    """Shared generation path. `on_response(r)` called per chunk (may be none)."""
    prompt_ids = TOK.encode(prompt_text, add_special_tokens=False)
    # Chat templates already include BOS/specials when apply_chat_template was used.
    if not isinstance(prompt_ids, list):
        prompt_ids = list(prompt_ids)

    cache, draft, prior, rest = _prepare_cache(prompt_ids)
    session_out = {}
    cached_n = len(prior)
    t_prefill0 = time.perf_counter()
    ttft_ms = None
    last = None
    for r in stream_generate(
        MODEL, DRAFTM, TOK, rest,
        block_size=BLOCK_SIZE, max_tokens=max_tokens, temperature=0.0,
        prompt_cache=cache, draft_cache=draft, prior_tokens=prior,
        session_out=session_out,
    ):
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_prefill0) * 1e3
        last = r
        if on_response is not None:
            on_response(r)
    if ttft_ms is None:
        ttft_ms = (time.perf_counter() - t_prefill0) * 1e3
    prefill_n = session_out.get("prefill_tokens", len(rest))

    _CONV["tokens"] = session_out.get("tokens", prompt_ids)
    _CONV["prompt_cache"] = session_out.get("prompt_cache", cache)
    _CONV["draft_cache"] = session_out.get("draft_cache", draft)
    return last, cached_n, prefill_n, ttft_ms


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[serve] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "gemma-4-31b-dflash", "object": "model", "owned_by": "local"}]})
        elif self.path == "/v1/reset_cache":
            _CONV["tokens"] = []
            _CONV["prompt_cache"] = None
            _CONV["draft_cache"] = None
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        except Exception as e:
            return self._json(400, {"error": f"bad request: {e}"})
        messages = req.get("messages", [])
        max_tokens = int(req.get("max_tokens", 512))
        stream = bool(req.get("stream", False))
        rid = f"chatcmpl-{int(time.time()*1000)}"
        # Optional per-request cache reset (new conversation).
        if req.get("reset_cache") or req.get("cache_reset"):
            _CONV["tokens"] = []
            _CONV["prompt_cache"] = None
            _CONV["draft_cache"] = None

        prompt = TOK.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        with GEN_LOCK:
            t0 = time.perf_counter()
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()

                # M2: writer thread — detok/SSE behind the next draft dispatch.
                write_q = queue.Queue()
                write_err = []

                def writer():
                    try:
                        while True:
                            item = write_q.get()
                            if item is None:
                                break
                            data = f"data: {json.dumps(item)}\n\n".encode()
                            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                            self.wfile.flush()
                    except Exception as e:
                        write_err.append(e)

                wt = threading.Thread(target=writer, daemon=True)
                wt.start()

                def on_response(r):
                    write_q.put({
                        "id": rid, "object": "chat.completion.chunk",
                        "model": "gemma-4-31b-dflash",
                        "choices": [{"index": 0, "delta": {"content": r.text},
                                     "finish_reason": None}],
                    })

                last, cached_n, prefill_n, ttft_ms = _run_generate(
                    prompt, max_tokens, on_response=on_response)
                tps = last.generation_tps if last else 0.0
                write_q.put({
                    "id": rid, "object": "chat.completion.chunk",
                    "model": "gemma-4-31b-dflash",
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": (last.finish_reason or "stop") if last else "stop"}],
                    "usage": {
                        "completion_tokens": last.generation_tokens if last else 0,
                        "decode_tok_s": round(tps, 1),
                        "cached_tokens": cached_n,
                        "prefill_tokens": prefill_n,
                        "ttft_ms": round(ttft_ms, 1),
                    },
                })
                write_q.put(None)
                wt.join(timeout=30)
                done = b"data: [DONE]\n\n"
                self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n")
                print(f"[serve] {rid}: {last.generation_tokens if last else 0} toks "
                      f"@ {tps:.1f} tok/s TTFT={ttft_ms:.0f}ms "
                      f"cached={cached_n} prefill={prefill_n} "
                      f"in {time.perf_counter()-t0:.1f}s", flush=True)
            else:
                text = []
                def on_response(r):
                    text.append(r.text)
                last, cached_n, prefill_n, ttft_ms = _run_generate(
                    prompt, max_tokens, on_response=on_response)
                tps = last.generation_tps if last else 0.0
                print(f"[serve] {rid}: {last.generation_tokens if last else 0} toks "
                      f"@ {tps:.1f} tok/s TTFT={ttft_ms:.0f}ms "
                      f"cached={cached_n} prefill={prefill_n} "
                      f"in {time.perf_counter()-t0:.1f}s", flush=True)
                self._json(200, {"id": rid, "object": "chat.completion",
                                 "model": "gemma-4-31b-dflash",
                                 "choices": [{"index": 0,
                                              "message": {"role": "assistant", "content": "".join(text)},
                                              "finish_reason": (last.finish_reason or "stop") if last else "stop"}],
                                 "usage": {
                                     "completion_tokens": last.generation_tokens if last else 0,
                                     "decode_tok_s": round(tps, 1),
                                     "cached_tokens": cached_n,
                                     "prefill_tokens": prefill_n,
                                     "ttft_ms": round(ttft_ms, 1),
                                 }})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args()
    load_all()
    srv = HTTPServer(("127.0.0.1", args.port), Handler)  # single-thread: mlx GPU streams are thread-local
    print(f"[serve] listening on http://127.0.0.1:{args.port}/v1/chat/completions", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
