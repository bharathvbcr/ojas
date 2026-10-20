#!/bin/bash
# usage: gp.sh <tool> '<json args>' [sleep-seconds]
# Drives gitpulse-mcp over stdio; prints result.content[0].text of the tools/call.
T=$1; A=$2; W=${3:-20}
# '@file' reads the JSON arguments from a file
case "$A" in @*) A=$(tr -d '\n' < "${A#@}");; esac
{ printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"ojas-a6","version":"1"}}}'
  printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/initialized"}'
  printf '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"%s","arguments":%s}}\n' "$T" "$A"
  sleep "$W"; } | /Users/bharath/.cargo/bin/gitpulse-mcp 2>/dev/null | python3 -c '
import sys,json
for l in sys.stdin:
    try: m=json.loads(l)
    except Exception: continue
    if m.get("id")==2:
        r=m.get("result") or m
        try: print(r["content"][0]["text"])
        except Exception: print(json.dumps(r))
'
