import json, sys
from collections import Counter
d = json.load(open(sys.argv[1]))
ts = d.get("tasks", d) if isinstance(d, dict) else d
if isinstance(d, dict):
    print("keys:", list(d.keys()))
print(len(ts), Counter(t.get("status") for t in ts))
for t in ts:
    if t.get("status") != "done":
        print(t.get("status"), t.get("priority"), t.get("severity"), t.get("item_id", t.get("id", ""))[:11], (t.get("title") or "")[:120])
