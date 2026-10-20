import json, sys
D = "/Users/bharath/Code/research/ojas/backups/a6-tools"
for tid in sys.argv[1:]:
    d = json.load(open(f"{D}/card-{tid}.json"))
    t = d.get("task", d)
    print("=====", t.get("item_id"), t.get("status"), "rev", t.get("revision"))
    for k, v in t.items():
        if k in ("item_id", "status", "revision") or v in (None, "", [], {}):
            continue
        s = v if isinstance(v, str) else json.dumps(v, indent=1)
        print(f"-- {k}:")
        print(s[:8000])
