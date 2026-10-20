"""Group the top CPU processes by working directory, to name the load sources for the owner."""
import subprocess, collections
ps = subprocess.run(["ps", "-Ao", "pid=,%cpu=,comm="], capture_output=True, text=True).stdout.split("\n")
rows = []
for line in ps:
    parts = line.split(None, 2)
    if len(parts) == 3:
        try:
            rows.append((int(parts[0]), float(parts[1]), parts[2]))
        except ValueError:
            pass
rows.sort(key=lambda r: -r[1])
groups = collections.defaultdict(lambda: [0.0, 0, set()])
for pid, cpu, comm in rows[:60]:
    if cpu < 5:
        continue
    out = subprocess.run(["lsof", "-a", "-d", "cwd", "-p", str(pid), "-Fn"], capture_output=True, text=True).stdout
    cwd = next((l[1:] for l in out.splitlines() if l.startswith("n")), "?")
    key = "/".join(cwd.split("/")[:7])
    g = groups[key]
    g[0] += cpu
    g[1] += 1
    g[2].add(comm.rsplit("/", 1)[-1][:20])
for key, (cpu, n, names) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:10]:
    print(f"{cpu:7.0f}% {n:3d} procs  {key}  [{', '.join(sorted(names))[:70]}]")
