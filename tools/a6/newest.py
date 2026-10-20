"""Newest non-build file per a6 lane worktree, and every file modified after a cutoff."""
import os, sys, time
R = "/Users/bharath/Code/research"
lanes = sys.argv[1].split(",")
cutoff = time.mktime(time.strptime(sys.argv[2], "%Y-%m-%d %H:%M"))
for lane in lanes:
    root = f"{R}/ojas-a6-{lane}"
    newest, after = (0, ""), []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("target", ".git", "__pycache__")]
        for f in files:
            if f == ".DS_Store":
                continue
            p = os.path.join(d, f)
            try:
                m = os.lstat(p).st_mtime
            except OSError:
                continue
            if m > newest[0]:
                newest = (m, p)
            if m > cutoff:
                after.append(p[len(root) + 1:])
    print(lane, time.strftime("%H:%M:%S", time.localtime(newest[0])), newest[1][len(root) + 1:], "| after cutoff:", after[:8])
