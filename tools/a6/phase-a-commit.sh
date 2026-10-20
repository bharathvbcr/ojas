#!/bin/bash
# Commit in the phase-A worktree only, and only on its branch (owner authorised ojas-a6 to run phase A,
# 2026-10-18). Usage: phase-a-commit.sh <subject> <body>
# Stages everything in ../ojas-phase-a except build output, shows the stat, commits with the trailer.
set -euo pipefail
WT=/Users/bharath/Code/research/ojas-phase-a
BRANCH=gp/a6-phase-a-import
[ "$(git -C "$WT" rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { echo "REFUSED: $WT is not on $BRANCH" >&2; exit 1; }
[ $# -eq 2 ] || { echo "usage: $0 <subject> <body>" >&2; exit 2; }
# target/ is gitignored; naming it in a pathspec makes `git add` fail under set -e.
git -C "$WT" add -A
if git -C "$WT" diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
git -C "$WT" diff --cached --stat | tail -30
git -C "$WT" commit -q -m "$1" -m "$2" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git -C "$WT" log --oneline -1
