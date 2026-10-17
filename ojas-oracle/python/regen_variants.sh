#!/usr/bin/env bash
# Regenerate the GQA and sliding-window goldens (fixtures/variants) and the
# sampler golden (fixtures/sampler), or check them.
#
#   bash ojas-oracle/python/regen_variants.sh          # rewrite the fixtures
#   bash ojas-oracle/python/regen_variants.sh check    # write nothing: generate
#       twice into scratch directories and fail unless both runs are
#       byte-identical to each other and to the checked-in files
#
# PYTHON overrides the interpreter (default: the torch 2.13 / Python 3.14 one
# bench/ uses). CPU only. The tiny fixtures (fixtures/tiny, regen_tiny.sh) are
# not touched.
set -euo pipefail

ORACLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/opt/homebrew/opt/python@3.14/bin/python3.14}"
export PYTHONDONTWRITEBYTECODE=1

generate() {
    local root="$1"
    "$PY" "$ORACLE/python/variant_fixture.py" --dir "$root/variants"
    "$PY" "$ORACLE/python/sampler_golden.py" --out "$root/sampler/golden.txt"
}

case "${1:-write}" in
write)
    generate "$ORACLE/fixtures"
    ;;
check)
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/ojas-oracle-variants.XXXXXX")"
    trap 'rm -rf "$WORK"' EXIT
    generate "$WORK/run1"
    generate "$WORK/run2"
    diff -r "$WORK/run1" "$WORK/run2" || { echo "two runs differ" >&2; exit 1; }
    diff -r "$WORK/run1/variants" "$ORACLE/fixtures/variants" \
        || { echo "fixtures/variants is not what the generator writes" >&2; exit 1; }
    diff -r "$WORK/run1/sampler" "$ORACLE/fixtures/sampler" \
        || { echo "fixtures/sampler is not what the generator writes" >&2; exit 1; }
    echo "byte-identical across 2 runs and to the checked-in files"
    ;;
*)
    echo "usage: regen_variants.sh [write|check]" >&2
    exit 2
    ;;
esac
