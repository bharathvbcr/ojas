#!/usr/bin/env bash
# Tests for stream_check.sh, on the recorded streams. No GPU or weights needed:
#   bash gemma-metal/bench/test_stream_check.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=stream_check.sh
. "$here/stream_check.sh"

fail=0
expect() { # expect <degenerate|healthy> <label> <stream>
  if stream_degenerate "$3" >/dev/null; then got=degenerate; else got=healthy; fi
  if [ "$got" = "$1" ]; then
    echo "ok   $2: $got"
  else
    echo "FAIL $2: want $1, got $got"
    fail=1
  fi
}

# fusion_ab_31b_20260719T080223Z.json baseline (both arms recorded this and passed).
expect degenerate "31b mode-locked baseline" \
  "[759, 759, 759, 537, 759, 537, 537, 537, 537, 537, 537, 537, 537, 537, 240017, 240017]"
# fusion_ab_e4b_20260720T012511Z.json baseline.
expect healthy "e4b baseline" \
  "[133533, 91962, 3301, 217288, 236782, 236743, 196059, 805, 217288, 236881, 59425, 240296, 28698, 178662, 67991, 17682]"
expect degenerate "all one id" "[5, 5, 5, 5, 5, 5, 5, 5]"
expect degenerate "one id over half, many distinct" "[1, 9, 9, 9, 9, 9, 2, 3]"
expect healthy "exactly half distinct, no id over half" "[1, 1, 2, 2, 3, 3, 4, 4]"
expect degenerate "empty" "[]"
expect degenerate "missing" ""
expect degenerate "garbage" "NA"

exit "$fail"
