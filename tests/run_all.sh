#!/bin/bash
# Runs every test in this folder, in ascending cost order, one at a time.
#
# SEQUENTIAL ON PURPOSE. The env-backed tests each spawn several Node
# env-bridge workers and a torch process; run in parallel they compete with a
# live training run for the RAM that gets it OOM-killed. Check `free -m`
# before starting one of these while training is running -- see CLAUDE.md.
#
# usage: tests/run_all.sh [-q]     (-q: only the per-file verdict)
# exit 0 only if every test passed.

set -uo pipefail
cd "$(dirname "$0")"
PY=/home/developer/miniconda/envs/openfront-rl/bin/python
QUIET=${1:-}

# Pure-function tests first (seconds, no env), then env-backed ones (minutes,
# each standing up real OpenFrontIO episodes).
TESTS=(
  test_schedules.py
  test_eval_cadence.py
  test_curriculum.py
  test_param_entropy_norm.py
  test_big_tile_head.py
  smoke_test.py
  smoke_test_multiopp.py
  test_neutral_boats.py
  test_eval_suite.py
  test_expand_mask.py
  test_boat_mask.py
  test_reward_modes.py
  test_potential_beta.py
)

fail=0
for t in "${TESTS[@]}"; do
  echo "=== $t ==="
  start=$SECONDS
  if [ "$QUIET" = "-q" ]; then
    out=$("$PY" -u "$t" 2>&1); rc=$?
    echo "$out" | tail -3
  else
    "$PY" -u "$t"; rc=$?
  fi
  el=$((SECONDS - start))
  if [ $rc -eq 0 ]; then echo "--- PASS $t (${el}s)"; else echo "--- FAIL $t (rc=$rc, ${el}s)"; fail=1; fi
done
[ $fail -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $fail
