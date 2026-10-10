#!/bin/bash
# Launches (or resumes) the connected-first run, config pinned.
#
# Checkpoints store weights, optimizer state and curriculum position but NOT
# hyperparameters, so a resume that forgets a flag silently continues under
# argparse defaults. Always resume via this script, never by retyping the
# command. The env-var gated behaviours below are part of the config too:
# train.py prints them at startup, and every env reset fails if the Node
# worker is not actually running with them.
#
# START IT COLD. It reuses checkpoints/ like every launcher, so archive the
# previous run first (see CLAUDE.md). If checkpoints/latest.pt is still the
# previous run's, train.py refuses to resume it -- that checkpoint has no
# curriculum ladder, i.e. "difficulty", not "connected-first".
#
# Hyperparameters are the beta=2 shaping-only gamma anneal's
# (run_gamma_anneal_shaping_only_beta2.sh), unchanged. What changes is
# everything around them, and it changes AT ONCE, so this is not a
# single-variable comparison against that run:
#
#   OPENFRONT_EXPAND_MASK=1, OPENFRONT_BOAT_MASK=1
#       "expand" is only legal when a neutral tile borders AGENT, and
#       "boat_attack" only when AGENT owns shore. Without them both actions
#       can be exact no-ops that earn exactly what "noop" earns, so there is
#       no gradient between them -- eval 0ab47d4b issued 9,159 no-op expands
#       over 91,591 ticks. First training run to use either mask.
#   OPENFRONT_NEUTRAL_BOATS=1
#       Boats can land on unowned land, as the Nation AI's and a human's can.
#       Before this AGENT's reachable world was its spawn landmass plus
#       opponent territory -- ~36% of the land on an average SMALL_MAPS map
#       (env-bridge/src/mapLandmasses.ts).
#   --curriculum connected-first
#       easy on CONNECTED_MAPS -> easy on SMALL_MAPS -> medium -> hard ->
#       impossible, so the land game is learned on maps where it can win
#       before the island maps demand a boat.
#   Eval suites (train.py's defaults)
#       Every eval plays 64 fixed seeds with the SAMPLED policy and promotes
#       at a 0.75 win rate -- instead of one greedy game feeding a 9/12
#       window, which never opened in any run (greedy eval won 3/198 in the
#       2026-09-13 anneal while the sampled policy won ~24%).
#
# Cost, estimated rather than measured for this config:
#   * Eval wall time. A competent policy's games against easy average ~80k
#     ticks (~8,000 decision steps; measured with a scripted policy), so a
#     64-episode suite at 16 workers is ~4 rounds of ~8,000 steps -- several
#     minutes per eval, and at --eval-every-steps 3000 late in the anneal
#     (3 updates of ~156s between evals) evals could roughly double wall
#     time. Raise --eval-envs on a box with the cores (it does not change
#     training), or --eval-every-steps if it is still too slow (it changes
#     eval-index comparability with earlier runs).
#   * Memory: as the beta=2 run's launcher, plus one Node worker per
#     --eval-envs during an eval (~100-160MB each). train.py frees the
#     rollout buffer before evaluating.

set -euo pipefail
cd "$(dirname "$0")"

# ARCHIVED 2026-10-10 at update 1650. This run now lives in
# checkpoints/archive/run_2026-10-09_connected-first/ (logs, summary and final checkpoint also
# committed under analysis/run_2026-10-09_connected-first/). Kept for provenance; running the command
# below as-is would start a NEW run in checkpoints/ under this config. To genuinely
# resume this run, copy the archive folder out to its own directory and pass
# --checkpoint-dir/--log-csv/--eval-log-csv/--eval-dir pointing at it.
if [ "${I_KNOW_THIS_RUN_IS_ARCHIVED:-}" != "1" ]; then
  echo "run_connected_first.sh: this run is archived; see the comment in this script." >&2
  echo "To replicate the big-tile-head run, see analysis/run_2026-10-09_connected-first_big-tile-head/REPLICATE.md." >&2
  exit 1
fi

export OPENFRONT_EXPAND_MASK=1
export OPENFRONT_BOAT_MASK=1
export OPENFRONT_NEUTRAL_BOATS=1

exec "${OPENFRONT_PYTHON:-/home/developer/miniconda/envs/openfront-rl/bin/python}" -u train.py \
  --num-envs 10 \
  --reward-mode shaping_only \
  --potential-beta 2 \
  --gamma-start 0.98 \
  --gamma 0.999 \
  --gamma-anneal-updates 3000 \
  --rollout-match-horizon \
  --epochs 8 \
  --weight-decay 0.1 \
  --minibatch-size 128 \
  --entropy-coef 0.02 \
  --checkpoint-every 5 \
  --eval-every-steps 3000 \
  --curriculum connected-first \
  --eval-episodes 64 \
  --promote-threshold 0.75 \
  "$@"
