#!/bin/bash
# Launches (or resumes) the PARAMETER-ENTROPY-NORM arm of the connected-first
# run, config pinned. Runs in parallel with run_connected_first.sh, so it has
# its own checkpoint, log and replay directories (and the second GPU) --
# never point it at the regular checkpoints/.
#
# Identical to run_connected_first.sh EXCEPT --param-entropy-norm, so a
# comparison against it isolates that one change. Same eval seeds too: eval
# suite k of each run plays the same 64 games, so the two eval_episodes.csv
# files compare game for game.
#
# Hypothesis (2026-10-09, from the connected-first run at update ~305):
#   The entropy bonus is a sum of the type head's entropy (<= ln 4 = 1.4)
#   and, when boat_attack is sampled, the tile head's (<= ln 1024 = 6.9), all
#   under one coefficient -- so it pulls hardest on keeping boat TARGETS
#   uniform. Measured on that run's checkpoint: the tile head sat at 88-93% of
#   its maximum entropy over 240-510 legal cells (most likely cell 1-3%), the
#   policy sampled boat_attack on up to 22% of steps, and its eval win rate
#   collapsed in both stretches where boat_attack probability rose (eval
#   at update 114: 4/64, after 166's 50/64; 263: 8/64 after 216's 36/64).
#   --param-entropy-norm divides each parameter head's entropy by
#   log(#legal choices), so its bonus is a fraction of its own maximum.
#   Prediction: tile-head entropy falls well below its maximum, boats aim, and
#   the eval win rate stops collapsing whenever boat_attack becomes common.
#
# Note: train_log.csv's `entropy` column here is the NORMALIZED bonus entropy,
# not comparable to run 1's; compare type_entropy instead.

set -euo pipefail
cd "$(dirname "$0")"

export OPENFRONT_EXPAND_MASK=1
export OPENFRONT_BOAT_MASK=1
export OPENFRONT_NEUTRAL_BOATS=1

RUN=run2_param_entropy_norm
mkdir -p "checkpoints/$RUN" "replays/$RUN"

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
  --param-entropy-norm \
  --checkpoint-dir "checkpoints/$RUN" \
  --log-csv "checkpoints/$RUN/train_log.csv" \
  --eval-log-csv "checkpoints/$RUN/eval_log.csv" \
  --eval-dir "replays/$RUN" \
  --device cuda:1 \
  "$@"
