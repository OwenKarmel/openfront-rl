#!/bin/bash
# FINISHED on the RTX 5000 Ada workstation 2026-10-10 at update 1437 (stopped
# by hand at hard/small); archived to
# checkpoints/archive/run_2026-10-09_connected-first_big-tile-head/ and
# committed under analysis/run_2026-10-09_connected-first_big-tile-head/.
# Left ENABLED on purpose: it is the run to replicate on the GTX 1660 box --
# see REPLICATE.md in that analysis folder (pass --device cuda:0 there).
#
# Launches (or resumes) the BIG-TILE-HEAD arm of the connected-first run,
# config pinned. Runs in parallel with run_connected_first.sh, so it has its
# own checkpoint, log and replay directories (and the second GPU) -- never
# point it at the regular checkpoints/.
#
# Identical to run_connected_first.sh EXCEPT --tile-head big, so a comparison
# against it isolates that one change (NOT run_connected_first_paramnorm.sh's
# --param-entropy-norm, which is off here). Same eval seeds: eval suite k of
# each run plays the same 64 games.
#
# Hypothesis (2026-10-09): boat_attack's target head never learns to aim.
# On the connected-first run it sat at 96-97% of its maximum entropy at update
# ~670, and normalizing its entropy bonus (the paramnorm arm) left it at
# 95-96% -- so the bonus was not what held it flat. The default head is a
# single 1x1 conv: each cell's logit sees only that cell's local features,
# nothing about the rest of the map or the game state, while choosing a
# landing needs both. BigTileHead (network.py) adds the legal mask and x/y
# coordinates per cell, the trunk's game state as a FiLM modulation, and
# dilated residual convs whose receptive field spans the whole 32x32 grid.
# Supervised on "the legal cell nearest AGENT's territory" it reaches ~98-99%
# accuracy vs ~64% for the default head (tests/test_big_tile_head.py).
# Caveat: the default head CAN learn that task under supervision, so a weak,
# delayed RL signal may still be the main obstacle; this arm tests whether
# capacity and context are enough to get boats aimed under RL.
# Prediction: tile-head entropy falls well below its maximum, boat_attack
# launches more often per attempt, and the island maps' win rate improves.

set -euo pipefail
cd "$(dirname "$0")"

export OPENFRONT_EXPAND_MASK=1
export OPENFRONT_BOAT_MASK=1
export OPENFRONT_NEUTRAL_BOATS=1

RUN=run3_big_tile_head
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
  --tile-head big \
  --checkpoint-dir "checkpoints/$RUN" \
  --log-csv "checkpoints/$RUN/train_log.csv" \
  --eval-log-csv "checkpoints/$RUN/eval_log.csv" \
  --eval-dir "replays/$RUN" \
  --device cuda:1 \
  "$@"
