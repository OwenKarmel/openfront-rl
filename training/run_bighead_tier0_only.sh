#!/bin/bash
# Launches (or resumes) RUN7: the TIER-0-ONLY CONTROL for run6, config pinned.
# GTX 1660 box (cuda:0, --eval-envs 8). Prepared 2026-11-28.
#
# Identical to run_bighead_attacksize.sh (run6) except that it has NO
# --attack-size-head: every attack sends troops/5, as in run4. Same
# --init-from (run4's update-5310 checkpoint, hard/small), same Tier-0 fixes
# (--kl-stop epoch --value-norm --adv-norm minibatch, canary evals,
# --eval-report-exclude-maps labyrinth), same memory flags (--recycle-every 3
# --canary-envs 4). Keep every other flag in step with run6's launcher; see
# its header for the reason behind each one.
#
# WHY: run6 differs from run4 in five ways at once. By its update-5400 suite
# it peaked lower than run4's final checkpoint on 62 of the same 64 hard
# seeds (mean peak territory 0.074 vs 0.196) with 0/128 suite wins, and its
# KL stop still ended updates after ~3.5 of 8 epochs. This run separates the
# four Tier-0 fixes from the attack-size head: compare its suites at updates
# 5350 / 5400 / 5450 one-for-one with run6's (same seeds), suites with
# suites. If it also falls below run4, the drop comes from the Tier-0 fixes
# (most likely the value/advantage normalization or the epoch KL stop), not
# the head.
#
# PLAN: about 140 updates (5310 -> 5450, ~14-15 h), run while run6 is paused
# at a checkpoint -- the two cannot share this box (6 GB GPU, ~10 GB RAM
# each). Then decide which one to continue.
#
# Safe to relaunch: with latest.pt present, train.py resumes it and ignores
# --init-from. RUN_NAME overrides the run directory -- for smoke tests only.

set -euo pipefail
cd "$(dirname "$0")"

export OPENFRONT_EXPAND_MASK=1
export OPENFRONT_BOAT_MASK=1
export OPENFRONT_NEUTRAL_BOATS=1

RUN=${RUN_NAME:-run7_bighead_tier0_only}
SEED_DIR=checkpoints/run4_bighead_hard_terminal
mkdir -p "checkpoints/$RUN" "replays/$RUN"
if [ ! -e "checkpoints/$RUN/latest.pt" ] && [ ! -e "checkpoints/$RUN/notified_milestones.json" ]; then
  # So milestones run4 already announced do not fire again.
  cp "$SEED_DIR/notified_milestones.json" "checkpoints/$RUN/notified_milestones.json"
fi

exec "${OPENFRONT_PYTHON:-/home/developer/miniconda/envs/openfront-rl/bin/python}" -u train.py \
  --num-envs 10 \
  --reward-mode shaping_and_terminal \
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
  --curriculum connected-first \
  --eval-episodes 64 \
  --promote-threshold 0.75 \
  --tile-head big \
  --init-from "$SEED_DIR/latest.pt" \
  --kl-stop epoch \
  --value-norm \
  --adv-norm minibatch \
  --eval-every-steps 50000 \
  --canary-every-steps 10000 \
  --canary-episodes 16 \
  --canary-envs 4 \
  --recycle-every 3 \
  --eval-report-exclude-maps labyrinth \
  --checkpoint-dir "checkpoints/$RUN" \
  --log-csv "checkpoints/$RUN/train_log.csv" \
  --eval-log-csv "checkpoints/$RUN/eval_log.csv" \
  --eval-dir "replays/$RUN" \
  --device cuda:0 \
  --eval-envs 8 \
  "$@"
