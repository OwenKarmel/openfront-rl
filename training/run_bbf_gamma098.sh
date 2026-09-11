#!/bin/bash
# Launches (or resumes) the BBF-style gamma=0.98 run with its config pinned.
#
# Exists because checkpoints store weights, optimizer state and curriculum
# position but NOT hyperparameters (see save_checkpoint in train.py). A
# resume that forgets a flag therefore does not fail -- it silently
# continues the same checkpoint under argparse defaults, flipping gamma
# 0.98 -> 0.99, rollout 50 -> 64 and epochs 8 -> 4 mid-run, and nothing in
# the logs says so. This project has already lost a branch to a quieter
# version of that mistake (see archive/run_2026-09-07_..._LOCAL-DIVERGENT).
# Always resume via this script, never by retyping the command.
#
# Config rationale:
#   --gamma 0.98 / --rollout-length 50  gamma's effective horizon is
#       1/(1-gamma) = 50 decision steps, and one decision step is 1.0s of
#       game time (ticks-per-step 10 x msPerTick 100). So the rollout is
#       sized to exactly one discount horizon -- collecting past the point
#       the return stops weighting buys nothing for credit assignment.
#   --epochs 8 / --weight-decay 0.1     BBF (Schwarzer et al. 2023) pairs a
#       high replay ratio with AdamW decay to offset the overfitting that
#       reusing each transition invites. 8 epochs is replay ratio 8, BBF's
#       own setting; 0.1 is BBF's weight decay verbatim.
#   --eval-every 60                     keeps experience-per-eval at 3000
#       steps, matching the previous run's 10 updates x 300 steps. At
#       rollout 50 an eval-every of 10 would run an eval (a full game to
#       completion, ~28 min of game time) every ~2.5 min of training.

set -euo pipefail
cd "$(dirname "$0")"

exec /home/developer/miniconda/envs/openfront-rl/bin/python -u train.py \
  --num-envs 10 \
  --rollout-length 50 \
  --gamma 0.98 \
  --epochs 8 \
  --weight-decay 0.1 \
  --minibatch-size 128 \
  --entropy-coef 0.02 \
  --checkpoint-every 20 \
  --eval-every 60 \
  "$@"
