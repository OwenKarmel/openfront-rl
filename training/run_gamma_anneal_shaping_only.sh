#!/bin/bash
# Launches (or resumes) the shaping-only gamma-anneal run with its config pinned.
#
# Checkpoints store weights, optimizer state and curriculum position but NOT
# hyperparameters, so a resume that forgets a flag silently continues under
# argparse defaults -- here that would quietly restore the +/-1 terminal
# reward, drop the gamma schedule and snap the rollout back to 64, with
# nothing in the logs to record it. Always resume via this script, never by
# retyping the command.
#
# Identical to the archived gamma-anneal run
# (checkpoints/archive/run_2026-09-13_gamma-anneal098-0999_rollout-horizon/,
# launcher run_gamma_anneal.sh) EXCEPT --reward-mode shaping_only, so a
# comparison against that run isolates the reward change. Known issues found
# in that run are deliberately NOT fixed here, for the same reason: the
# leftover-minibatch KL early stops, and the entropy bonus overpowering
# batch-normalized advantages at spawn as gamma rises.
#
# Config rationale:
#   --reward-mode shaping_only
#       Tile-share delta only, no +/-1 when the game ends. The undiscounted
#       return telescopes to final share minus starting share, which still
#       ranks a win (the game ends at 80% domination) above any loss (an
#       opponent at 80% leaves AGENT at most 20%). A loss is never negative,
#       though: holding the spawn until losing earns exactly 0.
#   --gamma-start 0.98 --gamma 0.999 --gamma-anneal-updates 3000
#       Horizon 50 -> 1000 decision steps (0.8 -> 16.7 min of game time),
#       geometric in (1-gamma) -- see gamma_at() in train.py.
#   --rollout-match-horizon
#       Rollout sized to exactly one discount horizon every update: 50 -> 1000.
#   --eval-every-steps 3000
#       Eval index keeps meaning 3000 decision steps per env while the rollout
#       varies, comparable with every run in analysis/.
#   --checkpoint-every 5
#       Updates reach ~160s around T=640 and more past that.
#
# Cost, stated up front (measured in the previous anneal run):
#   * Memory: a T=1000 update peaked at 8.3GB host RAM for the training tree
#     with fresh env workers and 5.6/6.1GB VRAM (ppo_update moves the whole
#     rollout's tile_grid onto the GPU). By T=625 the workers had grown and
#     the tree was already 8.3GB, projecting ~10.3GB at T=1000 (reached around
#     update 3000). Does not fit beside a second run; stop the OpenFrontIO dev
#     server before then if headroom is short.
#   * Time: ~951k decision steps per env over the anneal.

set -euo pipefail
cd "$(dirname "$0")"

# ARCHIVED 2026-09-17 at update 2885. This run now lives in
# checkpoints/archive/run_2026-09-17_gamma-anneal098-0999_shaping-only/.
# The next run reuses checkpoints/, and running the command below as-is would
# resume THAT run's latest.pt under this run's reward mode and gamma schedule
# -- silently, since checkpoints do not store hyperparameters. Kept for
# provenance. To genuinely resume this run, copy the archive folder out to its
# own directory and pass --checkpoint-dir/--log-csv/--eval-dir pointing at it.
# The successor is run_gamma_anneal_shaping_only_beta2.sh.
if [ "${I_KNOW_THIS_RUN_IS_ARCHIVED:-}" != "1" ]; then
  echo "run_gamma_anneal_shaping_only.sh: this run is archived; see the comment in this script." >&2
  echo "The live run is run_gamma_anneal_shaping_only_beta2.sh." >&2
  exit 1
fi

exec /home/developer/miniconda/envs/openfront-rl/bin/python -u train.py \
  --num-envs 10 \
  --reward-mode shaping_only \
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
  "$@"
