#!/bin/bash
# Launches (or resumes) the gamma-anneal run with its config pinned.
#
# Same reason run_bbf_gamma098.sh exists, and more so here: checkpoints store
# weights, optimizer state and curriculum position but NOT hyperparameters. A
# resume that forgets --gamma-start silently drops the schedule and trains at
# a constant --gamma from wherever the update counter is; one that forgets
# --rollout-match-horizon snaps the rollout back to 64. Always resume via this
# script, never by retyping the command.
#
# Everything is kept identical to run_bbf_gamma098.sh EXCEPT the discount
# schedule and what follows from it, so this run isolates the anneal. The
# expand/boat action masks stay at their default (off), same as that run --
# they are being tested separately on Kaggle, and stacking them here would
# confound both comparisons.
#
# Config rationale:
#   --gamma-start 0.98 --gamma 0.999 --gamma-anneal-updates 3000
#       Horizon 50 -> 1000 decision steps (0.8 -> 16.7 min of game time).
#       At 0.98 a terminal reward is weighted 0.98^1172 ~ 5e-11 at the start
#       of a median-length eval episode -- numerically absent, which is how an
#       infinitely repeatable free no-op cost the agent nothing visible. At
#       0.999 it is ~0.31. Short horizons early keep the value function easy
#       to fit; the long one is only worth bootstrapping from once V is.
#       The anneal is geometric in (1-gamma) -- see gamma_at() in train.py.
#   --rollout-match-horizon
#       Keeps the rollout sized to exactly one discount horizon, as in the
#       gamma=0.98 run (50 steps there), recomputed every update: 50 -> 1000.
#   --eval-every-steps 3000
#       The rollout varies, so update-keyed eval would drift from 3000 to
#       60000 steps between evals. Step-keyed keeps eval index meaning 3000
#       decision-steps per env, comparable with every run in analysis/.
#   --checkpoint-every 5
#       An update costs ~14s at T=50 but ~280s at T=1000; 20 updates would
#       risk ~93 min of work per crash late in the anneal. A checkpoint is
#       ~2.4MB, so saving more often costs nothing measurable.
#   default --checkpoint-dir / --log-csv / --eval-dir (checkpoints/, replays/)
#       This is the live run, so it uses the regular directories -- the
#       replay server lists training/replays/ non-recursively, so a subfolder
#       hides its evals. It ran in checkpoints/gamma_anneal/ until update
#       1835, when the gamma=0.98 run was archived to
#       checkpoints/archive/run_2026-09-11_bbf-adamw_gamma098_rollout50/
#       and this run moved up. Its log has had the gamma/rollout_length
#       columns since row 0.
#
# Cost, stated up front:
#   * Memory: T=1000 x 10 envs x 512x512 int8 tile_grid is 2.6GB in the
#     rollout buffer, ~5.2GB at peak (ppo_update stacks a second copy). That
#     does not fit beside a second run on this box. Measured with one real
#     T=1000 update: 8.3GB host RAM for the training tree, 5.6/6.1GB VRAM
#     (ppo_update moves the whole rollout's tile_grid onto the GPU).
#   * Time: 951k decision steps per env over the anneal, 6.3x a constant
#     T=50 run's 3000 updates -- ~74h at this box's current rate.
#   * Optimizer pressure: minibatch stays 128, so gradient steps per update
#     grow with the rollout, 32 -> 632 at 8 epochs. The KL early stop will
#     likely start cutting updates short late in the anneal; watch
#     stopped_early and entropy in the log.
#
# To warm-start a NEW anneal from some other run's checkpoint instead, give it
# its own --checkpoint-dir holding that latest.pt and pass
# --gamma-anneal-start <its update>; do not point it at checkpoints/, which
# holds this run.

set -euo pipefail
cd "$(dirname "$0")"

# ARCHIVED 2026-09-14 at update 2560. This run now lives in
# checkpoints/archive/run_2026-09-13_gamma-anneal098-0999_rollout-horizon/.
# The next run will reuse checkpoints/, and running the command below as-is
# would resume THAT run's latest.pt under this run's gamma schedule and
# horizon-matched rollout -- silently, since checkpoints do not store
# hyperparameters. Kept for provenance. To genuinely resume this run, copy the
# archive folder out to its own directory and pass
# --checkpoint-dir/--log-csv/--eval-dir pointing at it.
if [ "${I_KNOW_THIS_RUN_IS_ARCHIVED:-}" != "1" ]; then
  echo "run_gamma_anneal.sh: this run is archived; see the comment in this script." >&2
  echo "No run is live. Start the next one from a new pinned run_*.sh launcher." >&2
  exit 1
fi

exec /home/developer/miniconda/envs/openfront-rl/bin/python -u train.py \
  --num-envs 10 \
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
