#!/bin/bash
# Launches (or resumes) the shaping-only gamma-anneal run with a CONVEX
# shaping potential, config pinned.
#
# Checkpoints store weights, optimizer state and curriculum position but NOT
# hyperparameters, so a resume that forgets a flag silently continues under
# argparse defaults -- here that would quietly restore the +/-1 terminal
# reward, flatten the potential back to linear, drop the gamma schedule and
# snap the rollout back to 64, with nothing in the logs to record it. Always
# resume via this script, never by retyping the command.
#
# Identical to the archived shaping-only run
# (checkpoints/archive/run_2026-09-17_gamma-anneal098-0999_shaping-only/,
# launcher run_gamma_anneal_shaping_only.sh) EXCEPT --potential-beta 2, so a
# comparison against that run isolates the potential's convexity. Known issues
# inherited from it are deliberately NOT fixed here, for the same reason: the
# leftover-minibatch KL early stops, and the entropy bonus competing against a
# std-normalized advantage at spawn.
#
# Config rationale (only the new line; the rest is unchanged and explained in
# run_gamma_anneal_shaping_only.sh):
#   --potential-beta 2
#       Phi(p) = (exp(2p)-1)/2 over AGENT's tile share p, so the per-tile
#       weight is Phi'(p) = exp(2p): 1.00 at spawn, 1.65 at 25% territory,
#       2.72 at 50%, 4.95 at the 80% win threshold. Gains and losses are
#       weighted alike, because the weight is a function of the STATE -- a
#       gain/loss asymmetry would leave a cycle that pays to lose a tile
#       cheaply and retake it dearly, forever.
#
#       Why: OpenFrontIO's own economy taxes the leader. maxTroops =
#       2*(tiles^0.6 * 1000 + 50000) (Config.ts), so going from a 52-tile
#       spawn to 370k tiles -- 7,100x the territory -- buys only ~37x the
#       troop cap. The agent therefore has weak intrinsic reason to push from
#       40% toward 80%, and the previous run's sampled-policy episodes stall
#       in a 20-30% territory band. This is a direct counterweight.
#
#       Why exponential and not p^k: Phi'(0) = 1 for every beta, so beta only
#       ADDS weight above spawn and never subtracts below it. p^k has
#       Phi'(0) = 0, and spawn here is p ~ 5e-5 (52 tiles of ~1e6), so p^2
#       would weight the spawn-expand action at ~1e-4 of its current value --
#       the exact gradient the previous runs' noop collapse was eating.
#
#       Why beta=2 and not more: measured on 12 sampled-policy episodes from
#       the previous run's latest.pt (11,732 decision steps), 81% of decision
#       steps sit at p < 0.2, so convexity multiplies a small, high-variance
#       minority of steps. beta=2 raises the per-step reward std 1.64x while
#       leaving the spawn advantage unchanged, which shrinks the spawn
#       entropy-vs-advantage equilibrium z* = dA_raw/(c*sigma) to 0.61x.
#       beta=3 would cost 0.47x. That quantity is what collapsed onto noop in
#       the 2026-09-13 run, so it is not somewhere to be greedy.
#
# Cost, stated up front (measured in the previous anneal runs):
#   * Memory: the previous run reached T=891 without trouble. A T=1000 update
#     peaked at 8.3GB host RAM for the training tree with fresh env workers
#     and 5.6/6.1GB VRAM (ppo_update moves the whole rollout's tile_grid onto
#     the GPU); with grown workers that projects to ~10.3GB around update
#     3000. Does not fit beside a second run; stop the OpenFrontIO dev server
#     before then if headroom is short.
#   * Time: ~951k decision steps per env over the anneal; the previous run
#     took ~70h to reach update 2885.

set -euo pipefail
cd "$(dirname "$0")"

exec /home/developer/miniconda/envs/openfront-rl/bin/python -u train.py \
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
  "$@"
