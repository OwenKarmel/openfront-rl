#!/bin/bash
# Launches (or resumes) the HARD-STAGE TERMINAL-REWARD arm, config pinned.
# GTX 1660 box (cuda:0, 12 threads -> --eval-envs 8).
#
# Question (2026-10-14): does adding the +1/-1 terminal reward get the agent
# moving at hard/small? Every connected-first run stalled there under
# --reward-mode shaping_only: the GTX 1660 replication of the big-tile-head
# run (checkpoints/run3_big_tile_head/, paused at update ~2305) won 12 of
# 6,144 hard eval games over 870 updates while its mean peak territory slid
# 0.31 -> 0.22. Under shaping_only a loss at 13% territory still beats the
# starting position and nothing marks a win, so at hard -- where wins are
# ~1% of games -- there is almost no signal separating winning from losing.
#
# Identical to run_connected_first_bighead.sh EXCEPT:
#   --reward-mode shaping_and_terminal  (the change under test)
#   --device cuda:0 --eval-envs 8       (this box; wall time only)
#   its own checkpoint/log/replay dirs
#
# START POINT: run3_big_tile_head kept no checkpoint at its promotion to hard
# (update 1432; latest.pt is overwritten every 5 updates), so this starts from
# the big-tile-head REFERENCE run's final checkpoint, update 1435 -- same
# config, 62 updates after ITS promotion to hard at 1373, curriculum stage 3
# (hard/small). Seeded below on first launch only. Resume rebuilds gamma, the
# rollout length and the eval schedule from the update number, so they pick
# up exactly where the reference left off. CONFOUND when comparing with
# run3_big_tile_head: different starting weights (different seed), not only
# a different reward. analysis/run_2026-10-14_bighead-hard-terminal/
# compare_stage.py aligns the two by updates since entering hard.

set -euo pipefail
cd "$(dirname "$0")"

export OPENFRONT_EXPAND_MASK=1
export OPENFRONT_BOAT_MASK=1
export OPENFRONT_NEUTRAL_BOATS=1

RUN=run4_bighead_hard_terminal
SEED_CKPT=../analysis/run_2026-10-09_connected-first_big-tile-head/final_checkpoint_update1435.pt
mkdir -p "checkpoints/$RUN" "replays/$RUN"
if [ ! -e "checkpoints/$RUN/latest.pt" ]; then
  cp "$SEED_CKPT" "checkpoints/$RUN/latest.pt"
  echo "seeded checkpoints/$RUN/latest.pt from $SEED_CKPT"
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
  --eval-every-steps 3000 \
  --curriculum connected-first \
  --eval-episodes 64 \
  --promote-threshold 0.75 \
  --tile-head big \
  --checkpoint-dir "checkpoints/$RUN" \
  --log-csv "checkpoints/$RUN/train_log.csv" \
  --eval-log-csv "checkpoints/$RUN/eval_log.csv" \
  --eval-dir "replays/$RUN" \
  --device cuda:0 \
  --eval-envs 8 \
  "$@"
