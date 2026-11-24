#!/bin/bash
# Launches (or resumes) the HARD-STAGE SELF-PLAY + TERMINAL-ONLY arm, config
# pinned. GTX 1660 box (cuda:0; --num-envs 8 --eval-envs 6, see below).
#
# Continues run4_bighead_hard_terminal (run_bighead_hard_terminal.sh), which
# stopped at update 5310 (2026-11-06; killed from outside, no traceback),
# still at hard/small: 448 wins in 17,280 hard eval games over its last ~800
# updates, never near the 0.75 promotion gate. Two changes at once, so a
# difference from run4 is NOT attributable to either alone:
#
#   --reward-mode terminal_only   +1/-1 at the engine's ending only. run4 paid
#       shaping (beta=2 tile-share potential) on top of the terminal reward.
#       --potential-beta is kept for pinning only; it is inert here.
#   --expert-frac 1,1,0.5,0.5,0.5 self-play (selfplay.py, README "Self-play"):
#       from medium/small on, half of all training episodes are against a
#       copy of the agent (live network, or a frozen snapshot from a pool of
#       20, one every 20 updates) instead of the Nation AI. This run starts at
#       hard (stage 4 of 5), so it plays 50/50 from its first update. Eval and
#       promotion are still against the Nation AI only.
#
# Plus, for memory (2026-11-23): the first launch died silently 2.5 minutes
# into update 5312 (no traceback, like run4; probably OOM -- the container's
# memory peak hit 13.7 of 15 GB -- but unconfirmed). Relaunched with
#   --num-envs 8    (run4: 10) -- smaller rollout buffer, two fewer Node
#                   workers. A THIRD difference from run4: 20% less data per
#                   update, so per-update numbers are not comparable either.
#   --eval-envs 6   (run4: 8)  -- wall time only; eval results unchanged.
# and mem_logger.sh sampling memory every 30s into mem_log.csv.
#
# 2026-11-24, from update ~5415: --eval-every-steps 12000 (was 3000, as in
# run4). Evals of 64 games took ~60% of wall time at one per 3 updates; now
# one per 12 updates. Eval results per suite are unchanged, but promotion is
# checked 4x less often, and eval_log.csv rows are 4x sparser from here on.
#
# Everything else identical to run_bighead_hard_terminal.sh, plus its own
# checkpoint/log/replay dirs. train_log.csv's win_rate/episodes now count
# Nation-AI episodes only; self-play episodes are in selfplay_* columns.
#
# START POINT: run4's final checkpoint (update 5310, curriculum stage 3 =
# hard/small), COPIED in on first launch only -- run4's own directory is
# never written. Gamma (0.999, anneal finished) and the rollout length (1000)
# rebuild from the update number. The self-play pool starts empty, so the
# first 20 updates' self-play opponents are all the live network.
# notified_milestones.json is copied too, so milestones run4 already
# announced do not fire again.

set -euo pipefail
cd "$(dirname "$0")"

export OPENFRONT_EXPAND_MASK=1
export OPENFRONT_BOAT_MASK=1
export OPENFRONT_NEUTRAL_BOATS=1

RUN=run5_bighead_selfplay_terminal
SEED_DIR=checkpoints/run4_bighead_hard_terminal
mkdir -p "checkpoints/$RUN" "replays/$RUN"
if [ ! -e "checkpoints/$RUN/latest.pt" ]; then
  cp "$SEED_DIR/latest.pt" "checkpoints/$RUN/latest.pt"
  cp "$SEED_DIR/notified_milestones.json" "checkpoints/$RUN/notified_milestones.json"
  echo "seeded checkpoints/$RUN/ from $SEED_DIR (latest.pt, notified_milestones.json)"
fi

exec "${OPENFRONT_PYTHON:-/home/developer/miniconda/envs/openfront-rl/bin/python}" -u train.py \
  --num-envs 8 \
  --reward-mode terminal_only \
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
  --eval-every-steps 12000 \
  --curriculum connected-first \
  --eval-episodes 64 \
  --promote-threshold 0.75 \
  --tile-head big \
  --expert-frac 1,1,0.5,0.5,0.5 \
  --selfplay-pool-size 20 \
  --selfplay-snapshot-every 20 \
  --selfplay-latest-prob 0.5 \
  --checkpoint-dir "checkpoints/$RUN" \
  --log-csv "checkpoints/$RUN/train_log.csv" \
  --eval-log-csv "checkpoints/$RUN/eval_log.csv" \
  --eval-dir "replays/$RUN" \
  --device cuda:0 \
  --eval-envs 6 \
  "$@"
