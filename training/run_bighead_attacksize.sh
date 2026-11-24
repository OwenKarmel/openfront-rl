#!/bin/bash
# Launches (or resumes) RUN6: the HARD-STAGE run with the Tier-0 training
# fixes and an ATTACK-SIZE HEAD, config pinned. GTX 1660 box (cuda:0, 12
# threads -> --eval-envs 8). Prepared 2026-11-24.
#
# Warm-started from run4_bighead_hard_terminal's final checkpoint (update
# 5310, curriculum stage 3 = hard/small, connected-first ladder) via
# --init-from -- weights, the optimizer state of every matched parameter,
# update count and curriculum stage. run4's directory is only ever read.
# Gamma (0.999, anneal long finished) and the rollout length (1000) rebuild
# from the update number, exactly as for a resume. Every flag of
# run_bighead_hard_terminal.sh is copied unchanged (shaping_and_terminal,
# beta 2, gamma schedule, connected-first, big tile head, 10 envs, the three
# env-bridge flags). No self-play. What changes vs run4, and why (README
# "Status" has the evidence in full):
#
#   --kl-stop epoch      The minibatch KL early stop fired on 93-100% of
#                        updates in the last three runs while mean approx_kl
#                        was 0.004-0.011: a noisy k1 estimate on 128 samples
#                        -- or on the tiny leftover minibatch -- ended most
#                        updates early, so it is unknown how many of the 8
#                        epochs ever ran. Now: k3, averaged per epoch, checked
#                        between epochs; partial minibatch folded in.
#                        grad_steps / epochs_completed are logged per update.
#   --value-norm         PopArt value normalization. The 2026-09-13 noop
#                        collapse traced to the advantage std growing ~5x
#                        under gamma 0.999 with terminal rewards. Statistics
#                        start at identity, so run4's critic is untouched.
#   --adv-norm minibatch advantages normalized per minibatch.
#   --eval-every-steps 50000 + --canary-every-steps 10000 --canary-episodes 16
#                        run4 played 1,033 suites (66,112 games, ~311 h)
#                        against ~290 h of training -- evaluation cost more
#                        than training. Now a full 64-game suite (gates
#                        promotion, as before) every ~50 updates and a
#                        logged-only 16-game canary (fixed, map-stratified
#                        subset of the same seeds) every ~10. Estimated from
#                        run4's timings: ~14% of wall time.
#   --eval-report-exclude-maps labyrinth
#                        1,075 of run4's 1,096 hard eval wins were on
#                        labyrinth; without it run4 won 21 of ~61,000.
#                        eval_log.csv gets wins_excl/episodes_excl, and every
#                        suite prints per-map wins.
#   --attack-size-head   The agent could not choose how many troops to send:
#                        every attack sent troops/5. The Nation AI sends
#                        troops minus a 30-40% reserve once it reaches a 50-60%
#                        trigger, and a 10-line scripted policy that waits
#                        for 45% of maxTroops wins 18/64 vs impossible where
#                        every RL checkpoint wins 1/64. The head picks 10, 20,
#                        35, 50 or 75% of current troops per attack; it starts
#                        at P(20%) = 0.9 in every state.
#
# CONFOUNDS: five changes at once against run4, so a difference is not
# attributable to any one of them. THE SAME LAUNCHER WITHOUT
# --attack-size-head IS THE TIER-0-ONLY CONTROL: about a day of that is the
# cleaner check of the first four fixes on their own. Per-update numbers are
# comparable with run4's (same 10 envs, same T), eval rows are 1/16 as frequent
# and canary rows are a different, smaller sample -- compare suites with
# suites.
#
# SUCCESS BAR (README "Status" repeats it):
#   - vs impossible on the stage's eval seeds, at least the scripted policy's
#     18/64 (check with eval_checkpoint.py --checkpoint ... --difficulties
#     impossible, or analysis/eval_vs_script.py --difficulties impossible);
#   - hard win rate excluding labyrinth well above run4's 21/~61,000;
#   - eval under 15% of wall time (train_log.csv's eval_seconds_total /
#     (train_seconds_total + eval_seconds_total));
#   - full epochs actually running (epochs_completed = 8 on most updates).
#
# Safe to relaunch: with latest.pt present, train.py resumes it and ignores
# --init-from. RUN_NAME overrides the run directory -- for smoke tests only.

set -euo pipefail
cd "$(dirname "$0")"

export OPENFRONT_EXPAND_MASK=1
export OPENFRONT_BOAT_MASK=1
export OPENFRONT_NEUTRAL_BOATS=1

RUN=${RUN_NAME:-run6_bighead_attacksize}
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
  --eval-report-exclude-maps labyrinth \
  --attack-size-head \
  --checkpoint-dir "checkpoints/$RUN" \
  --log-csv "checkpoints/$RUN/train_log.csv" \
  --eval-log-csv "checkpoints/$RUN/eval_log.csv" \
  --eval-dir "replays/$RUN" \
  --device cuda:0 \
  --eval-envs 8 \
  "$@"
