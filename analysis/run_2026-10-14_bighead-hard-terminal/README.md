# Big-tile-head at hard, shaping + terminal reward (2026-10-14 →)

**In progress.** Launcher: `training/run_bighead_hard_terminal.sh`; live
files in `training/checkpoints/run4_bighead_hard_terminal/`.

## Question

Every connected-first run stalled at hard/small under `--reward-mode
shaping_only`. The GTX 1660 replication of the big-tile-head run
(`training/checkpoints/run3_big_tile_head/`) won 12 of 6,144 hard eval games
over ~900 updates, while its mean peak eval territory slid from 0.31 to 0.22.
Hard games mostly end with the Nation AI reaching 80% while AGENT still holds
~13%. Under shaping_only that loss still nets a positive return, and nothing
marks a win. Does adding the ±1 terminal reward (`shaping_and_terminal`)
change that?

## Setup

- Identical to `run_connected_first_bighead.sh` except `--reward-mode
  shaping_and_terminal` (plus `--device cuda:0 --eval-envs 8` and its own
  directories).
- **Start point:** the replication kept no checkpoint at its promotion to
  hard, so this run starts from the big-tile-head reference's final
  checkpoint (`../run_2026-10-09_connected-first_big-tile-head/final_checkpoint_update1435.pt`),
  at hard/small, 62 updates after that run's promotion. γ, the rollout
  length and the eval schedule resume from update 1435.
- **Confound:** compared with the replication, both the reward and the
  starting weights differ (a different, unseeded run of the same config). The
  reference's own 4 hard suites (5, 6, 1, 1 wins/64) are the history of these
  starting weights under shaping_only.
- GTX 1660 box, 10 training envs, 8 eval envs.

## Comparing

```bash
python analysis/run_2026-10-14_bighead-hard-terminal/compare_stage.py \
  training/checkpoints/run3_big_tile_head \
  training/checkpoints/run4_bighead_hard_terminal --entry 1=1373
```

This aligns runs by updates since entering hard. `--entry 1=1373` dates run4
from the reference's real promotion, not from where its own log starts
(1435). The final totals cover only the span both runs have
reached. `mean_reward` is left out, because the terminal term puts it on a
different scale.
