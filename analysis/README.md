# Analysis

One directory per training run. Each analysis inside is self-contained and
rerun-safe: a `build_dataset.py` that scrapes the run's own logs or replays,
a plot script, the generated CSV/PNG, and a README saying what the run shows
and where the comparison caveats are.

Plots of the same kind are **deliberately built identically across runs** —
same metric, same window, same axis limits, same colors — so they can be laid
against each other by eye. Restyling one breaks the only thing the family is
for.

## Runs

| directory | run | analyses |
|---|---|---|
| `run_2026-09-02_pre-naval_num-envs8/` | pre-naval, 8 envs, `onion` | `territorial_extent/`, `territory_as_game_progresses/` |
| `run_2026-09-06_winner-fix_gamma098/` | gamma=0.999 (upd 0-709) then gamma=0.98 (upd 710+) | `territory_moving_average/` |
| `run_2026-09-07_gamma0995/` | gamma=0.995, `max_episode_steps=2500`, cold start | `territory_moving_average/` |
| `run_2026-09-11_bbf-adamw_gamma098_rollout50/` | gamma=0.98, rollout 50, epochs 8, AdamW wd=0.1, cold start | `territory_moving_average/`, `training_win_rate/`, `tile_gain_vs_episode_progress/` |
| `run_2026-10-09_connected-first/` | eval suites + `connected-first` + all three env-bridge flags, beta=2 hyperparameters; reached hard | logs, `summary.txt`, final checkpoint and impossible eval (no plots) |
| `run_2026-10-09_connected-first_big-tile-head/` | the same with `--tile-head big`; reached hard. **The run to replicate on the GTX 1660 box** | logs, `summary.txt`, `summarize_run.py`, `probe_tile_head.py`, `REPLICATE.md` |

Runs with eval suites (from `training/run_connected_first.sh` on) no longer
print one `eval vs <difficulty>: {...}` line per eval for `build_dataset.py`
to scrape. Each eval is a suite of fixed-seed episodes logged to
`eval_log.csv` (one row per suite) and `eval_episodes.csv` (one row per game)
beside the run's checkpoint — read those instead. Their win rates are not
comparable with the earlier runs' greedy single-episode evals.

## Cross-run comparison is mostly confounded — check before comparing

Every pair of runs differs in more than one variable, and two traps have
already caught out a naive comparison:

- **`run_2026-09-06` is not a gamma=0.98 run for its first ~700 updates.**
  It ran gamma=0.999 until update 710. Its first 423 training episodes span
  updates 1-658, so a "first N episodes" comparison against it is comparing
  against **gamma=0.999**, not gamma=0.98.
- **`run_2026-09-07` used `max_episode_steps=2500`** against every other
  run's 20000. A truncated episode scores as a loss (`EnvServer.ts`,
  terminal reward -1), so a tighter cap mechanically deflates its win rate.

What *is* comparable: eval index, deliberately. `--eval-every` is chosen per
run to hold experience-between-evals fixed at 3000 decision-steps per env, so
eval *n* means the same amount of gameplay in every run even where the update
numbers differ by 6x.

## Regenerating

Each analysis directory has its own instructions, but the shape is the same:

```bash
cd <run>/<analysis>
python build_dataset.py && python plot_*.py
```

Analyses that need per-tick data (`tile_gain_vs_episode_progress/`) first run
`./run_all.sh`, which re-simulates every replay through the real OpenFrontIO
core via `TerritoryTimeSeries.ts`. That is ~2 minutes for 40 episodes, is
rerun-safe (skips series it already has), and is `nice`d because training is
normally live on the same cores.
