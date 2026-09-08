# Territory moving-average analysis

Same analysis as
`../../run_2026-09-06_winner-fix_gamma098/territory_moving_average/`, for the
`gamma=0.995` experiment. Deliberately identical construction -- same metric,
same window, same axes limits, same 80% reference line -- so the two plots
can be compared by eye.

## Files

- **`build_dataset.py`** — scrapes every `eval vs <difficulty>: {...}` line
  from `training/checkpoints/train_stdout.log` and writes
  `territory_moving_average.csv`.
- **`plot_territory_moving_average.py`** — reads that CSV, writes
  `territory_moving_average.png`.
- **`territory_moving_average.csv`**, **`territory_moving_average.png`**

Regenerate with:

```bash
python build_dataset.py && python plot_territory_moving_average.py
```

## Difference from the gamma098 version of this script

That run's `build_dataset.py` computed the update number as
`(eval_index + 1) * eval_every`. That would be **wrong here**: this run was
OOM-killed at update 437 and resumed from its update-430 checkpoint, so
updates 430-437 were replayed. This version instead reads the update from
the most recent `update N` line preceding each eval, which is robust to any
number of crash/resume cycles.

The resume also cannot be detected from the update column, because the 7
lost updates produced no evals -- eval updates still increase monotonically
(429 -> 439). So `build_dataset.py` emits an explicit `segment` column that
increments on each `resumed from` line, and the plot marks the boundary from
that.

## Run configuration

`gamma=0.995`, `max_episode_steps=2500`, fresh weights from update 0. Only
gamma differs from the previous run's segment B (which used 0.98), making
this a single-variable comparison -- though see the caveat below about
warm-start.

## What the run shows (as of eval 109 / update ~1090)

The moving average is flat at ~0.03-0.06 for the first ~78 evals, then
climbs sharply to ~0.15-0.21 and holds. One AGENT win, at eval 80, right at
the 80% threshold. So `gamma=0.995` **does** learn -- it just takes ~780
updates to leave the floor.

Comparison against the previous run at **equal training from scratch**
(first 109 evals of each):

| | gamma=0.995 | previous run |
|---|---|---|
| mean `peak_territory_frac` | 0.0774 | 0.0842 |
| max `peak_territory_frac` | **0.8043** | 0.5332 |
| AGENT wins | 1 | 1 |

Essentially tied on the mean. The previous run's first 109 evals are not a
clean control though -- they are 71 evals at `gamma=0.999` plus 38 at
`gamma=0.98`, not a single setting.

Recent form is also close:

| | mean `peak_territory_frac` |
|---|---|
| this run, last 30 evals | 0.1885 |
| previous run, all of segment B (`gamma=0.98`) | 0.1912 |

**Read this cautiously.** The 0.98 number comes from a policy that was
warm-started at update 710 from an already-trained network, while this run
built up from scratch. Reaching parity from a cold start in fewer updates is
arguably the stronger result, but the two are not directly comparable and
this run is only ~half the length of the one it is being compared to.

The specific prediction motivating `gamma=0.995` -- that a ~200-step horizon
makes naval play learnable, since a full-map boat crossing (~262 steps)
retains 27% of its weight at 0.995 versus 0.5% at 0.98 -- is **not** settled
by this plot. Territory alone cannot distinguish it; that needs the
`type_prob_boat_attack` series.

## Metric

`peak_territory_frac` — AGENT's share of the map's **land** tiles at its
high-water mark within the episode, from `territory_fraction()` in
`train.py` (computed from `tile_grid` class proportions, so it stays valid on
the resized `map_pool` canvas). Peak rather than final, because an episode
can reach a high share and then lose ground before ending.

The dotted line at **0.80** is `percentageTilesOwnedToWin()` for FFA
(`Config.ts`) -- the threshold at which `WinCheckExecution` ends the game.
