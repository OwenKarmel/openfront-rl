# Territory moving-average analysis

Same analysis as `../../run_2026-09-06_winner-fix_gamma098/` and
`../../run_2026-09-07_gamma0995/`, for the BBF-style run. Deliberately
identical construction -- same metric, same window, same axes limits, same
80% reference line, same colors -- so all three plots can be compared by eye.

## Files

- **`build_dataset.py`** — scrapes every `eval vs <difficulty>: {...}` line
  from `training/checkpoints/nohup_launch.log` and writes
  `territory_moving_average.csv`.
- **`plot_territory_moving_average.py`** — reads that CSV, writes
  `territory_moving_average.png`.
- **`territory_moving_average.csv`**, **`territory_moving_average.png`**

Regenerate with:

```bash
python build_dataset.py && python plot_territory_moving_average.py
```

## Differences from the earlier versions of this script

**Log file.** Reads `nohup_launch.log`, not `train_stdout.log`. This run is
launched by `training/run_bbf_gamma098.sh` under nohup, which redirects
train.py's stdout there; no `train_stdout.log` exists for this run.

**Eval cadence, and why the x axis is still comparable.** This run evaluates
every 60 updates; the earlier two evaluated every 10. That was chosen to hold
*experience between evals* constant at 3000 decision-steps per env
(60 x rollout 50 here, 10 x rollout 300 there), because eval cost is fixed
(one full greedy game) while update cost fell 6x with the shorter rollout.
So **eval index means the same amount of gameplay in all three runs** and the
three x axes line up, even though the update numbers do not.

## Run configuration

`gamma=0.98`, `rollout_length=50`, `epochs=8`, AdamW `weight_decay=0.1`,
`num_envs=10`, `entropy_coef=0.02`, fresh weights from update 0. Rollout
length is sized to exactly one discount horizon: `1/(1-0.98) = 50` decision
steps, and one decision step is 1.0s of game time.

## What the run shows (as of eval 37 / update ~2299, 8.5h wall-clock)

**The moving average is flat.** First windowed value 0.0383, last 0.0389,
peak 0.0600. Over 38 evals and ~2300 updates there is no upward trend in
greedy-eval territory. Zero AGENT wins; best single eval 0.2405, well under
the 0.80 domination threshold.

At **equal experience** (first 38 evals of each run) it is nonetheless the
strongest of the three on the mean:

| first 38 evals | mean `peak_territory_frac` | max | AGENT wins |
|---|---|---|---|
| **this run** (gamma=0.98, rollout 50, BBF) | **0.0417** | 0.2405 | 0 |
| run_2026-09-06 (gamma=0.98) | 0.0307 | 0.4625 | 0 |
| run_2026-09-07 (gamma=0.995) | 0.0222 | 0.4184 | 0 |

All three are at zero wins this early, and both earlier runs were also flat
near the floor at this point -- gamma0995 did not leave the floor until
~eval 78. So a flat first 38 evals is **not** yet evidence the BBF changes
failed; it is the same shape the other two showed before they moved.

The higher mean but lower max is consistent with a policy that is more
uniformly mediocre rather than occasionally lucky: this run never had the
0.42-0.46 outlier evals the other two did.

## The training/eval gap is the thing to watch

`train_log.csv` reports a **12.6% win rate over 374 training episodes**,
while greedy eval is **0/38**. Training episodes are played with the
stochastic (sampled) policy; evals use argmax. Two supporting observations:

- `noop` is the modal action in most windows (0.42-0.64 mean probability),
  so an argmax policy plausibly noops far more often than a sampled one.
- Eval episodes average **25,382 ticks** (~42 min of game time), well above
  the ~16,862-tick average measured across the earlier runs' replays --
  long, passive games rather than decisive ones.

That is the pattern you would expect if the greedy policy is substantially
more passive than the stochastic one. It is a hypothesis consistent with the
aggregate numbers, not a proven mechanism -- confirming it needs per-state
argmax analysis, not rollout-averaged action probabilities.

## Metric

`peak_territory_frac` — AGENT's share of the map's **land** tiles at its
high-water mark within the episode, from `territory_fraction()` in
`train.py` (computed from `tile_grid` class proportions, so it stays valid on
the resized `map_pool` canvas). Peak rather than final, because an episode
can reach a high share and then lose ground before ending.

The dotted line at **0.80** is `percentageTilesOwnedToWin()` for FFA
(`Config.ts`) -- the threshold at which `WinCheckExecution` ends the game.
