# Training win-rate analysis

Win rate against the Nation AI over the course of training, with the
greedy/argmax eval win rate on the same axis. Companion to
`../territory_moving_average/`, which plots territorial share from the evals
only; this one uses the far denser training-episode signal.

## Files

- **`build_dataset.py`** — expands `training/checkpoints/train_log.csv`'s
  per-update `win_rate` column into one row per training episode, writes
  `training_win_rate.csv`.
- **`plot_training_win_rate.py`** — reads that CSV plus the sibling
  `../territory_moving_average/territory_moving_average.csv` (single source
  of truth for eval outcomes), writes `training_win_rate.png`.

Regenerate with:

```bash
python build_dataset.py && python plot_training_win_rate.py
```

## Why the data is expanded to one row per episode

`train.py` logs `win_rate` as (episodes won / episodes finished) *within that
update's rollout*. At rollout 50 x 10 envs only ~0.18 episodes finish per
update, so **84% of rows are `nan`** and the rest are quantised to
{0, 0.5, 0.75, 1.0}. Plotting the column directly would be almost all holes
and spikes, and averaging it over updates would weight a 1-episode update
equally with a 4-episode one. Expanding to one row per episode makes the
moving average a true trailing mean over episodes.

The training mean uses a trailing **50-episode** window. At this run's ~14%
base rate that carries about +/-5pp of binomial noise, which is the
resolution the line should be read at. The eval series uses a trailing 12 --
`CurriculumScheduler.window` -- so the plotted line is exactly the quantity
the promotion gate tests against its 0.75 threshold.

## What the run shows (423 episodes, updates 12-2397)

**Overall training win rate 13.95% (59/423), trending up.**

| segment | win rate |
|---|---|
| quarter 1 (ep 0-104, upd 12-585) | 0.124 |
| quarter 2 (ep 105-209, upd 588-1270) | 0.133 |
| quarter 3 (ep 210-314, upd 1274-1798) | 0.105 |
| **quarter 4 (ep 315-419, upd 1802-2391)** | **0.200** |
| first 100 episodes | 0.130 |
| **last 100 episodes** | **0.210** |
| **last 50 episodes** | **0.280** |

Per-episode regression slope is `+0.0002/episode`, i.e. about **+8.5
percentage points across the run so far**. The trailing-50 line ends at
0.280, its maximum. The series is volatile -- it dips to 0.02 twice (around
updates 700-800 and 1750-1900) -- so the last point should not be read as an
established level; but quarter 4 being the best quarter, on 105 episodes, is
past what the volatility alone explains.

## The training/eval gap is the headline

Recent training win rate is ~28% while **greedy eval is 0/38**. The green
line sits flat on the axis for the whole run. Since promotion gates on the
eval rate crossing 0.75, the curriculum has never been close to promoting,
and difficulty has stayed `easy` for all 2394 updates.

Training episodes use the stochastic (sampled) policy; evals use argmax.
See `../territory_moving_average/README.md` for the supporting evidence that
the greedy policy is markedly more passive (noop is the modal action; eval
episodes run ~25k ticks, well above the ~17k average of the earlier runs'
replays).

## There is no clean cross-run baseline for this metric

Both obvious comparisons are confounded, and the naive version of this table
is actively misleading:

| first 423 training episodes | win rate | last 50 of those | what it actually is |
|---|---|---|---|
| **this run** | 0.1395 | 0.280 | cold start, gamma=0.98 |
| run_2026-09-07 (gamma=0.995) | 0.0804 | 0.020 | cold start, but `max_episode_steps=2500` |
| run_2026-09-06 ("gamma=0.98") | 0.5248 | 0.580 | **all 423 fall in its gamma=0.999 segment** |

- The **gamma098 archive run cannot be used here at all** for a gamma=0.98
  comparison: its first 423 episodes span updates 1-658, and that run did not
  switch to gamma=0.98 until update 710. Every one of those episodes is a
  gamma=0.999 episode. Its headline 0.5248 is a gamma=0.999 number.
- The **gamma0995 run** is a genuine cold start, but ran with
  `max_episode_steps=2500` against this run's 20000. A truncated episode
  scores as a loss (`EnvServer.ts`, terminal reward -1), so a tighter cap
  mechanically deflates its win rate. Its 0.0804 is not a like-for-like
  floor.

So the trustworthy signal here is the **within-run trend**, not the
cross-run level. A fair cross-run comparison would need a run matched on both
`max_episode_steps` and warm-start status, which does not currently exist.

## Metric

`won` is reconstructed as `round(win_rate * episodes)` per update, from
train.py's own per-update logging. Ordering of episodes within a single
update is not recoverable from the log (only the count and fraction are
recorded); wins are emitted before losses inside an update, which is
irrelevant at any sane window size. `difficulty` is carried through and is
`easy` for every episode in this run, so no curriculum change confounds the
series.
