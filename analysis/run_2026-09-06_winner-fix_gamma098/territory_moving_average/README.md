# Territory moving-average analysis

Answers: is the agent actually gaining territory over training, or is the
per-eval number just bouncing around? Produced in response to "plot a moving
average of average territory the agent controls versus evaluation run."

Raw per-eval territory in this project is close to unreadable: every eval
episode draws a random map from `SMALL_MAPS`, and those span 210k
(`onion`) to 1.1M (`didier`) land tiles with very different geography, so
two consecutive evals can differ by an order of magnitude for reasons that
have nothing to do with the policy. The moving average is what makes a trend
visible; the raw points are kept in the background so the noise level stays
honest.

## Files

- **`build_dataset.py`** — scrapes every `eval vs <difficulty>: {...}` line
  out of the live run's `training/checkpoints/train_stdout.log` (each is one
  greedy/argmax eval episode, in chronological order, so the row index *is*
  the evaluation-run number) and writes `territory_moving_average.csv`.
- **`plot_territory_moving_average.py`** — reads that CSV, writes
  `territory_moving_average.png`.
- **`territory_moving_average.csv`** — 197 eval rows.
- **`territory_moving_average.png`** — the plot.

Regenerate with:

```bash
python build_dataset.py && python plot_territory_moving_average.py
```

## Metric

`peak_territory_frac` — AGENT's share of the map's **land** tiles at its
high-water mark within that episode. Computed by `territory_fraction()` in
`train.py` from `tile_grid`'s own class proportions rather than from raw
tile counts, so it remains valid on the resized `map_pool` canvas (see
`RESIZE_DIM` in `openfront_env.py`). Peak rather than final because an
episode can reach a high share and then lose ground before ending.

The dotted line at **0.80** is not an arbitrary reference: it is
`percentageTilesOwnedToWin()` for FFA (`Config.ts`), i.e. the threshold at
which `WinCheckExecution` ends the game and declares a winner. Every AGENT
win in this run sits on that line.

## What the run shows

The run switched hyperparameters mid-flight at update 710 (marked on the
plot), so it is really two segments:

| segment | evals | mean `peak_territory_frac` |
|---|---|---|
| `gamma=0.999`, `max_steps=20000` | 71 | **0.0513** |
| `gamma=0.98`, `max_steps=2500` | 126 | **0.1930** |

Before the switch the moving average is flat at ~0.02-0.07 for 70 evals --
no learning. After it, the average climbs to ~0.20-0.30 and **all but one of
the 8 AGENT wins occur**, clustered from eval ~108 onward.

Caveats worth keeping attached to this plot:

- **Two variables changed at once** (`gamma` *and* `max_episode_steps`), so
  the improvement cannot be attributed to either alone.
- **The `max_steps` cut is partly mechanical**: capping episodes at 2500
  steps raises episodes-per-update ~3x, which alone changes the eval
  population.
- **35% of post-switch evals are truncated** by that cap, which activates
  the known GAE truncation-bootstrap bug (`dones = terms | truncs` in
  `train.py`; see the archived run's README). Its footprint is ~0.02% of
  training samples, so it is unlikely to be shaping this curve, but the
  ceiling should be read with it in mind.
- `approx_update` in the CSV is derived as `(eval_index + 1) * eval_every`,
  which is exact only while `--eval-every` stays at 10 for the whole run.
