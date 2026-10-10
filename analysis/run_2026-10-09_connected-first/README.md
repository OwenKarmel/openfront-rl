# Connected-first run (2026-10-09 → 10)

The first run with sampled-policy eval suites gating the curriculum, the
`connected-first` ladder and all three env-bridge flags
(`OPENFRONT_EXPAND_MASK`, `OPENFRONT_BOAT_MASK`, `OPENFRONT_NEUTRAL_BOATS`).
Launcher: `training/run_connected_first.sh` (now archived/disabled). Sibling
run with only `--tile-head big` changed:
`../run_2026-10-09_connected-first_big-tile-head/`.

**First run in the project to leave easy.** It reached hard, then stalled.

## Files

| file | what |
|---|---|
| `train_log.csv` | one row per PPO update (train.py's own log) |
| `eval_log.csv` | one row per eval suite (64 sampled games on fixed seeds) |
| `eval_episodes.csv` | one row per eval game |
| `train_stdout.log.gz` | the run's stdout, gzipped |
| `final_checkpoint_update1650.pt` | weights + optimizer at the stop (default tile head) |
| `final_eval_impossible.csv` | final checkpoint vs impossible, 64 games (`training/eval_checkpoint.py`) |
| `summary.txt`, `summary.json` | per-stage summary from `../run_2026-10-09_connected-first_big-tile-head/summarize_run.py` |

Replays and the run's checkpoint folder are archived on the RTX 5000 Ada
workstation in `training/checkpoints/archive/run_2026-10-09_connected-first/`
(gitignored).

## Setup

- **Code:** branch `rework`, commit `9e525d5` (eval suites + connected-first),
  OpenFrontIO `44e03d8`. Config pinned in `training/run_connected_first.sh`:
  the beta=2 run's hyperparameters (10 envs, shaping-only reward,
  potential β = 2, γ annealed 0.98 → 0.999 over 3000 updates with the
  rollout sized to the horizon, 8 epochs, AdamW wd 0.1, entropy 0.02,
  minibatch 128) plus `--curriculum connected-first`, 64-game eval suites
  every 3000 steps/env, promotion at ≥ 48/64.
- **Hardware:** RTX 5000 Ada workstation (GPU 0; Threadripper 7960X, 48
  threads), `--eval-envs 32`, Node v22.17.0, torch 2.7.0. It shared the
  machine with a second run the whole time (the param-norm arm, then the
  big-head run).
- **Duration:** started 2026-10-09 ~17:40 EDT, stopped by hand 2026-10-10
  11:38 EDT at update 1650: 6.5 h in updates + 11.4 h in eval suites.

## Results

| stage | updates | evals (wins of 64, in order) | outcome |
|---|---|---|---|
| easy / connected maps | 0–165 | 16 4 50 | promoted at 166 |
| easy / all 22 maps | 166–432 | 36 8 37 42 13 49 | promoted at 433 |
| medium / all 22 maps | 433–1431 | 44 suites: median 5/64, max 49/64 | promoted at 1432 |
| hard / all 22 maps | 1432–1650 | 5 9 5 3 2 1 0 1 1 4 0 0 1 0 0 1 | stopped |

- **Training win rate** (sampled, all games finished in rollouts): 0.14 at
  easy/connected, 0.62 at easy/small, 0.19 at medium, 0.04 at hard.
- **Volatility:** consecutive suites on the same 64 seeds swing by tens of
  games (50 → 4 → 37 → 13 → 49; at medium 5 → 35 → 42 → 45 → 5 → 10 → 49).
  Every map rises and falls together, so the whole policy flips; the bad
  stretches lined up with rising `boat_attack` probability.
- **Boat targeting never learned:** the default 1×1-conv tile head sat at
  88–93% of its maximum entropy by update ~305 and 98–99% at the end
  (`../run_2026-10-09_connected-first_big-tile-head/probe_tile_head.py`);
  on a tourney4 probe 3 of 92 sampled boats launched.
- **KL early stop** fired on 65–77% of updates.
- **Final checkpoint vs impossible:** 1/64. The 10-line scripted threshold
  policy won 18/64 on the same seeds
  (`../run_2026-10-09_connected-first_big-tile-head/final_eval_impossible_scripted_baseline.csv`).

See `summary.txt` for the per-stage action mix, entropy and timing.

Regenerate the summary:

```bash
cd ../run_2026-10-09_connected-first_big-tile-head
python summarize_run.py ../run_2026-10-09_connected-first --json ../run_2026-10-09_connected-first/summary.json > ../run_2026-10-09_connected-first/summary.txt
```
