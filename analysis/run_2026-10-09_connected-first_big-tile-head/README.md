# Connected-first run, big tile head (2026-10-09 → 10)

Identical to `../run_2026-10-09_connected-first/` except `--tile-head big`:
`boat_attack`'s target head is `BigTileHead` (`training/models/network.py`)
instead of a 1×1 conv. Launcher: `training/run_connected_first_bighead.sh`,
left enabled because **this is the run to replicate on the GTX 1660 box —
see [REPLICATE.md](REPLICATE.md).**

It reached hard, like the default-head run, and its boat targeting improved
a little. It did not beat impossible.

## Files

| file | what |
|---|---|
| `train_log.csv` | one row per PPO update |
| `eval_log.csv` | one row per eval suite (64 sampled games on fixed seeds) |
| `eval_episodes.csv` | one row per eval game |
| `train_stdout.log.gz` | the run's stdout, gzipped |
| `final_checkpoint_update1435.pt` | weights + optimizer at the stop (big tile head) |
| `final_eval_impossible.csv` | final checkpoint vs impossible, 64 games |
| `final_eval_impossible_scripted_baseline.csv` | the scripted threshold policy on the same 64 games |
| `summary.txt`, `summary.json` | per-stage summary (`summarize_run.py`) |
| `summarize_run.py` | builds that summary from any run's logs; `--compare` sets two runs side by side |
| `probe_tile_head.py` | how random a checkpoint's boat targeting is, on real games |
| `REPLICATE.md` | how to rerun this on the GTX 1660 box and judge the result |

Replays are archived on the RTX 5000 Ada workstation in
`training/checkpoints/archive/run_2026-10-09_connected-first_big-tile-head/`
(gitignored).

## Setup

- **Code:** branch `rework`, commit `fee3dc4` (`--tile-head big`),
  OpenFrontIO `44e03d8`. Config: exactly `../run_2026-10-09_connected-first/`'s
  plus `--tile-head big` (whole network 199k → 530k parameters).
- **Hardware:** RTX 5000 Ada workstation, GPU 1 (`--device cuda:1`), Threadripper
  7960X (48 threads), `--eval-envs 32`, Node v22.17.0, torch 2.7.0, numpy
  1.26.4, gymnasium 1.4.0. Ran alongside the default-head run throughout.
- **Duration:** started 2026-10-09 ~22:40 EDT, stopped by hand 2026-10-10 11:38
  EDT at update 1437: 4.7 h in updates + 8.3 h in eval suites. **Evals took
  ~64% of wall time.**

## Results

| stage | updates | evals (wins of 64, in order) | outcome |
|---|---|---|---|
| easy / connected maps | 0–470 | 4 3 0 1 0 7 6 4 40 57 | promoted at 471 |
| easy / all 22 maps | 471–507 | 63 | promoted at 508 |
| medium / all 22 maps | 508–1372 | 38 suites: median 1.5/64, max 51/64 | promoted at 1373 |
| hard / all 22 maps | 1373–1437 | 5 6 1 1 | stopped |

| | big head (this run) | default head (sibling) |
|---|---|---|
| promoted to easy/small, medium, hard at update | 471, 508, 1373 | 166, 433, 1432 |
| medium: eval win rate mean / median | 0.09 / 0.02 | 0.17 / 0.08 |
| medium: training win rate | 0.09 | 0.19 |
| P(`boat_attack`) at medium, hard | 0.20, 0.24 | 0.16, 0.16 |
| KL early-stop rate (all stages) | **0.90–0.97** | 0.65–0.77 |
| boat-target entropy / max, final checkpoint | 0.92–0.96 | 0.98–0.99 |
| tourney4 probe: boats launched / sampled | 68 / 112 | 3 / 92 |
| final checkpoint vs impossible | 1/64 | 1/64 |
| scripted threshold policy vs impossible, same seeds | 18/64 | 18/64 |

What this does and does not show:

- **The head learned a little.** Its targeting is measurably less random and
  far more of its boats launch, but it is still close to uniform.
- **No curriculum speed-up.** Both runs reached hard within ~60 updates of
  each other; this one was slower on the first stage and faster on the
  second. With one seed each, neither difference is evidence.
- **It updates less.** The KL early stop cut 90–97% of its updates short (the
  default head: 65–77%), probably because the big head's log-probs move more
  per step. The README already suspects a leftover tiny minibatch of
  inflating `approx_kl`; worth fixing before reading much into this run.
- **Neither run comes close to impossible**, while a 10-line script on the
  same action space wins 28% there. The learner, not the action space, is
  the limit.
- Eval win rate swings hard between consecutive suites on the same seeds
  (e.g. 30 → 21 → 51 at medium), so single suites mean little; see the
  per-stage medians.

Supervised check of the head itself (`tests/test_big_tile_head.py`): on
"the legal cell nearest AGENT's territory" it reaches ~98% accuracy vs ~64%
for the default head, so under a direct signal both can aim; the RL signal
is the bottleneck.
