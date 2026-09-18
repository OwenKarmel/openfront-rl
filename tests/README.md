# tests

Every test for this project's own code. OpenFrontIO's own test suite is not
here — it lives in `OpenFrontIO/` and is upstream's.

Run one directly, or all of them with `tests/run_all.sh`:

```bash
/home/developer/miniconda/envs/openfront-rl/bin/python tests/test_schedules.py
tests/run_all.sh          # all eight, in ascending cost order (~13min)
tests/run_all.sh -q       # same, only the per-file verdict
```

Each file is a plain script with its own `main()` and a non-zero exit on
failure — no pytest, no collection, no shared fixtures. They import from
`training/` via a two-line bootstrap at the top of each file:

```python
TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))
```

## Run them one at a time

`run_all.sh` is sequential deliberately. The env-backed tests each stand up
several Node env-bridge workers plus a torch process; run in parallel, or run
alongside a training run that is late in its gamma anneal, they compete for
the RAM that gets training OOM-killed. Check `free -m` first — see the
"Before starting a long run" section of `CLAUDE.md`, and note that `/tmp` is
tmpfs on this box.

## What each one covers

| test | needs | runtime | covers |
|---|---|---|---|
| `test_schedules.py` | — | 2s | `gamma_at`/`rollout_at`/`steps_before`/`horizon_steps`: the gamma anneal is geometric in (1−γ), the rollout equals exactly one discount horizon, and every schedule is a pure function of the update counter so a resume lands identically |
| `test_eval_cadence.py` | — | 1s | `eval_due`: step-keyed eval cadence fires on the same eval indices as the old update-keyed one when the rollout is constant, so eval index stays comparable across runs |
| `smoke_test.py` | env-bridge | 1s | one real episode end to end: obs shapes, action round trip, reward, termination |
| `smoke_test_multiopp.py` | env-bridge | 2s | variable-N opponent encoding: padding contributes nothing, permutation invariance, `attack_target_mask` respected, pointer-head log_prob composition, env round trip |
| `test_expand_mask.py` | env-bridge, snapshot | 71s | `OPENFRONT_EXPAND_MASK=1`: reproduces eval `0ab47d4b` (the episode that conquered its island then pulsed forever) and measures where the mass goes once `expand` is masked off |
| `test_boat_mask.py` | env-bridge, snapshot | 32s | `OPENFRONT_BOAT_MASK=1`: reproduces eval `cc83d2cc` (landlocked all game) and measures the same redistribution for `boat_attack` |
| `test_reward_modes.py` | env-bridge | 3min | `--reward-mode`: the three modes in lockstep on one seed — dynamics identical, `shaping_and_terminal == shaping_only + terminal_only` exactly, and the shaping sum really is the tile-share delta at every prefix. Plus the four rejection paths |
| `test_potential_beta.py` | env-bridge, archive | 8min | `--potential-beta`: β 0/2/5 in lockstep, reward equals `Φ(p′) − Φ(p)` exactly against the engine's own tile counts, gains and losses weighted alike by `e^{βp}`, `Φ′(0) = 1`. Plus the rejection paths |

The two mask tests are for behaviour that is **env-var gated and off by
default** (`OPENFRONT_EXPAND_MASK`, `OPENFRONT_BOAT_MASK`); they set the
variable themselves. No training run has used the masks — they were tested on
Kaggle separately.

## Fixed checkpoints, not `latest.pt`

Three tests drive a real policy. All three load a **fixed archived
checkpoint**, never `training/checkpoints/latest.pt`:

- the mask tests load
  `training/checkpoints/archive/run_2026-09-11_bbf-adamw_gamma098_rollout50/policy_snapshot_update3000.pt`
- `test_potential_beta.py` loads
  `training/checkpoints/archive/run_2026-09-17_gamma-anneal098-0999_shaping-only/latest.pt`

`latest.pt` moves with whatever run is live, so pointing at it would silently
change what these tests cover — a freshly cold-started policy reaches neither
high territory nor sustained tile losses, and `test_potential_beta.py`'s
high-`p` and loss-weighting checks would keep passing while covering nothing.
Both paths are under `training/checkpoints/`, which is gitignored, so a fresh
clone has neither: `test_potential_beta.py` skips that arm with a note and
still runs its closed-form checks, and the mask tests exit with a message
naming the missing file.
