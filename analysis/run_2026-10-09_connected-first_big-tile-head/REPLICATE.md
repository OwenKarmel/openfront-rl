# Replicating the big-tile-head run on the GTX 1660 box

The reference run (this folder) trained on the RTX 5000 Ada workstation. This
reruns the identical config on the GTX 1660 box and checks whether it lands in
the same place. Training is not seeded, so a replication is **statistical**:
the same curriculum milestones in roughly the same number of updates, the same
behavioural signatures, the same end point — not identical numbers.

Only two things differ from the reference command, and neither changes the
training config: `--device cuda:0` (the box has one GPU; the launcher pins
`cuda:1`) and `--eval-envs` (how many eval games run in parallel; it changes
wall time only).

## 1. Get the code

```bash
cd ~/openfront-rl            # wherever the repo lives on the 1660 box
git fetch && git checkout rework
git submodule update --init  # OpenFrontIO stays at 44e03d8, as before
(cd OpenFrontIO && npm ci)
(cd env-bridge && npm ci)
```

Python is the existing `openfront-rl` conda env (the launcher's default path).
It needs torch, numpy and gymnasium, as for every earlier run. The reference
ran Node v22.17.0; the box's own Node (the README says v24) is fine.

## 2. Check the box is free

- Nothing else training — not the beta=2 run. This launcher writes to its own
  `training/checkpoints/run3_big_tile_head/`, so it never touches
  `checkpoints/latest.pt`, but two runs will not fit in 6 GB of VRAM.
- `free -m` and `df -h /tmp` (`/tmp` is tmpfs there — see CLAUDE.md), and
  `nvidia-smi` for VRAM.
- `nproc`. Use `--eval-envs` ≈ `nproc − 12`, at least 4 (training keeps 10 env
  workers plus the Python process).

## 3. Run the quick tests

```bash
/home/developer/miniconda/envs/openfront-rl/bin/python tests/test_big_tile_head.py   # ~16 s on GPU
/home/developer/miniconda/envs/openfront-rl/bin/python tests/test_neutral_boats.py
/home/developer/miniconda/envs/openfront-rl/bin/python tests/test_eval_suite.py
```

All three must print `ALL PASSED`.

## 4. Launch

```bash
cd training
mkdir -p checkpoints/run3_big_tile_head
nohup ./run_connected_first_bighead.sh --device cuda:0 --eval-envs <N> \
  > checkpoints/run3_big_tile_head/train_stdout.log 2>&1 &
```

The first lines of `train_stdout.log` must read:

```
device=cuda:0 ... reward_mode=shaping_only potential_beta=2.0 param_entropy_norm=False tile_head=big
env flags: OPENFRONT_EXPAND_MASK=1 OPENFRONT_BOAT_MASK=1 OPENFRONT_NEUTRAL_BOATS=1
curriculum connected-first: easy/connected -> easy/small -> medium/small -> hard/small -> impossible/small; ...
```

If any of those differ, stop: it is not the same run.

**Expect it to be slow.** The reference took 4.7 h of updates plus 8.3 h of
eval suites to reach update 1437, using 32 eval workers on 48 threads. Eval
time scales roughly with 1/`--eval-envs`, so at 8 workers expect the suites
alone to take on the order of a day or more, plus slower updates on the 1660.
Do not change `--eval-every-steps` or `--eval-episodes` to save time:
promotion can only happen at an eval, so they change when stages are reached.

**Memory.** Within the replicated window (to update ~1450) the rollout grows
only to T ≈ 210, well below the T = 1000 where the beta=2 run peaked at 5.6
of 6.1 GB VRAM with the smaller default head. If you continue past ~2000
updates, watch `nvidia-smi`: the big head adds activations, and T = 1000 may
not fit.

## 5. Compare while it runs

From the repo root:

```bash
python analysis/run_2026-10-09_connected-first_big-tile-head/summarize_run.py \
  analysis/run_2026-10-09_connected-first_big-tile-head \
  --compare training/checkpoints/run3_big_tile_head
```

This prints both runs stage by stage: promotion update, eval win rates, the
training win rate, the action mix, the KL early-stop rate and timing. The
sibling default-head run (`../run_2026-10-09_connected-first`) is a second
sample of nearly the same config, and the bands below are set from both runs.

## 6. Did it replicate?

| check | reference (big head) | default-head sibling | replicated if |
|---|---|---|---|
| startup lines | as above | — | identical (hard requirement) |
| left easy/connected (stage 1) at update | 471 | 166 | ≤ 1000 |
| left easy/small (stage 2), updates after entering | 37 | 267 | ≤ 400 |
| reached medium at update | 508 | 433 | ≤ 1400 |
| medium: eval win rate median | 0.02 | 0.08 | ≤ 0.15 |
| medium: at least one suite ≥ 30/64 before promotion | yes (max 51) | yes (max 49) | yes |
| reached hard at update | 1373 | 1432 | ≤ 2500, or still at medium with the above holding |
| KL early-stop rate per stage | 0.90–0.97 | 0.65–0.77 | ≥ 0.85 (the big head's signature) |
| P(`boat_attack`) at medium | 0.20 | 0.16 | 0.10–0.30 |
| P(`noop`) at medium | 0.45 | 0.52 | 0.35–0.60 |
| boat-target entropy / max, final checkpoint (`probe_tile_head.py`) | 0.92–0.96 | 0.98–0.99 | 0.85–0.97 |
| final checkpoint vs impossible (`eval_checkpoint.py`) | 1/64 | 1/64 | ≤ 4/64 |

Read it as a whole, not cell by cell. With two seeds behind every band, one
miss is not a failure. Clear failures are: different startup lines; never
leaving easy/connected by update ~1500; a KL early-stop rate like the
default head's, which suggests the big head is not in use; or a crash or OOM.

Hardware-dependent numbers (update seconds, eval seconds) are expected to
differ, and are not part of the check.

## 7. Afterwards

Final checkpoint vs impossible, the same 64 games as the reference:

```bash
cd training
OPENFRONT_EXPAND_MASK=1 OPENFRONT_BOAT_MASK=1 OPENFRONT_NEUTRAL_BOATS=1 \
  /home/developer/miniconda/envs/openfront-rl/bin/python eval_checkpoint.py \
  --checkpoint checkpoints/run3_big_tile_head/latest.pt --difficulties impossible \
  --device cuda:0 --envs <N> --out final_eval_impossible.csv
```

Boat targeting, comparable with the table above:

```bash
OPENFRONT_EXPAND_MASK=1 OPENFRONT_BOAT_MASK=1 OPENFRONT_NEUTRAL_BOATS=1 \
  python analysis/run_2026-10-09_connected-first_big-tile-head/probe_tile_head.py \
  training/checkpoints/run3_big_tile_head/latest.pt
```

Then archive per CLAUDE.md. Commit the replication's logs and summary as a new
folder under `analysis/` (e.g. `run_<date>_connected-first_big-tile-head_gtx1660/`),
laid out like this one, with `summarize_run.py --compare` output in its README.
