# openfront-rl

Self-play PPO agent vs OpenFrontIO's built-in Nation AI. `README.md` is the
authoritative spec for architecture, action space, hyperparameters and status
— read it rather than inferring from code when the two seem to disagree, and
update it when you change what it documents.

## Roadmap

@.claude/roadmap.md

It covers only unbuilt work (Phases 2 and 3). Shipped designs are pruned out
of it as they land, so the README's Architecture / Action space / Status
sections — not the roadmap — are the live spec for anything already built.

## Running training

Two machines. The **GTX 1660 box** (conda env below, `/tmp` is tmpfs) ran
every run up to the beta=2 one, resumed with
`training/run_gamma_anneal_shaping_only_beta2.sh` (the gamma anneal with
`--reward-mode shaping_only --potential-beta 2`). The **RTX 5000 Ada
workstation** (2 GPUs, shared — follow the GPU-reservation check in your
global instructions; Python in this repo's `.venv`, Node in
`/data/owen/tools`, root disk nearly full so keep temp files on `/data`) ran
the connected-first runs of 2026-10-09/10, now finished and archived in
`training/checkpoints/archive/run_2026-10-09_connected-first*/`, with logs
and summaries committed under `analysis/`.

**Next:** replicate the big-tile-head run on the GTX 1660 box —
`analysis/run_2026-10-09_connected-first_big-tile-head/REPLICATE.md`. Its
launcher `training/run_connected_first_bighead.sh` is the only
connected-first launcher still enabled; `run_connected_first.sh` and
`run_connected_first_paramnorm.sh` are disabled like the older archived
ones (`run_gamma_anneal_shaping_only.sh`, `run_gamma_anneal.sh`,
`run_bbf_gamma098.sh`).

Never resume a run by retyping the `train.py` command. Checkpoints store
weights, optimizer state and curriculum position (plus, for connected-first
runs, the curriculum ladder) but **not** hyperparameters, so a resume that
omits a flag does not fail — it silently continues under argparse defaults,
restoring the ±1 terminal reward, flattening the shaping potential back to
linear, dropping the gamma schedule, swapping the tile head or reverting the
env-bridge flags, with nothing in the logs to record it. `train.py` refuses
to resume a checkpoint trained under a different curriculum ladder, and a
big-head checkpoint will not load into the default head. Resuming a
pre-eval-suite run with the current `train.py` switches its eval/promotion
gate to suites — do that from that run's own commit if it must stay
comparable.

The live run always uses the regular `training/checkpoints/` and
`training/replays/` (flat — the replay server does not read subfolders).
Finished runs go to `training/checkpoints/archive/run_<date>_<descriptor>/`
with `README.md`, `latest.pt`, `train_log.csv`, `train_stdout.log`,
`notified_milestones.json` and `replays/` — plus `eval_log.csv` and
`eval_episodes.csv` for runs with eval suites; see the existing folders there.
When archiving, disable the old launcher and repoint that run's `analysis/`
scripts at the archive, or both will silently read the next run's files.

On the GTX 1660 box, Python lives in the `openfront-rl` conda env:
`/home/developer/miniconda/envs/openfront-rl/bin/python` (the launchers'
default; override with `OPENFRONT_PYTHON`). The base env has no torch.

## Before starting a long run

Check `df -h /tmp` as well as `free -m`. `/tmp` is **tmpfs (RAM-backed)** on
this box, so large scratchpad artifacts consume physical RAM and will
OOM-kill training. `free -m` reports this as *shared*, which is easy to
misread as unrelated. Move large files to real disk rather than deleting them
— some may be data that was explicitly asked for.

## Analysis conventions

Plots of the same kind are built **identically across runs** — same metric,
window, axis limits and colors — so they can be compared by eye. Restyling
one breaks the only thing the family is for. See `analysis/README.md`, which
also records which cross-run comparisons are confounded and why.
