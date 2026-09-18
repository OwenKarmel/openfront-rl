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

Resume the current run with `training/run_gamma_anneal_shaping_only_beta2.sh`
(the gamma anneal with `--reward-mode shaping_only --potential-beta 2`),
never by retyping the `train.py` command. The two previous anneal runs are
archived at
`training/checkpoints/archive/run_2026-09-17_gamma-anneal098-0999_shaping-only/`
and `.../run_2026-09-13_gamma-anneal098-0999_rollout-horizon/`; their
launchers `training/run_gamma_anneal_shaping_only.sh` and
`training/run_gamma_anneal.sh` are disabled, like `run_bbf_gamma098.sh`.
Checkpoints store weights, optimizer state and curriculum position but
**not** hyperparameters, so a resume that omits a flag does not fail — it
silently continues under argparse defaults, restoring the ±1 terminal
reward, flattening the shaping potential back to linear, dropping the gamma
schedule and snapping the rollout to 64 mid-run with nothing in the logs to
record it.

The live run always uses the regular `training/checkpoints/` and
`training/replays/` (flat — the replay server does not read subfolders).
Finished runs go to `training/checkpoints/archive/run_<date>_<descriptor>/`
with `README.md`, `latest.pt`, `train_log.csv`, `train_stdout.log`,
`notified_milestones.json` and `replays/`; see the existing folders there.
When archiving, disable the old launcher and repoint that run's `analysis/`
scripts at the archive, or both will silently read the next run's files.

Python lives in the `openfront-rl` conda env:
`/home/developer/miniconda/envs/openfront-rl/bin/python`. The base env has no
torch.

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
