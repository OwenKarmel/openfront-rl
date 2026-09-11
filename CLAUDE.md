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

Resume the current run with `training/run_bbf_gamma098.sh`, never by
retyping the `train.py` command. Checkpoints store weights, optimizer state
and curriculum position but **not** hyperparameters, so a resume that omits a
flag does not fail — it silently continues under argparse defaults, flipping
`gamma` 0.98→0.99 and rollout 50→64 mid-run with nothing in the logs to
record it.

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
