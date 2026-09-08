#!/bin/bash
# Bootstraps a Kaggle session into a working openfront-rl training tree.
#
# Reconstructs the exact directory layout the code expects, because several
# paths are resolved relatively and will break otherwise:
#   openfront-rl/
#     OpenFrontIO/     <- cloned here at the pinned SHA
#     env-bridge/      <- from the Kaggle Dataset
#     training/        <- from the Kaggle Dataset
# In particular env-bridge/package.json invokes
# `tsx --tsconfig ../OpenFrontIO/tsconfig.json`, and openfront_env.py builds
# paths off its own location, so OpenFrontIO must be a SIBLING of env-bridge.
#
# Usage: ./setup_kaggle.sh <dataset_dir> <work_dir>
#   dataset_dir  e.g. /kaggle/input/openfront-rl-src
#   work_dir     e.g. /kaggle/working/openfront-rl

set -euo pipefail

DATASET="${1:?usage: setup_kaggle.sh <dataset_dir> <work_dir>}"
WORK="${2:?usage: setup_kaggle.sh <dataset_dir> <work_dir>}"

OPENFRONTIO_REPO="https://github.com/openfrontio/OpenFrontIO.git"
PINNED_SHA="$(cat "$DATASET/PINNED_SHA" 2>/dev/null | tr -d '[:space:]')"

echo "=== node version check ==="
# tsx requires Node 18+. Kaggle images ship Node for JupyterLab but the
# version has moved around, so fail loudly rather than deep inside a tsx
# stack trace.
if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: node not found. Install with: conda install -y -c conda-forge nodejs" >&2
  exit 1
fi
node --version
NODE_MAJOR="$(node --version | sed 's/^v\([0-9]*\).*/\1/')"
if [ "$NODE_MAJOR" -lt 18 ]; then
  echo "ERROR: node $NODE_MAJOR is too old for tsx (needs >=18)." >&2
  echo "       conda install -y -c conda-forge 'nodejs>=20'" >&2
  exit 1
fi

echo
echo "=== laying out $WORK ==="
mkdir -p "$WORK"
cp -r "$DATASET/env-bridge" "$WORK/"
cp -r "$DATASET/training" "$WORK/"
mkdir -p "$WORK/training/checkpoints" "$WORK/training/replays"

# Resume state, if the Dataset carried any.
if [ -f "$DATASET/checkpoint/latest.pt" ]; then
  cp "$DATASET/checkpoint/latest.pt" "$WORK/training/checkpoints/"
  for f in notified_milestones.json train_log.csv train_stdout.log; do
    [ -f "$DATASET/checkpoint/$f" ] && cp "$DATASET/checkpoint/$f" "$WORK/training/checkpoints/"
  done
  echo "restored checkpoint -> will RESUME"
else
  echo "no checkpoint in dataset -> will start from update 0"
fi

echo
echo "=== cloning OpenFrontIO @ ${PINNED_SHA:0:12} ==="
if [ -z "$PINNED_SHA" ]; then
  echo "ERROR: $DATASET/PINNED_SHA missing or empty." >&2
  exit 1
fi
# Full clone, not --depth 1: a shallow clone cannot check out an arbitrary
# older SHA, and pinning matters because the engine's simulation must match
# the one the checkpoint was trained against.
git clone --quiet "$OPENFRONTIO_REPO" "$WORK/OpenFrontIO"
git -C "$WORK/OpenFrontIO" checkout --quiet "$PINNED_SHA"
echo "checked out: $(git -C "$WORK/OpenFrontIO" rev-parse --short HEAD)"

echo
echo "=== installing node deps (this is the slow part, ~2-5 min) ==="
# --ignore-scripts mirrors the project's own `npm run inst` (see
# OpenFrontIO/CLAUDE.md, which says explicitly not to use plain npm install).
( cd "$WORK/OpenFrontIO" && npm ci --ignore-scripts --no-audit --no-fund --silent )
( cd "$WORK/env-bridge" && npm ci --no-audit --no-fund --silent 2>/dev/null \
    || npm install --no-audit --no-fund --silent )

echo
echo "=== verifying the env-bridge actually runs ==="
# Cheap end-to-end check BEFORE committing GPU hours: if the engine, tsconfig
# paths or map assets are wrong, this fails in seconds instead of the run
# dying an hour in.
cd "$WORK/training"
timeout 300 python smoke_test.py 2>&1 | tail -4

echo
echo "setup complete: $WORK"
