#!/bin/bash
# Builds the payload uploaded to Kaggle as a Dataset.
#
# Deliberately does NOT include OpenFrontIO: it is a public repo and the
# notebook clones it at the pinned SHA instead (see PINNED_SHA below). That
# keeps this payload at ~2.5MB instead of ~555MB, which matters because it
# gets re-uploaded on every code change.
#
# It also does not include node_modules -- the notebook runs `npm ci` in the
# session, since native modules must be built against Kaggle's own toolchain.
#
# Why a Dataset at all rather than `git clone` of openfront-rl: at the time
# this was written the entire multi-opponent architecture, the
# winner-detection fix, recycle_every and oom_guard.sh were uncommitted
# working-tree changes. Cloning would silently get the OLD, broken code --
# the version where the agent was punished for winning. If/when this work is
# committed and pushed, switch the notebook to clone openfront-rl too and
# retire this script.
#
# Usage: ./package_source.sh [output_dir]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/openfront-rl-kaggle-payload}"

rm -rf "$OUT"
mkdir -p "$OUT/payload"

cd "$REPO_ROOT"

# --- our source (0.2MB) -----------------------------------------------------
mkdir -p "$OUT/payload/env-bridge" "$OUT/payload/training"
cp -r env-bridge/src "$OUT/payload/env-bridge/"
cp env-bridge/package.json "$OUT/payload/env-bridge/"
[ -f env-bridge/package-lock.json ] && cp env-bridge/package-lock.json "$OUT/payload/env-bridge/"

cp training/*.py training/*.sh "$OUT/payload/training/" 2>/dev/null || true
cp -r training/envs training/models "$OUT/payload/training/"
[ -f training/requirements.txt ] && cp training/requirements.txt "$OUT/payload/training/"
find "$OUT/payload" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# The notebook invokes this from inside the Dataset.
cp kaggle/setup_kaggle.sh "$OUT/payload/"

# .ntfy_topic is deliberately EXCLUDED. Anyone holding the topic string can
# read (and publish to) that ntfy channel, and a Kaggle Dataset is one
# checkbox away from public. send_ntfy() no-ops without it, which only costs
# push notifications. Opt in explicitly if you accept that:
#   INCLUDE_NTFY=1 ./package_source.sh
if [ "${INCLUDE_NTFY:-0}" = "1" ] && [ -f training/.ntfy_topic ]; then
  cp training/.ntfy_topic "$OUT/payload/training/"
  echo "WARNING: bundled .ntfy_topic -- keep this Dataset PRIVATE"
fi

# --- checkpoint so a session resumes instead of restarting from scratch -----
mkdir -p "$OUT/payload/checkpoint"
if [ -f training/checkpoints/latest.pt ]; then
  cp training/checkpoints/latest.pt "$OUT/payload/checkpoint/"
  [ -f training/checkpoints/notified_milestones.json ] && \
    cp training/checkpoints/notified_milestones.json "$OUT/payload/checkpoint/"
  # Carried so a resumed session appends to one continuous history rather
  # than starting a fresh log every 12h. The analysis scripts under
  # analysis/*/territory_moving_average/ parse these directly.
  [ -f training/checkpoints/train_log.csv ] && \
    cp training/checkpoints/train_log.csv "$OUT/payload/checkpoint/"
  [ -f training/checkpoints/train_stdout.log ] && \
    cp training/checkpoints/train_stdout.log "$OUT/payload/checkpoint/"
  echo "included checkpoint: $(du -h training/checkpoints/latest.pt | cut -f1)"
else
  echo "NOTE: no latest.pt -- the Kaggle run will start from scratch at update 0"
fi

# --- metadata for `kaggle datasets create/version` ---------------------------
cat > "$OUT/payload/dataset-metadata.json" <<'JSON'
{
  "title": "openfront-rl source and checkpoint",
  "id": "REPLACE_USERNAME/openfront-rl-src",
  "licenses": [{"name": "other"}]
}
JSON

cat > "$OUT/payload/PINNED_SHA" <<EOF
$(cd "$REPO_ROOT/OpenFrontIO" && git rev-parse HEAD)
EOF

echo
echo "payload built at: $OUT/payload"
du -sh "$OUT/payload"
echo
echo "next:"
echo "  1. edit $OUT/payload/dataset-metadata.json -> replace REPLACE_USERNAME"
echo "  2. kaggle datasets create -p $OUT/payload      (first time)"
echo "     kaggle datasets version -p $OUT/payload -m 'update'   (subsequent)"
