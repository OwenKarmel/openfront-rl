#!/usr/bin/env bash
# Generates a per-tick tile-count time series CSV for every replay in
# training/replays/, by copying TerritoryTimeSeries.ts into
# OpenFrontIO/tests/replay/ (see the note at the top of that file for why)
# and invoking it once per file (one process per file -- loadTerrainMap()'s
# module-level cache corrupts state if reused across files in one process).
#
# Run from anywhere; paths below are relative to this script's location.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
OPENFRONTIO_DIR="$REPO_ROOT/OpenFrontIO"
REPLAYS_DIR="$REPO_ROOT/training/replays"
OUT_DIR="$HERE/series"

mkdir -p "$OUT_DIR"
cp "$HERE/TerritoryTimeSeries.ts" "$OPENFRONTIO_DIR/tests/replay/TerritoryTimeSeries.ts"

cd "$OPENFRONTIO_DIR"
n=0
for f in "$REPLAYS_DIR"/*.json; do
  base="$(basename "$f" .json)"
  out="$OUT_DIR/$base.csv"
  if [ -f "$out" ]; then
    continue # already generated, skip (rerun-safe)
  fi
  echo "Processing $base..."
  npx tsx tests/replay/TerritoryTimeSeries.ts "$f" "$out"
  n=$((n + 1))
done
echo "Done. Generated $n new series CSVs in $OUT_DIR"
