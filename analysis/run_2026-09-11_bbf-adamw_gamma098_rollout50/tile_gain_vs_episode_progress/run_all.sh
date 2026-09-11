#!/usr/bin/env bash
# Generates a per-tick tile-count time series CSV for every eval replay of
# this run, by copying TerritoryTimeSeries.ts into OpenFrontIO/tests/replay/
# (see the note at the top of that file for why) and invoking it once per
# file -- one process per file, because loadTerrainMap()'s module-level
# cache corrupts state if reused across files in one process.
#
# TerritoryTimeSeries.ts is reused verbatim from
# ../../run_2026-09-02_pre-naval_num-envs8/territory_as_game_progresses/.
#
# nice'd because a training run is normally live while this is generating,
# and re-simulation is CPU-bound on the same cores the Node envs use.
#
# Run from anywhere; paths below are relative to this script's location.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
OPENFRONTIO_DIR="$REPO_ROOT/OpenFrontIO"
REPLAYS_DIR="$REPO_ROOT/training/replays"
SRC_TS="$REPO_ROOT/analysis/run_2026-09-02_pre-naval_num-envs8/territory_as_game_progresses/TerritoryTimeSeries.ts"
OUT_DIR="$HERE/series"

mkdir -p "$OUT_DIR"
cp "$SRC_TS" "$OPENFRONTIO_DIR/tests/replay/TerritoryTimeSeries.ts"

cd "$OPENFRONTIO_DIR"
n=0
for f in "$REPLAYS_DIR"/*.json; do
  base="$(basename "$f" .json)"
  out="$OUT_DIR/$base.csv"
  if [ -f "$out" ]; then
    continue # already generated, skip (rerun-safe)
  fi
  nice -n 10 npx tsx tests/replay/TerritoryTimeSeries.ts "$f" "$out" >/dev/null 2>&1 \
    || { echo "FAILED: $base" >&2; continue; }
  n=$((n + 1))
  printf '.'
done
echo
echo "Done. Generated $n new series CSVs in $OUT_DIR"

# Leave the OpenFrontIO working tree as we found it.
rm -f "$OPENFRONTIO_DIR/tests/replay/TerritoryTimeSeries.ts"
