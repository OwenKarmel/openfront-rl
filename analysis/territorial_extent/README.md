# Territorial extent analysis

Answers: across all dumped eval-episode replays so far, how much territory did
AGENT ever actually hold, and how has that changed over training? Produced in
response to "do a search of the replay that achieved the greatest territorial
extent so far" + "plot greatest territorial extent versus evaluation run."

## Files

- **`FindMaxTerritory.ts`** — replays a single `training/replays/*.json`
  `GameRecord` through the real OpenFrontIO core simulation (same
  construction `npm run replay:game` uses — real `Config`,
  `createNationsForGame`/`createGame`, real `Executor`) and reports the
  greatest tile count AGENT reached at any point in the episode, not just at
  the end. **Run one file per process invocation** — see the note at the top
  of the file for why (a module-level map cache in `loadTerrainMap()` bit an
  earlier in-process-loop version, silently corrupting every file after the
  first). To actually run it: copy into `OpenFrontIO/tests/replay/` (its
  import paths match `ReplayGame.ts`, which lives there) and run from
  `OpenFrontIO/`:
  ```bash
  npx tsx tests/replay/FindMaxTerritory.ts ../training/replays/<file>.json
  # prints: RESULT <peakTiles> <finalTiles> <ticks> <ALIVE|DEAD> <filename>
  ```
  Loop over every file in `training/replays/`, collect the `RESULT` lines
  into a text file (one per line).

- **`build_dataset.py`** — takes that collected results file and
  `training/checkpoints/train_stdout.log`, and merges them into
  `territorial_extent.csv`. The approximate training update for each eval is
  recovered by walking `train_stdout.log` in order and remembering the last
  `update N ...` line seen before each `eval vs ...` line — replay filenames
  only encode a wall-clock timestamp + gameID hash (see `EnvServer.ts`'s
  `flushRecord()`), not the update number, so this log-order correlation is
  the only way to recover it. **Caveat**: eval-log-entry count and
  replay-file count were off by one when this was built (110 vs 111) —
  pairs the first N chronologically and drops the rest rather than risk
  silent misalignment. Good enough to see the real trend, not
  update-number-exact.

- **`plot_territorial_extent.py`** — reads `territorial_extent.csv`, writes
  `territorial_extent.png` (peak tiles and final tiles per eval episode,
  against approximate training update).

- **`territorial_extent.csv`** — the merged dataset (all eval episodes
  available as of this analysis, not just the top 10).

- **`territorial_extent.png`** — the chart.

## Headline finding

Best so far: **`2026-09-02T16-05-13-350_265ffbf9.json`, peak 74,674 tiles**
(final 41,995, still alive at truncation, ~899 updates in) — verified via
`npm run replay:game` (IN SYNC, real numbers). Full top-10 and methodology
notes were also written to conversation history at the time.

The chart shows a noisy, bursty pattern — occasional spikes into the
30k-75k tile range interspersed with long stretches stuck at ~52 tiles (the
spawn footprint, never expanded) — rather than a steady upward trend. This
matches the ongoing type-head-plateau diagnosis from the same session
(`training/train.py`'s per-update `type_probs` logging): the greedy/argmax
policy intermittently discovers real expansion but hasn't reliably learned
to sustain it yet.

## Reproducing

1. Regenerate replays (`training/replays/*.json` — via `train.py`'s
   `--eval-every`, or a one-off eval run).
2. Copy `FindMaxTerritory.ts` into `OpenFrontIO/tests/replay/`, loop it over
   every replay file, save `RESULT` lines to a text file.
3. `python build_dataset.py <results-file>`
4. `python plot_territorial_extent.py`
