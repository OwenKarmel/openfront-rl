# Territory as the game progresses

Answers a different question than `../territorial_extent/`: not "what was
the single best replay," but "for every eval episode we have, how does
AGENT's tile count evolve tick-by-tick over the course of the match, and
has that shape changed as training has progressed?" Produced in response to
"create a plot with game ticks on the x axis and territorial extent on the
y axis, plot every evaluation run as a line, earlier runs lighter red,
later runs darker red."

## Files

- **`TerritoryTimeSeries.ts`** — like `../territorial_extent/FindMaxTerritory.ts`,
  replays a single `training/replays/*.json` `GameRecord` through the real
  OpenFrontIO core simulation (same construction `npm run replay:game`
  uses), but instead of reporting only peak/final tile counts, writes a
  full per-tick `tick,tiles` CSV. Same one-file-per-process discipline and
  same reason (`loadTerrainMap()`'s module-level map cache). Run via
  `run_all.sh`, not directly — see the note at the top of the file for the
  copy-into-`OpenFrontIO/tests/replay/` requirement (Node's bare-specifier
  resolution needs it there to find `node_modules`).

- **`run_all.sh`** — copies `TerritoryTimeSeries.ts` into
  `OpenFrontIO/tests/replay/` and runs it once per file in
  `training/replays/`, writing one CSV per replay into `series/`. Rerun-safe
  (skips files that already have an output CSV), so it can be re-run later
  to pick up new replays without redoing the existing 116.

- **`series/`** — one CSV per replay episode (`tick,tiles`), named after the
  replay file. Not deduplicated/pruned — every replay in
  `training/replays/` as of generation time.

- **`plot_territory_progression.py`** — reads every CSV in `series/`, sorts
  them chronologically by filename (replay filenames are timestamp-prefixed,
  so this is also training order), and plots each as its own line (all on
  one axes): x = game tick, y = AGENT tiles owned. All lines are red; alpha
  ramps linearly from 0.01 (earliest episode) to 1.0 (latest), so the plot
  reads as a density/progression gradient even with 100+ overlapping lines.
  Writes `territory_progression.png`.

## Reproducing / updating

1. Regenerate/accumulate replays (`training/replays/*.json`, via
   `train.py`'s `--eval-every`, or a one-off eval run).
2. `bash run_all.sh` (only processes replays that don't already have a CSV
   in `series/`).
3. `python plot_territory_progression.py` (needs `matplotlib`; the
   `openfront-rl` conda env has it installed).

## Observations (as of 116 episodes, ~update 1123)

- Most episodes are heavily left-compressed: the x-axis top end is set by a
  handful of very long games (one reaches ~93.5k ticks), while the bulk of
  episodes resolve (win, loss, or truncate) well under 10k ticks — visually
  most of the interesting variation is packed into the first ~10-15% of the
  plot width. A log-scaled or tick-capped x-axis would spread that region
  out better if closer inspection of early-game shape is needed; not done
  here since the request was for a straightforward tick-vs-tiles plot.
- Many lines sit flat at 0 for their entire length — episodes where AGENT
  never expanded past nothing (consistent with the type-head-plateau
  finding in `../territorial_extent/README.md`: long stretches where the
  policy is indifferent between `noop`/`expand`).
- The handful of lines reaching 70k+ tiles show a common shape: fast climb
  in the first few thousand ticks, then a long flat plateau (holding
  territory, not still expanding), then in some cases a late decline
  (losing ground) rather than a hold-to-the-end. Both light-red (early
  training) and dark-red (later training) lines reach the high plateau, so
  this isn't purely a "recent checkpoints only" pattern -- consistent with
  the bursty/noisy (not steadily improving) picture from the peak-vs-update
  scatter in `../territorial_extent/`.
