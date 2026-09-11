# Tile gain per rollout vs. position in episode

For every 500-tick window of every eval episode, how many tiles the agent
netted, plotted against how far into the episode that window fell, coloured
by when in training the episode happened (light = early, dark = late).

500 ticks = `--rollout-length 50` x `--ticks-per-step 10`, i.e. exactly the
slice of game time one PPO update sees.

## Files

- **`run_all.sh`** — re-simulates every replay in `training/replays/` through
  the real OpenFrontIO core and writes a per-tick `tick,tiles` series into
  `series/`. Reuses `TerritoryTimeSeries.ts` verbatim from
  `../../run_2026-09-02_pre-naval_num-envs8/territory_as_game_progresses/`.
  One process per file (`loadTerrainMap()` caches a mutable map at module
  scope), `nice`d because training is normally live. Rerun-safe. ~2 minutes
  for 40 episodes.
- **`build_dataset.py`** — collapses those series into one row per rollout
  window, writes `tile_gain.csv`.
- **`plot_tile_gain.py`** — writes `tile_gain.png`.

Regenerate with:

```bash
./run_all.sh && python build_dataset.py && python plot_tile_gain.py
```

Episode -> training-update mapping is **verified, not assumed**: each
series' tick count must equal the `ticks` its eval log line recorded.
All 40 matched.

## The finding: the agent freezes partway through every episode

**84.5% of all rollout windows (1728/2045) gain exactly zero tiles.** Not
"a small gain" -- literally no change in tile count. The percentiles from p5
through p75 are all 0.

Broken down by how far into the episode the window falls:

| ticks into episode | windows | % gaining exactly 0 | mean tile gain |
|---|---|---|---|
| 0 – 5,000 | 398 | 46.7% | **+1885.1** |
| 5,000 – 15,000 | 553 | 81.2% | −431.6 |
| 15,000 – 40,000 | 571 | **99.8%** | −0.1 |
| 40,000 – end | 523 | **100.0%** | 0.0 |

The agent does essentially all of its territorial work in the **first ~5,000
ticks (~8 minutes of game time)**. From ~15,000 ticks onward it is frozen:
across 523 windows beyond 40,000 ticks, spanning tens of thousands of ticks
of play, *not a single tile changes hands*. Episodes nonetheless run to a
mean of 25,790 ticks and a maximum of 102,011.

This is the mechanism behind the other two analyses in this directory:

- it explains why eval episodes average ~25k ticks (vs ~17k across the
  earlier runs' replays) -- the game grinds on long after the agent stops
  playing;
- it explains the flat territory moving average -- peak territory is set in
  the first few minutes and never revisited;
- it is consistent with greedy eval going 0/40 while the sampled policy wins
  ~14-28% of training episodes.

## Training is improving this, but slowly

Split at the median update, activity is clearly rising:

| training half | windows | % gaining exactly 0 | mean tile gain |
|---|---|---|---|
| early (update <= 1319) | 1189 | 87.1% | +90.3 |
| late (update > 1319) | 856 | 80.8% | **+472.2** |

Later episodes freeze less often and gain ~5x more per active window, which
lines up with the rising training win rate in `../training_win_rate/`. The
freeze is being unlearned -- just not fast enough to move greedy eval off
zero wins yet.

## This is a known pathology, and the shipped fix has not resolved it

The roadmap's §1.3 motivated `self_troops_ratio` from exactly this
observation: *"a replay where AGENT sat at `troops/maxTroops == 1.000` for
~100k ticks with 200k+ tiles of unclaimed neutral land sitting right
there"*, the reasoning being that `troopIncreaseRate()`'s
`(1 - troops/maxTroops)` factor stops troop regrowth at the cap, and a raw
troop count cannot express "capped, waiting is wasted".

That observation was a single replay. **This plot is the quantified version
across 40 episodes, and it shows the pathology is still present** even
though `self_troops_ratio` and the per-opponent `troops_ratio` have shipped
(README "Upcoming Architecture": Phase 1 is built). So the capped-army
signal being *observable* has not been sufficient to make the agent act on
it. Whatever is causing the freeze, it is not purely an observability gap.

## Caveat: tile counts are raw, not normalised

Each eval draws a random map from `SMALL_MAPS`, spanning 210k-1.1M land
tiles, so +1,000 tiles is not equally impressive on every map, and the
vertical spread of the non-zero points partly reflects which map was drawn
rather than how well the agent played. The series CSVs carry only
`tick,tiles`, not the map's land-tile total, so normalising would require
re-deriving it per episode.

This does **not** weaken the headline: a window gaining *exactly zero* tiles
is map-independent, and that is 84.5% of them.

## Reading the plot

Y is **symlog** (linear within ±100, logarithmic outside). A linear axis
would collapse everything onto the zero line, since the modal outcome is 0
while the extremes run −91,473 to +75,870. Colour is a sequential
single-hue ramp (matplotlib `Blues` truncated to 0.25–1.0 so the lightest
step still reads against white), with a colourbar rather than a categorical
legend, because training update is an ordered magnitude and not an identity.
