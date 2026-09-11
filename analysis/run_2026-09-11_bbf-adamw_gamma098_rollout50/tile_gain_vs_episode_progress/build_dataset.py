"""Turns the per-tick tile-count series in series/ into one row per rollout
window and writes tile_gain.csv next to this file.

A "rollout window" is 500 ticks: --rollout-length 50 decision steps x
--ticks-per-step 10. That is deliberately the same slice of game time the
agent's PPO update sees, so a point here is "how many tiles did the agent
net over one rollout's worth of play, and how far into the episode was it".

Episode -> training-time mapping: eval replays are written one per eval, in
order, so sorting the replay JSONs by mtime gives eval index, which
territory_moving_average.csv maps to a training update. The mapping is
checked, not assumed -- each series' row count must equal the `ticks` the
eval line recorded for that index, and a mismatch is reported rather than
silently producing a mislabelled plot.

Tile counts are RAW, not normalised by map size. Each eval draws a random
map from SMALL_MAPS spanning 210k-1.1M land tiles, so a gain of N tiles is
not equally impressive on every map -- see the README's caveat. Raw is what
was asked for and is what the agent actually gained; the normalising factor
(total land tiles) is not recorded per episode in the series CSVs.
"""

import csv
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES_DIR = HERE / "series"
REPLAYS_DIR = Path("/home/developer/openfront-rl/training/replays")
EVAL_CSV = HERE.parent / "territory_moving_average" / "territory_moving_average.csv"
OUT = HERE / "tile_gain.csv"

WINDOW_TICKS = 500  # rollout_length 50 x ticks_per_step 10


def main() -> None:
    series = sorted(
        SERIES_DIR.glob("*.csv"),
        key=lambda p: os.path.getmtime(REPLAYS_DIR / f"{p.stem}.json"),
    )
    evals = list(csv.DictReader(EVAL_CSV.open()))
    if len(series) != len(evals):
        print(f"WARNING: {len(series)} series but {len(evals)} eval rows -- "
              "mapping may be off; using the shorter of the two")

    rows = []
    mismatches = []
    for idx, path in enumerate(series):
        if idx >= len(evals):
            break
        ticks = [(int(t), int(n)) for t, n in
                 list(csv.reader(path.open()))[1:]]
        if not ticks:
            continue
        expected = int(evals[idx]["ticks"])
        if abs(len(ticks) - expected) > 1:
            mismatches.append((idx, path.stem, len(ticks), expected))
        update = int(evals[idx]["update"])

        by_tick = {t: n for t, n in ticks}
        last_tick = ticks[-1][0]
        for start in range(1, last_tick - WINDOW_TICKS + 1, WINDOW_TICKS):
            end = start + WINDOW_TICKS
            if start not in by_tick or end not in by_tick:
                continue
            rows.append({
                "eval_index": idx,
                "update": update,
                "episode": path.stem,
                "window_start_tick": start,
                "window_mid_tick": start + WINDOW_TICKS // 2,
                "tiles_at_start": by_tick[start],
                "tile_gain": by_tick[end] - by_tick[start],
                "episode_total_ticks": last_tick,
            })

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "eval_index", "update", "episode", "window_start_tick",
            "window_mid_tick", "tiles_at_start", "tile_gain", "episode_total_ticks",
        ])
        w.writeheader()
        w.writerows(rows)

    gains = [r["tile_gain"] for r in rows]
    print(f"wrote {len(rows)} rollout-window rows from {len(series)} episodes -> {OUT}")
    print(f"  update range {rows[0]['update']} -> {rows[-1]['update']}")
    print(f"  tile_gain: min={min(gains)} max={max(gains)} mean={sum(gains)/len(gains):.1f}")
    print(f"  negative-gain windows: {sum(1 for g in gains if g < 0)}/{len(gains)}")
    if mismatches:
        print(f"  WARNING: {len(mismatches)} series/eval tick mismatches (mapping suspect):")
        for m in mismatches[:5]:
            print(f"    idx={m[0]} {m[1]}: series={m[2]} ticks, eval line said {m[3]}")
    else:
        print("  tick counts match the eval log for all episodes (mapping verified)")


if __name__ == "__main__":
    main()
