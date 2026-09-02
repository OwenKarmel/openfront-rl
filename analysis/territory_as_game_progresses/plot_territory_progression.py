"""Plots AGENT tile count (y) vs. game tick (x) as one line per evaluation
episode, all overlaid on the same axes. Lines are colored red, with alpha
ramping from ~1% (earliest eval, by chronological replay filename) to 100%
(latest eval) -- so the plot visually shows the training-progression trend
as a gradient even though every episode is drawn.

Reads per-episode series CSVs from series/ (see run_all.sh /
TerritoryTimeSeries.ts to generate them) and writes
territory_progression.png in this directory.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SERIES_DIR = HERE / "series"
OUT_PATH = HERE / "territory_progression.png"

MIN_ALPHA = 0.01
MAX_ALPHA = 1.0


def main() -> None:
    files = sorted(SERIES_DIR.glob("*.csv"))  # filenames are timestamp-prefixed -> chronological
    if not files:
        raise SystemExit(f"No series CSVs found in {SERIES_DIR} -- run run_all.sh first.")

    fig, ax = plt.subplots(figsize=(13, 7))

    n = len(files)
    for i, f in enumerate(files):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        ticks = [int(r["tick"]) for r in rows]
        tiles = [int(r["tiles"]) for r in rows]
        # earliest eval (i=0) -> MIN_ALPHA, latest eval (i=n-1) -> MAX_ALPHA
        frac = i / (n - 1) if n > 1 else 1.0
        alpha = MIN_ALPHA + frac * (MAX_ALPHA - MIN_ALPHA)
        ax.plot(ticks, tiles, color="red", alpha=alpha, linewidth=1)

    ax.set_xlabel("game tick")
    ax.set_ylabel("AGENT tiles owned")
    ax.set_title(
        f"Territorial extent vs. game tick, across {n} evaluation episodes\n"
        "(each line = one eval episode; lighter red = earlier training, darker red = later training)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH} ({n} episodes plotted)")


if __name__ == "__main__":
    main()
