"""Plots AGENT tile count (y) vs. game tick (x) as one line per evaluation
episode, all overlaid on the same axes. Lines are colored red, with alpha
ramping from ~1% (earliest eval, by chronological replay filename) to 100%
(latest eval) -- so the plot visually shows the training-progression trend
as a gradient even though every episode is drawn.

Reads per-episode series CSVs from series/ (see run_all.sh /
TerritoryTimeSeries.ts to generate them) and writes two files in this
directory:
  - territory_progression.png: full tick range. Heavily left-compressed --
    a handful of very long episodes (one reaches ~93.5k ticks) stretch the
    x-axis far past where most episodes resolve.
  - territory_progression_zoom.png: same data, x-axis capped to the first
    10% of the longest episode's tick count, where nearly all of the
    across-episode variation actually happens.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SERIES_DIR = HERE / "series"
OUT_PATH = HERE / "territory_progression.png"
OUT_PATH_ZOOM = HERE / "territory_progression_zoom.png"

MIN_ALPHA = 0.01
MAX_ALPHA = 1.0
ZOOM_FRACTION = 0.10


def load_series(files: list[Path]) -> list[tuple[list[int], list[int]]]:
    series = []
    for f in files:
        rows = list(csv.DictReader(open(f)))
        if not rows:
            series.append(([], []))
            continue
        ticks = [int(r["tick"]) for r in rows]
        tiles = [int(r["tiles"]) for r in rows]
        series.append((ticks, tiles))
    return series


def plot(files: list[Path], series: list[tuple[list[int], list[int]]], out_path: Path, xlim: float | None) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    n = len(files)
    for i, (ticks, tiles) in enumerate(series):
        if not ticks:
            continue
        # earliest eval (i=0) -> MIN_ALPHA, latest eval (i=n-1) -> MAX_ALPHA
        frac = i / (n - 1) if n > 1 else 1.0
        alpha = MIN_ALPHA + frac * (MAX_ALPHA - MIN_ALPHA)
        ax.plot(ticks, tiles, color="red", alpha=alpha, linewidth=1)

    ax.set_xlabel("game tick")
    ax.set_ylabel("AGENT tiles owned")
    zoom_note = f" (first {ZOOM_FRACTION:.0%} of longest episode)" if xlim else ""
    ax.set_title(
        f"Territorial extent vs. game tick, across {n} evaluation episodes{zoom_note}\n"
        "(each line = one eval episode; lighter red = earlier training, darker red = later training)"
    )
    ax.grid(alpha=0.3)
    if xlim:
        ax.set_xlim(0, xlim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path} ({n} episodes plotted)")


def main() -> None:
    files = sorted(SERIES_DIR.glob("*.csv"))  # filenames are timestamp-prefixed -> chronological
    if not files:
        raise SystemExit(f"No series CSVs found in {SERIES_DIR} -- run run_all.sh first.")

    series = load_series(files)
    plot(files, series, OUT_PATH, xlim=None)

    max_tick = max((ticks[-1] for ticks, _ in series if ticks), default=0)
    plot(files, series, OUT_PATH_ZOOM, xlim=max_tick * ZOOM_FRACTION)


if __name__ == "__main__":
    main()
