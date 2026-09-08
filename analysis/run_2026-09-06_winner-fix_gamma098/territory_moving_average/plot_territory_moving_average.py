"""Plots a moving average of AGENT's territorial share against evaluation
run number. Reads territory_moving_average.csv (see build_dataset.py) and
writes territory_moving_average.png in the same directory.

Per-eval territory is extremely noisy here -- each eval draws a random map
from SMALL_MAPS, and those range from 210k to 1.1M land tiles with wildly
different geography -- so the raw series is nearly unreadable and a moving
average is what actually shows whether the agent is improving. Raw points
are kept in the background at low alpha so the noise level stays visible
rather than being hidden by the smoothing.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "territory_moving_average.csv"
OUT_PATH = HERE / "territory_moving_average.png"

WINDOW = 15


def moving_average(xs: list[float], window: int) -> tuple[list[int], list[float]]:
    """Trailing mean; index i averages the `window` values ending at i, so a
    point is only emitted once a full window exists (no partial-window edge
    artifacts that would exaggerate an early trend)."""
    idx, out = [], []
    for i in range(window - 1, len(xs)):
        out.append(sum(xs[i - window + 1 : i + 1]) / window)
        idx.append(i)
    return idx, out


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    ev = [int(r["eval_index"]) for r in rows]
    frac = [float(r["peak_territory_frac"]) for r in rows]
    gamma = [float(r["gamma"]) for r in rows]
    wins = [i for i, r in enumerate(rows) if r["winner"] == "AGENT"]

    switch = next((i for i, g in enumerate(gamma) if g == 0.98), None)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.scatter(ev, frac, s=14, alpha=0.28, color="#94a3b8",
               label="individual eval (raw)", zorder=2)

    mi, ma = moving_average(frac, WINDOW)
    ax.plot(mi, ma, lw=2.4, color="#2563eb",
            label=f"moving average (window={WINDOW})", zorder=4)

    if switch is not None:
        ax.axvline(switch - 0.5, color="#dc2626", ls="--", lw=1.6, zorder=3)
        ax.annotate(
            "gamma 0.999 -> 0.98\nmax_steps 20000 -> 2500\n(update 710)",
            xy=(switch - 0.5, ax.get_ylim()[1]),
            xytext=(switch + 2, 0.72),
            color="#dc2626", fontsize=9, ha="left", va="top",
        )

    if wins:
        ax.scatter(wins, [frac[i] for i in wins], s=90, marker="*",
                   color="#16a34a", edgecolor="white", linewidth=0.6,
                   label=f"AGENT win ({len(wins)})", zorder=5)

    # The engine ends a game as soon as someone owns 80% of the land
    # (percentageTilesOwnedToWin, FFA) -- so this line is the win condition,
    # not an arbitrary reference.
    ax.axhline(0.80, color="#16a34a", ls=":", lw=1.4, alpha=0.8, zorder=1)
    ax.text(0.5, 0.815, "80% = domination win threshold", fontsize=8.5,
            color="#16a34a")

    ax.set_xlabel("evaluation run (chronological; one eval every 10 updates)")
    ax.set_ylabel("AGENT share of map land tiles (peak in episode)")
    ax.set_title(
        "Territorial control vs. evaluation run\n"
        "run_2026-09-06_winner-fix_gamma098 -- greedy/argmax eval, random map per episode"
    )
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")

    if switch is not None:
        pre = frac[:switch]
        post = frac[switch:]
        print(f"  gamma=0.999: n={len(pre):3d} mean={sum(pre)/len(pre):.4f}")
        print(f"  gamma=0.98 : n={len(post):3d} mean={sum(post)/len(post):.4f}")


if __name__ == "__main__":
    main()
