"""Plots a moving average of AGENT's territorial share against evaluation
run number. Reads territory_moving_average.csv (see build_dataset.py) and
writes territory_moving_average.png in the same directory.

Deliberately the same construction as the gamma098 and gamma0995 runs'
versions of this plot -- same metric, same window, same axes limits, same 80%
reference line, same colors -- so all three are directly comparable by eye.
That comparability is the whole point of the plot family, so the palette is
held fixed rather than restyled.

The x axis is directly comparable across the three runs even though this run
evaluates every 60 updates and the others every 10: eval cadence was chosen
to hold *experience between evals* constant at 3000 decision-steps per env
(60 x rollout 50 here, 10 x rollout 300 there). Eval index therefore means
the same amount of gameplay in all three.

Per-eval territory is extremely noisy here: each eval draws a random map from
SMALL_MAPS, spanning 210k to 1.1M land tiles with very different geography,
so the raw series is nearly unreadable on its own. Raw points are kept in the
background at low alpha so the noise level stays visible rather than being
hidden by the smoothing.

Palette note: the two real categorical series (#2563eb moving average,
#16a34a AGENT win) pass all six checks of the dataviz validator, worst
adjacent CVD dE 30.3. The scatter grey (#94a3b8) is intentionally below the
chroma floor -- it is recessive background noise, not a categorical series --
and the 80% rule is a labelled reference line, not a series.
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
    updates = [int(r["update"]) for r in rows]
    wins = [i for i, r in enumerate(rows) if r["winner"] == "AGENT"]

    seg = [int(r["segment"]) for r in rows]
    resume = next((i for i in range(1, len(seg)) if seg[i] != seg[i - 1]), None)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.scatter(ev, frac, s=14, alpha=0.28, color="#94a3b8",
               label="individual eval (raw)", zorder=2)

    mi, ma = moving_average(frac, WINDOW)
    ax.plot(mi, ma, lw=2.4, color="#2563eb",
            label=f"moving average (window={WINDOW})", zorder=4)

    if resume is not None:
        ax.axvline(resume - 0.5, color="#f59e0b", ls="--", lw=1.6, zorder=3)
        ax.annotate(
            f"crash + resume\n(update {updates[resume - 1]} -> {updates[resume]})",
            xy=(resume - 0.5, 0.72), xytext=(resume + 1.5, 0.72),
            color="#b45309", fontsize=9, ha="left", va="top",
        )

    if wins:
        ax.scatter(wins, [frac[i] for i in wins], s=90, marker="*",
                   color="#16a34a", edgecolor="white", linewidth=0.6,
                   label=f"AGENT win ({len(wins)})", zorder=5)

    # The engine ends a game as soon as someone owns 80% of the land
    # (percentageTilesOwnedToWin, FFA) -- the win condition, not an
    # arbitrary reference.
    ax.axhline(0.80, color="#16a34a", ls=":", lw=1.4, alpha=0.8, zorder=1)
    ax.text(0.5, 0.815, "80% = domination win threshold", fontsize=8.5,
            color="#16a34a")

    ax.set_xlabel(
        "evaluation run (chronological; one eval every 60 updates "
        "= 3000 steps/env, the same experience-per-eval as the gamma098/gamma0995 runs)"
    )
    ax.set_ylabel("AGENT share of map land tiles (peak in episode)")
    ax.set_title(
        "Territorial control vs. evaluation run\n"
        "run_2026-09-11_bbf-adamw -- gamma=0.98, rollout=50, epochs=8, AdamW wd=0.1, "
        "greedy/argmax eval, random map per episode"
    )
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")

    print(f"  n={len(frac)}  mean={sum(frac)/len(frac):.4f}  max={max(frac):.4f}  wins={len(wins)}")
    if len(ma) >= 1:
        print(f"  moving avg: first={ma[0]:.4f}  last={ma[-1]:.4f}  peak={max(ma):.4f}")


if __name__ == "__main__":
    main()
