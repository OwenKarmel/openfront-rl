"""Plots training win rate against training update, with the greedy-eval win
rate on the same axis for comparison. Reads training_win_rate.csv (see
build_dataset.py) plus the sibling territory analysis' CSV for the eval
outcomes, and writes training_win_rate.png in the same directory.

Same visual construction as the territory_moving_average plots in this
analysis tree -- recessive raw points, a single bold trailing mean, a dotted
reference line at the goalpost, identical axis limits -- so the plots in this
run's directory read as one family.

Both series are win rates on one 0-1 scale, so they share a y axis; this is
NOT a dual-axis chart. Colors are the family's #2563eb / #16a34a pair, which
passes all six checks of the dataviz validator (worst adjacent CVD dE 30.3
deutan). The raw-point grey is deliberately below the chroma floor: it is
recessive background noise, not a categorical series.

Window choice: the training mean is a trailing 50-EPISODE window, not a
window over updates, because only ~0.18 episodes finish per update and an
update-based window would mostly average empty rows. At this run's ~14% base
rate a 50-episode window carries about +/-5pp of binomial noise, which is the
resolution the line should be read at -- wiggles smaller than that are not
signal. The eval series uses a trailing 12, matching CurriculumScheduler's
own window so the plotted line is the quantity the promotion gate actually
tests.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "training_win_rate.csv"
EVAL_CSV = HERE.parent / "territory_moving_average" / "territory_moving_average.csv"
OUT_PATH = HERE / "training_win_rate.png"

TRAIN_WINDOW = 50   # episodes
EVAL_WINDOW = 12    # episodes -- CurriculumScheduler.window
PROMOTE_THRESHOLD = 0.75  # CurriculumScheduler.promote_threshold


def trailing_mean(xs: list[float], window: int) -> tuple[list[int], list[float]]:
    """Index i averages the `window` values ending at i; emitted only once a
    full window exists, so there are no partial-window edge artifacts."""
    idx, out = [], []
    for i in range(window - 1, len(xs)):
        out.append(sum(xs[i - window + 1 : i + 1]) / window)
        idx.append(i)
    return idx, out


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    updates = [int(r["update"]) for r in rows]
    won = [int(r["won"]) for r in rows]
    overall = sum(won) / len(won)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Raw per-episode outcomes, jittered vertically so the 0/1 bands show
    # density rather than overprinting into two solid lines.
    import random
    random.seed(0)
    ax.scatter(updates, [w + random.uniform(-0.022, 0.022) for w in won],
               s=10, alpha=0.18, color="#94a3b8",
               label="individual training episode (raw, jittered)", zorder=2)

    ti, tm = trailing_mean([float(w) for w in won], TRAIN_WINDOW)
    ax.plot([updates[i] for i in ti], tm, lw=2.4, color="#2563eb",
            label=f"training win rate (trailing {TRAIN_WINDOW} episodes)", zorder=4)

    # Greedy/argmax eval outcomes, from the sibling territory analysis so
    # there is one source of truth for eval results per run.
    if EVAL_CSV.exists():
        ev = list(csv.DictReader(EVAL_CSV.open()))
        ev_upd = [int(r["update"]) for r in ev]
        ev_won = [1.0 if r["winner"] == "AGENT" else 0.0 for r in ev]
        ei, em = trailing_mean(ev_won, EVAL_WINDOW)
        if em:
            ax.plot([ev_upd[i] for i in ei], em, lw=2.2, color="#16a34a",
                    marker="o", markersize=4, markevery=3,
                    label=f"greedy-eval win rate (trailing {EVAL_WINDOW} = curriculum window)",
                    zorder=5)
    else:
        print(f"note: {EVAL_CSV} missing -- eval series omitted")

    ax.axhline(PROMOTE_THRESHOLD, color="#16a34a", ls=":", lw=1.4, alpha=0.8, zorder=1)
    ax.text(updates[0] + 20, PROMOTE_THRESHOLD + 0.015,
            "0.75 = curriculum promotion threshold (gates on the EVAL rate)",
            fontsize=8.5, color="#16a34a")

    ax.set_xlabel("training update")
    ax.set_ylabel("win rate vs Nation AI (difficulty: easy)")
    ax.set_title(
        "Training vs. greedy-eval win rate over training\n"
        f"run_2026-09-11_bbf-adamw -- gamma=0.98, rollout=50, epochs=8, AdamW wd=0.1  "
        f"(overall training win rate {overall:.1%}, n={len(won)} episodes)"
    )
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    # Centre-left, not upper-left: the raw wins sit at y=1.0 and an
    # upper-left legend hides the early ones. The 0.30-0.70 band is empty.
    ax.legend(loc="center left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")

    print(f"  training: n={len(won)} episodes  overall={overall:.4f}")
    if tm:
        print(f"  training trailing mean: first={tm[0]:.4f} last={tm[-1]:.4f} "
              f"min={min(tm):.4f} max={max(tm):.4f}")
        h = len(won) // 2
        print(f"  first half={sum(won[:h])/h:.4f}  second half={sum(won[h:])/(len(won)-h):.4f}")


if __name__ == "__main__":
    main()
