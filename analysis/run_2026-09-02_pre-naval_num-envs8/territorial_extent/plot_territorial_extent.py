"""Plots peak (and final) AGENT tile count per evaluation episode against
the approximate training update that produced it. Reads
territorial_extent.csv (see build_dataset.py) and writes
territorial_extent.png in the same directory.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "territorial_extent.csv"
OUT_PATH = HERE / "territorial_extent.png"


def main() -> None:
    rows = list(csv.DictReader(open(CSV_PATH)))
    updates = [int(r["approx_update"]) for r in rows]
    peak = [int(r["peak_tiles"]) for r in rows]
    final = [int(r["final_tiles"]) for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(updates, peak, s=18, alpha=0.7, label="peak tiles (episode max)", color="#2563eb")
    ax.scatter(updates, final, s=18, alpha=0.5, label="final tiles (episode end)", color="#94a3b8")
    ax.set_xlabel("training update (approx.)")
    ax.set_ylabel("AGENT tiles owned")
    ax.set_title("Greatest territorial extent vs. evaluation run\n(each point = one eval episode, greedy/argmax policy)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
