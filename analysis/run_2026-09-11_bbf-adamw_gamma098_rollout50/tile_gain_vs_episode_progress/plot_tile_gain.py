"""Plots tile gain per rollout window against how far into the episode that
window fell, with colour encoding when in training the episode happened
(light = early, dark = late).

Reads tile_gain.csv (see build_dataset.py), writes tile_gain.png.

Colour is a SEQUENTIAL encoding -- one hue, light to dark, monotonic in
lightness -- because training update is an ordered magnitude, not an
identity. It is the family blue truncated to 0.25-1.0 of matplotlib's Blues
so the lightest step still reads against a white surface. A colourbar is the
legend; a categorical legend would be wrong for a continuous variable.

Y is symlog, not linear. 84.5% of windows have EXACTLY zero tile change and
the remainder span -91k to +76k, so a linear axis collapses everything
interesting onto the zero line. symlog keeps zero at zero, is linear inside
+/-100 (so the dense zero band does not smear), and logarithmic outside it.
Zero is drawn as a reference line rather than left implicit, since "no change
at all" is the modal outcome and the single most important thing to see.

A rollout window is 500 ticks: --rollout-length 50 x --ticks-per-step 10,
i.e. exactly the slice of game time one PPO update sees.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "tile_gain.csv"
OUT_PATH = HERE / "tile_gain.png"

LINTHRESH = 100  # symlog: linear inside +/-100, log outside


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    tick = np.array([int(r["window_mid_tick"]) for r in rows])
    gain = np.array([int(r["tile_gain"]) for r in rows])
    update = np.array([int(r["update"]) for r in rows])

    # Truncate Blues to 0.25-1.0: the bottom of the ramp is near-white and
    # would vanish against the surface.
    base = plt.get_cmap("Blues")
    cmap = LinearSegmentedColormap.from_list(
        "Blues_trunc", base(np.linspace(0.25, 1.0, 256))
    )

    # Chronological draw order so the most recent episodes sit on top --
    # the question the plot answers is "where is training now".
    order = np.argsort(update)

    fig, ax = plt.subplots(figsize=(12.5, 6.8))

    ax.axhline(0, color="#64748b", lw=1.0, alpha=0.7, zorder=1)

    sc = ax.scatter(
        tick[order], gain[order], c=update[order], cmap=cmap,
        s=26, alpha=0.82, linewidth=0.3, edgecolor="white", zorder=3,
    )

    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.set_xlabel("ticks into episode  (1 tick = 0.1s of game time; 500 ticks = one rollout)")
    ax.set_ylabel(f"tiles gained per 500-tick rollout window\n(symlog, linear within ±{LINTHRESH})")

    zero_frac = float((gain == 0).mean())
    ax.set_title(
        "Tile gain per rollout vs. position in episode\n"
        f"run_2026-09-11_bbf-adamw -- {len(rows)} rollout windows across 40 eval episodes; "
        f"{zero_frac:.1%} of windows gained exactly zero tiles"
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("training update  (light = early in training, dark = late)")

    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")

    print(f"  windows={len(rows)}  zero-gain={zero_frac:.3f}")
    print(f"  gain: min={gain.min()} max={gain.max()} mean={gain.mean():.1f}")
    # Does activity shift with training? Split on median update.
    mid = np.median(update)
    early, late = gain[update <= mid], gain[update > mid]
    print(f"  early half (update<={mid:.0f}): n={len(early)} zero={float((early==0).mean()):.3f} "
          f"mean={early.mean():.1f}")
    print(f"  late  half (update> {mid:.0f}): n={len(late)} zero={float((late==0).mean()):.3f} "
          f"mean={late.mean():.1f}")
    # Does activity shift with episode progress?
    for lo, hi in [(0, 5000), (5000, 15000), (15000, 40000), (40000, 10**9)]:
        m = (tick >= lo) & (tick < hi)
        if m.sum():
            print(f"  ticks {lo:>6}-{hi if hi < 10**8 else 'end':>6}: n={int(m.sum()):4d} "
                  f"zero={float((gain[m]==0).mean()):.3f} mean_gain={gain[m].mean():8.1f}")


if __name__ == "__main__":
    main()
