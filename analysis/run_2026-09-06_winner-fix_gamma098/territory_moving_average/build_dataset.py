"""Extracts one row per eval episode from the live run's train_stdout.log
and writes territory_moving_average.csv next to this file.

Each `eval vs <difficulty>: {...}` line in the log is one greedy/argmax
evaluation episode, in chronological order, so the row index IS the
evaluation-run number. `peak_territory_frac` is AGENT's share of the map's
LAND tiles at its high-water mark during that episode (see
territory_fraction() in train.py -- it is read off tile_grid's own class
proportions, so it stays valid on the resized map_pool canvas).

Also records which gamma each eval ran under: this run switched from
gamma=0.999 to gamma=0.98 (and max_episode_steps 20000 -> 2500) at update
710, mid-run, which is the comparison the plot exists to show.
"""

import ast
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path("/home/developer/openfront-rl/training/checkpoints/train_stdout.log")
OUT = HERE / "territory_moving_average.csv"

# train.py runs an eval every --eval-every updates, at update+1.
EVAL_EVERY = 10
# The run resumed at update 710 with gamma 0.999 -> 0.98 and
# max_episode_steps 20000 -> 2500 (see "resumed from" in the log).
GAMMA_SWITCH_UPDATE = 710


def main() -> None:
    eval_re = re.compile(r"eval vs (\w+): (\{.*\})")
    rows = []
    for line in LOG.read_text().splitlines():
        m = eval_re.search(line)
        if not m:
            continue
        difficulty, payload = m.group(1), m.group(2)
        try:
            d = ast.literal_eval(payload)
        except (ValueError, SyntaxError):
            continue
        rows.append((difficulty, d))

    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "eval_index", "approx_update", "gamma", "difficulty",
            "peak_territory_frac", "self_tiles", "opp_tiles", "ticks", "winner",
        ])
        for i, (difficulty, d) in enumerate(rows):
            approx_update = (i + 1) * EVAL_EVERY
            gamma = 0.999 if approx_update <= GAMMA_SWITCH_UPDATE else 0.98
            w.writerow([
                i, approx_update, gamma, difficulty,
                d["peak_territory_frac"], d["self_tiles"], d["opp_tiles"],
                d["ticks"], d["winner"] if d["winner"] is not None else "",
            ])

    print(f"wrote {len(rows)} eval rows -> {OUT}")


if __name__ == "__main__":
    main()
