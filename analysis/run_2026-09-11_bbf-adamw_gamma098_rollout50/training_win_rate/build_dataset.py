"""Expands train_log.csv's per-update win_rate column into one row per
TRAINING episode and writes training_win_rate.csv next to this file.

Why an expansion rather than plotting the column directly: train.py logs
`win_rate` as (episodes won / episodes finished) *within that update's
rollout*, and at rollout 50 x 10 envs only ~0.18 episodes finish per update.
So 84% of rows are `nan` and the rest are quantised to {0, 0.5, 0.75, 1.0} --
a per-update series is nearly all holes and spikes, and averaging the column
over updates would weight a 1-episode update the same as a 4-episode one.

Expanding to one row per episode fixes both: the moving average in the plot
is then a true trailing mean over episodes (each episode counted once), and
the x position is the update during which that episode finished.

Ordering within a single update is not recoverable from the log (only the
count and the fraction are recorded), so wins are emitted before losses
inside an update. This is irrelevant at any sane window size and is noted
only so the file is not mistaken for a faithful per-episode ordering.

`difficulty` is carried through: the curriculum can promote mid-run, which
would make win rates before and after incomparable. For this run it is
`easy` throughout, so the whole series is on one footing.
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_CSV = Path("/home/developer/openfront-rl/training/checkpoints/train_log.csv")
OUT = HERE / "training_win_rate.csv"


def main() -> None:
    rows = list(csv.DictReader(LOG_CSV.open()))

    episodes = []  # (update, difficulty, won)
    for r in rows:
        n = int(r["episodes"])
        if n == 0:
            continue
        raw = r["win_rate"].strip().lower()
        if raw in ("", "nan"):
            continue
        update = int(r["update"])
        # win_rate is wins/episodes for this update; recover the integer
        # count. round() rather than int() so 2/3 -> 0.6666 does not
        # truncate to 1 win instead of 2.
        wins = round(float(r["win_rate"]) * n)
        for k in range(n):
            episodes.append((update, r["difficulty"], 1 if k < wins else 0))

    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode_index", "update", "difficulty", "won", "cum_win_rate"])
        cum = 0
        for i, (update, difficulty, won) in enumerate(episodes):
            cum += won
            w.writerow([i, update, difficulty, won, f"{cum / (i + 1):.6f}"])

    total = len(episodes)
    wins = sum(e[2] for e in episodes)
    print(f"wrote {total} training-episode rows -> {OUT}")
    print(f"  wins={wins}  overall win rate={wins / total:.4f}")
    print(f"  update range: {episodes[0][0]} -> {episodes[-1][0]}")
    difficulties = sorted({e[1] for e in episodes})
    print(f"  difficulties present: {difficulties}"
          + ("  (single difficulty -- series is comparable throughout)"
             if len(difficulties) == 1 else "  (MULTIPLE -- see README)"))


if __name__ == "__main__":
    main()
