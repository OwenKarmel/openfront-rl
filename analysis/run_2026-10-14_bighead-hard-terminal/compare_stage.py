"""Side-by-side of runs within ONE curriculum stage, aligned by updates since
each run entered it.

summarize_run.py averages each stage over however long a run spent there,
which is fine for checking a replication's milestones but misleading for two
runs that sit in the same stage for very different spans -- e.g. the
shaping-only replication (run3_big_tile_head, 870+ updates at hard) against
the terminal-reward arm started at hard (run4_bighead_hard_terminal). This
prints summarize_run.py's per-stage table, in the same layout, once per
window of updates since entry and then once over the span every run has
logged, so the comparison is always like for like.

run4 starts from the big-tile-head reference's update-1435 checkpoint, 62
updates after that run's promotion to hard at 1373 -- pass --entry 1=1373 so
both runs are aligned on their real promotion. Windows before a run's first
logged update show "-".

    python compare_stage.py training/checkpoints/run3_big_tile_head \
        training/checkpoints/run4_bighead_hard_terminal
    python compare_stage.py DIR1 DIR2 --stage medium/small --window 50

mean_reward is not shown: shaping_and_terminal adds +1/-1 at the end of a
game, so it is on a different scale from shaping_only. Only the standard
library; runs anywhere the CSVs are.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

ACTIONS = ("noop", "expand", "attack_opponent", "boat_attack")
PROMOTE_WINS = 48  # 0.75 x 64-episode suite


def load(run_dir: Path, stage: str, entry: int | None = None) -> dict:
    # Keep the last row per update: a resume from a checkpoint a few updates
    # back re-runs (and re-logs) those updates.
    by_update = {}
    for r in csv.DictReader(open(run_dir / "train_log.csv")):
        by_update[int(r["update"])] = r
    train = [by_update[u] for u in sorted(by_update)]
    rows = [r for r in train if f"{r['difficulty']}/{r['maps']}" == stage]
    if not rows:
        raise SystemExit(f"{run_dir}: no updates at stage {stage}")
    # A run resumed from another run's checkpoint enters the stage in its
    # own log later than its weights did; --entry aligns on the real one.
    entry = int(rows[0]["update"]) if entry is None else entry
    after = [r for r in train if int(r["update"]) > int(rows[-1]["update"])]
    evals = [
        e for e in csv.DictReader(open(run_dir / "eval_log.csv"))
        if e["gates_curriculum"] == "True" and f"{e['difficulty']}/{e['maps']}" == stage
    ]
    return {
        "dir": str(run_dir),
        "entry": entry,
        "span": int(rows[-1]["update"]) - entry + 1,
        "first": int(rows[0]["update"]),
        "left_to": f"{after[0]['difficulty']}/{after[0]['maps']} at update {after[0]['update']}" if after else None,
        "reward_mode": rows[-1]["reward_mode"],
        "rows": [(int(r["update"]) - entry, r) for r in rows],
        "evals": [(int(e["update"]) - entry, e) for e in evals],
    }


def stats(run: dict, lo: int, hi: int) -> dict | None:
    """summarize_run.py's per-stage numbers, over updates [lo, hi) since entry."""
    rows = [r for off, r in run["rows"] if lo <= off < hi]
    if not rows:
        return None
    ev = [e for off, e in run["evals"] if lo <= off < hi]
    rates = [int(e["wins"]) / int(e["episodes"]) for e in ev]
    games = sum(int(r["episodes"]) for r in rows)
    wins = sum(round(float(r["win_rate"]) * int(r["episodes"])) for r in rows if r["episodes"] != "0")
    promoted = next((e for e in ev if int(e["wins"]) >= PROMOTE_WINS), None)
    return {
        "first_update": int(rows[0]["update"]),
        "last_update": int(rows[-1]["update"]),
        "promoted_at_update": int(promoted["update"]) if promoted else None,
        "evals": len(ev),
        "eval_wins": [int(e["wins"]) for e in ev],
        "eval_win_rate_mean": round(statistics.mean(rates), 3) if rates else None,
        "eval_win_rate_median": round(statistics.median(rates), 3) if rates else None,
        "eval_win_rate_max": round(max(rates), 3) if rates else None,
        "eval_mean_peak_territory": round(statistics.mean(float(e["mean_peak_territory"]) for e in ev), 3) if ev else None,
        "eval_mean_ticks": round(statistics.mean(float(e["mean_ticks"]) for e in ev)) if ev else None,
        "train_games": games,
        "train_win_rate": round(wins / games, 3) if games else None,
        **{f"type_prob_{a}": round(statistics.mean(float(r[f"type_prob_{a}"]) for r in rows), 3) for a in ACTIONS},
        "type_entropy": round(statistics.mean(float(r["type_entropy"]) for r in rows), 3),
        "kl_early_stop_rate": round(statistics.mean(float(r["stopped_early"]) for r in rows), 3),
        "rollout_T_at_end": int(rows[-1]["rollout_length"]),
        "update_seconds_median": round(statistics.median(float(r["seconds"]) for r in rows), 1),
        "eval_seconds_median": round(statistics.median(float(e["seconds"]) for e in ev)) if ev else None,
    }


# Same rows, names and layout as summarize_run.py's side-by-side, so the two
# scripts' output reads the same.
FIELDS = [
    ("updates in stage", lambda s: f"{s['first_update']}-{s['last_update']}"),
    ("promoted at update", lambda s: s["promoted_at_update"]),
    ("evals", lambda s: s["evals"]),
    ("eval win rate mean", lambda s: s["eval_win_rate_mean"]),
    ("eval win rate median", lambda s: s["eval_win_rate_median"]),
    ("eval win rate max", lambda s: s["eval_win_rate_max"]),
    ("eval mean peak territory", lambda s: s["eval_mean_peak_territory"]),
    ("eval mean game ticks", lambda s: s["eval_mean_ticks"]),
    ("training games", lambda s: s["train_games"]),
    ("training win rate", lambda s: s["train_win_rate"]),
    *[(f"P({a})", (lambda a: lambda s: s[f"type_prob_{a}"])(a)) for a in ACTIONS],
    ("type-head entropy", lambda s: s["type_entropy"]),
    ("KL early-stop rate", lambda s: s["kl_early_stop_rate"]),
    ("rollout T at end", lambda s: s["rollout_T_at_end"]),
    ("update seconds (median)", lambda s: s["update_seconds_median"]),
    ("eval seconds (median)", lambda s: s["eval_seconds_median"]),
]


def print_block(title: str, per_run: list[dict | None]) -> None:
    print(f"\n== {title}")
    for name, get in FIELDS:
        cells = [("-" if s is None else str(get(s))) for s in per_run]
        print(f"  {name:26s}" + "".join(f"{c:>34s}" for c in cells))
    for i, st in enumerate(per_run):
        if st is not None:
            print(f"  eval wins /64, run {i}:  " + " ".join(map(str, st["eval_wins"])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--stage", default="hard/small", help="difficulty/maps, as in train_log.csv")
    p.add_argument("--window", type=int, default=100, help="updates per block")
    p.add_argument(
        "--entry", action="append", default=[], metavar="I=UPDATE",
        help="treat run I (0-based) as having entered the stage at UPDATE, for a run "
             "started from another run's checkpoint already in the stage; repeatable",
    )
    a = p.parse_args()
    entries = {int(i): int(u) for i, u in (e.split("=") for e in a.entry)}
    runs = [load(d, a.stage, entries.get(i)) for i, d in enumerate(a.run_dirs)]

    print(f"stage {a.stage}, aligned by updates since entering it")
    for i, r in enumerate(runs):
        left = f", left to {r['left_to']}" if r["left_to"] else ", still there"
        logged = f" (logged from {r['first']})" if r["first"] != r["entry"] else ""
        print(f"  run {i}: {r['dir']}  reward={r['reward_mode']}  entered at update {r['entry']}{logged}, "
              f"{r['span']} updates in stage{left}")

    longest = max(r["span"] for r in runs)
    for lo in range(0, longest, a.window):
        hi = lo + a.window
        print_block(f"{a.stage}, updates [{lo}, {hi}) since entry", [stats(r, lo, hi) for r in runs])

    lo = max(r["first"] - r["entry"] for r in runs)
    hi = min(r["span"] for r in runs)
    print_block(f"{a.stage}, TOTAL over updates [{lo}, {hi}) since entry -- the span every run has logged",
                [stats(r, lo, hi) for r in runs])


if __name__ == "__main__":
    main()
