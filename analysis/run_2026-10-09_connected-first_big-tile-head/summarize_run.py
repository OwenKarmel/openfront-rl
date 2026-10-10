"""Summary statistics for a connected-first run, from its own logs.

Reads a run directory's train_log.csv and eval_log.csv (what train.py writes
next to its checkpoint) and prints, per curriculum stage: when it was
reached, the eval-suite win rates, the training-episode win rate, the action
mix, and timing. --compare puts a second run beside it -- the way to check
whether a replication matches this folder's reference run.

    python summarize_run.py .                               # this folder's run
    python summarize_run.py . --compare /path/to/checkpoints/run3_big_tile_head
    python summarize_run.py . --json summary.json

Only the standard library; runs anywhere the CSVs are.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ACTIONS = ("noop", "expand", "attack_opponent", "boat_attack")
PROMOTE_WINS = 48  # 0.75 x 64-episode suite


def summarize(run_dir: Path) -> dict:
    train = list(csv.DictReader(open(run_dir / "train_log.csv")))
    evals = list(csv.DictReader(open(run_dir / "eval_log.csv")))
    gating = [e for e in evals if e["gates_curriculum"] == "True"]

    stages: list[dict] = []
    for row in train:
        key = f"{row['difficulty']}/{row['maps']}"
        if not stages or stages[-1]["stage"] != key:
            stages.append({"stage": key, "rows": []})
        stages[-1]["rows"].append(row)

    out = {
        "run": str(run_dir),
        "updates": len(train),
        "update_hours": round(sum(float(r["seconds"]) for r in train) / 3600, 1),
        "eval_hours": round(sum(float(e["seconds"]) for e in evals) / 3600, 1),
        "stages": [],
    }
    for s in stages:
        rows = s["rows"]
        first, last = int(rows[0]["update"]), int(rows[-1]["update"])
        ev = [e for e in gating if f"{e['difficulty']}/{e['maps']}" == s["stage"]]
        rates = [int(e["wins"]) / int(e["episodes"]) for e in ev]
        games = sum(int(r["episodes"]) for r in rows)
        wins = sum(round(float(r["win_rate"]) * int(r["episodes"])) for r in rows if r["episodes"] != "0")
        promoted = ev[-1] if ev and int(ev[-1]["wins"]) >= PROMOTE_WINS else None
        out["stages"].append({
            "stage": s["stage"],
            "first_update": first,
            "last_update": last,
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
        })
    return out


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


def print_side_by_side(runs: list[dict]) -> None:
    print("\n".join(
        f"run {i}: {r['run']} -- {r['updates']} updates, {r['update_hours']}h in updates + "
        f"{r['eval_hours']}h in eval suites" for i, r in enumerate(runs)
    ))
    stages = list(dict.fromkeys(s["stage"] for r in runs for s in r["stages"]))
    for stage in stages:
        print(f"\n== {stage}")
        per_run = [next((s for s in r["stages"] if s["stage"] == stage), None) for r in runs]
        for name, get in FIELDS:
            cells = [("-" if s is None else str(get(s))) for s in per_run]
            print(f"  {name:26s}" + "".join(f"{c:>34s}" for c in cells))
        for i, st in enumerate(per_run):
            if st is not None:
                print(f"  eval wins /64, run {i}:  " + " ".join(map(str, st["eval_wins"])))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--compare", type=Path, help="a second run directory to print beside it")
    p.add_argument("--json", type=Path, help="also write the first run's summary here")
    a = p.parse_args()
    runs = [summarize(a.run_dir)] + ([summarize(a.compare)] if a.compare else [])
    print_side_by_side(runs)
    if a.json:
        a.json.write_text(json.dumps(runs[0], indent=2) + "\n")


if __name__ == "__main__":
    main()
