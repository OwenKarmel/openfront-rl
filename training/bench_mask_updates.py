"""End-to-end PPO update time under each mask configuration.

Companion to bench_masks.py (which isolates per-step cost on identical
states). This measures whole updates -- rollout collection plus the PPO
update itself -- which is what actually sets training throughput.

Unavoidable caveat, stated rather than hidden: unlike bench_masks.py this is
NOT a controlled comparison. The masks change which actions are legal, so
each config collects a different trajectory, and update time depends on what
the agent did (border size, episode resets, boat pathfinding). Treat it as
"what throughput looks like under each config", not "the cost of the mask".
bench_masks.py is the controlled number.

Runs with a small --num-envs so it does not fight a live training run for
the 12 cores; absolute numbers are therefore lower than a real 10-env run,
but the comparison between configs is the point. Configs are interleaved
round-robin to spread contention drift.

usage: python bench_mask_updates.py [updates_per_run] [rounds]
"""

import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = [
    ("baseline (both off)", {}),
    ("expand mask only", {"OPENFRONT_EXPAND_MASK": "1"}),
    ("boat mask only", {"OPENFRONT_BOAT_MASK": "1"}),
    ("both on", {"OPENFRONT_EXPAND_MASK": "1", "OPENFRONT_BOAT_MASK": "1"}),
]
FLAGS = ["OPENFRONT_EXPAND_MASK", "OPENFRONT_BOAT_MASK"]


def run(updates: int, extra: dict[str, str]) -> list[float]:
    env = dict(os.environ)
    for f in FLAGS:
        env.pop(f, None)
    env.update(extra)
    tmp = Path(tempfile.mkdtemp(prefix="benchupd_"))
    try:
        subprocess.run(
            [sys.executable, "-u", str(HERE / "train.py"),
             "--updates", str(updates), "--num-envs", "2", "--rollout-length", "50",
             "--gamma", "0.98", "--epochs", "8", "--weight-decay", "0.1",
             "--minibatch-size", "128", "--eval-every", "100000",
             "--checkpoint-every", "100000", "--device", "cpu",
             "--checkpoint-dir", str(tmp), "--log-csv", str(tmp / "log.csv"),
             "--eval-dir", str(tmp / "replays")],
            env=env, capture_output=True, text=True, cwd=str(HERE),
        )
        rows = list(csv.DictReader((tmp / "log.csv").open()))
        # drop the first update: process spawn + map load land in it
        return [float(r["seconds"]) for r in rows[1:]]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    updates = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    print(f"{updates} updates x {rounds} interleaved rounds per config, "
          f"2 envs, rollout 50 (first update of each run discarded)\n")

    samples: dict[str, list[float]] = {label: [] for label, _ in CONFIGS}
    for r in range(rounds):
        for label, extra in CONFIGS:
            samples[label] += run(updates, extra)
        print(f"  round {r + 1}/{rounds} done")

    base = statistics.median(samples["baseline (both off)"])
    print(f"\n{'config':<22}{'n':>5}{'median s/update':>18}{'spread':>10}"
          f"{'overhead':>11}{'vs base':>10}")
    for label, _ in CONFIGS:
        v = samples[label]
        med = statistics.median(v)
        print(f"{label:<22}{len(v):>5}{med:>18.2f}{max(v) - min(v):>10.2f}"
              f"{med - base:>+11.2f}{(med / base - 1) * 100:>+9.1f}%")


if __name__ == "__main__":
    main()
