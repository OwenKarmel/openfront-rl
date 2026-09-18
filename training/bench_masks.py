"""Per-step cost of the expand / boat masks, measured on identical states.

Controlled by construction: every step sends `noop`, so the game evolves
identically under all four flag combinations and the only difference is
whether canExpand()/hasShore() run while building each step's legalActions().
Comparing configs on diverging trajectories would confound mask cost with
whatever the different trajectory did (border size drives both masks'
O(border tiles) scan).

Configs are INTERLEAVED round-robin rather than run back-to-back: a training
run is normally live on the same 12 cores and its contention drifts over
minutes, which a sequential layout silently bakes into whichever config ran
during a busy stretch. Round-robin spreads that drift evenly across configs.
Each round spawns a fresh env per config, because the env vars are read at
module scope in the Node worker.

Reports the pooled median over all rounds plus the interquartile range, so a
difference smaller than the noise is visible as such rather than reported as
a result.

usage: python bench_masks.py [steps_per_round] [rounds]
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONFIGS = [
    ("baseline (both off)", {}),
    ("expand mask only", {"OPENFRONT_EXPAND_MASK": "1"}),
    ("boat mask only", {"OPENFRONT_BOAT_MASK": "1"}),
    ("both on", {"OPENFRONT_EXPAND_MASK": "1", "OPENFRONT_BOAT_MASK": "1"}),
]
FLAGS = ["OPENFRONT_EXPAND_MASK", "OPENFRONT_BOAT_MASK"]


def measure(steps: int, extra: dict[str, str]) -> list[float]:
    from train import make_env

    for f in FLAGS:
        os.environ.pop(f, None)
    os.environ.update(extra)

    args = argparse.Namespace(map=None, ticks_per_step=10, max_episode_steps=20000)
    env = make_env(args, "easy", seed_prefix="bench")
    env.reset(seed=7)
    for _ in range(40):          # warm-up: early game is a tiny map region
        env.step([0, 0, 0])

    out = []
    for _ in range(steps):
        t0 = time.perf_counter()
        _, _, term, trunc, _ = env.step([0, 0, 0])   # 0 == noop
        out.append((time.perf_counter() - t0) * 1000.0)
        if term or trunc:
            break
    env.close()
    return out


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(f"{steps} noop steps x {rounds} interleaved rounds per config "
          f"(identical trajectories)\n")

    samples: dict[str, list[float]] = {label: [] for label, _ in CONFIGS}
    for r in range(rounds):
        for label, extra in CONFIGS:
            samples[label] += measure(steps, extra)
        print(f"  round {r + 1}/{rounds} done")

    base = statistics.median(samples["baseline (both off)"])
    print(f"\n{'config':<22}{'n':>7}{'median ms':>12}{'IQR':>10}"
          f"{'overhead':>11}{'vs base':>10}")
    for label, _ in CONFIGS:
        v = samples[label]
        med = statistics.median(v)
        q = statistics.quantiles(v, n=4)
        iqr = q[2] - q[0]
        print(f"{label:<22}{len(v):>7}{med:>12.3f}{iqr:>10.3f}"
              f"{med - base:>+11.3f}{(med / base - 1) * 100:>+9.1f}%")

    print(f"\nNoise floor: baseline IQR is "
          f"{statistics.quantiles(samples['baseline (both off)'], n=4)[2] - statistics.quantiles(samples['baseline (both off)'], n=4)[0]:.3f} ms. "
          "Any overhead smaller than that is not resolvable here.")


if __name__ == "__main__":
    main()
