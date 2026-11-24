"""Per-map eval win rates of a run over time, against the scripted baseline
on the same seeds.

For a run directory with eval suites (eval_episodes.csv beside latest.pt):

  1. per-map win rates over time: full-suite games only (kind=suite; canary
     rows are left out, they are a subset of the same seeds), per curriculum
     stage, in windows of --window updates. Written to
     <run>/per_map_win_rates.csv and printed.
  2. the scripted baseline: training/eval_checkpoint.py --scripted (noop until
     troops >= 45% of maxTroops, then attack by legality) played ONCE per
     stage -- plus any --difficulties given -- on that stage's eval seeds, and
     cached beside the run as script_baseline_<difficulty>_<maps>_<n>.csv.
     Delete a cache file to replay it. Played under the env-bridge flags the
     run logged ("env flags:" in train_stdout.log), so both sides play the
     same game construction.
  3. agent vs script, per map: the latest window, all of the stage, and the
     script, each as wins/games -- with and without --exclude-maps.

    python analysis/eval_vs_script.py training/checkpoints/run6_bighead_attacksize
    python analysis/eval_vs_script.py RUN --window 250 --difficulties impossible
    python analysis/eval_vs_script.py RUN --no-script     # tables only

The script baseline is several minutes per 64-game suite (more at easy, whose
games run long); --envs sets its parallel workers. Check memory first if a
training run is live (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAINING = REPO / "training"
DEFAULT_PYTHON = os.environ.get("OPENFRONT_PYTHON", "/home/developer/miniconda/envs/openfront-rl/bin/python")
ENV_FLAG_VARS = ("OPENFRONT_EXPAND_MASK", "OPENFRONT_BOAT_MASK", "OPENFRONT_NEUTRAL_BOATS")


def suite_rows(run: Path) -> list[dict]:
    path = run / "eval_episodes.csv"
    if not path.exists():
        sys.exit(f"{path} not found -- is {run} a run with eval suites?")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Files predating the kind column hold full suites only.
    return [r for r in rows if r.get("kind", "suite") in ("suite", "")]


def suite_size(rows: list[dict], difficulty: str, maps: str) -> int:
    seeds = {int(r["seed"]) for r in rows if r["difficulty"] == difficulty and r["maps"] == maps}
    return max(seeds) + 1


def run_env_flags(run: Path) -> dict[str, str]:
    """The OPENFRONT_* flags the run's own log says it trained with."""
    log = run / "train_stdout.log"
    flags: dict[str, str] = {}
    if log.exists():
        with open(log, errors="replace") as fh:
            for line in fh:
                if line.startswith("env flags:"):
                    flags = dict(re.findall(r"(OPENFRONT_\w+)=(\d)", line))
    return flags


def script_baseline(run: Path, difficulty: str, maps: str, n: int, args) -> dict[int, dict]:
    """seed -> row of the scripted policy's game, from the cache or by playing it."""
    cache = run / f"script_baseline_{difficulty}_{maps}_{n}.csv"
    if not cache.exists():
        if args.no_script:
            return {}
        flags = run_env_flags(run)
        if not flags:
            sys.exit(f"no 'env flags:' line in {run / 'train_stdout.log'}; pass the run's OPENFRONT_* "
                     f"flags in the environment and --trust-env-flags")
        env = dict(os.environ)
        if not args.trust_env_flags:
            for var in ENV_FLAG_VARS:
                env.pop(var, None)
            env.update({k: v for k, v in flags.items() if v == "1"})
        print(f"playing the scripted baseline vs {difficulty} on {maps}, {n} seeds "
              f"({' '.join(f'{k}={v}' for k, v in flags.items())}) -> {cache.name}", flush=True)
        tmp = cache.with_suffix(".partial.csv")
        subprocess.run(
            [args.python, str(TRAINING / "eval_checkpoint.py"), "--scripted", "--out", str(tmp),
             "--difficulties", difficulty, "--maps", maps, "--episodes", str(n), "--envs", str(args.envs)],
            cwd=TRAINING, env=env, check=True,
        )
        tmp.rename(cache)
    with open(cache, newline="") as fh:
        return {int(r["seed"]): r for r in csv.DictReader(fh)}


def frac(w: int, g: int) -> str:
    return f"{w}/{g}" if g else "-"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="run directory (holds eval_episodes.csv)")
    p.add_argument("--window", type=int, default=500, help="updates per time window (default 500)")
    p.add_argument("--exclude-maps", default="labyrinth", help="comma-separated; '' for none")
    p.add_argument("--difficulties", default="", help="also play the script vs these difficulties on each stage's maps")
    p.add_argument("--no-script", action="store_true", help="never play the script; use caches if present")
    p.add_argument("--trust-env-flags", action="store_true",
                   help="use the OPENFRONT_* flags of this environment instead of the run's logged ones")
    p.add_argument("--envs", type=int, default=8, help="parallel workers for the scripted baseline")
    p.add_argument("--python", default=DEFAULT_PYTHON)
    args = p.parse_args()
    excl = {m for m in args.exclude_maps.split(",") if m}

    rows = suite_rows(args.run)
    stages: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        stages[(r["difficulty"], r["maps"])].append(r)

    out_rows = []
    for (difficulty, maps), srows in sorted(stages.items(), key=lambda kv: min(int(r["update"]) for r in kv[1])):
        n = suite_size(rows, difficulty, maps)
        updates = sorted({int(r["update"]) for r in srows})
        lo, hi = updates[0], updates[-1]
        print(f"\n=== {difficulty}/{maps}: {len(updates)} suites of {n}, updates {lo}-{hi} ===")
        windows: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        total: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in srows:
            w = (int(r["update"]) - lo) // args.window
            won = r["winner"] == "AGENT"
            for bucket in (windows[w][r["map"]], total[r["map"]]):
                bucket[0] += won
                bucket[1] += 1
        all_maps = sorted(total, key=lambda m: (-total[m][1], m))
        header = f"{'updates':>13s} " + " ".join(f"{m[:11]:>11s}" for m in all_maps) + f" {'all':>11s} {'excl':>11s}"
        print(header)
        for w in sorted(windows):
            a, b = lo + w * args.window, lo + (w + 1) * args.window - 1
            cells = []
            for m in all_maps:
                wins, games = windows[w][m]
                cells.append(f"{(wins / games if games else float('nan')):>11.3f}")
                out_rows.append({"difficulty": difficulty, "maps": maps, "window_start": a, "window_end": b,
                                 "map": m, "wins": wins, "games": games})
            tw = sum(v[0] for v in windows[w].values()); tg = sum(v[1] for v in windows[w].values())
            ew = sum(v[0] for m, v in windows[w].items() if m not in excl)
            eg = sum(v[1] for m, v in windows[w].items() if m not in excl)
            print(f"{a:>6d}-{b:<6d} " + " ".join(cells) + f" {tw / tg:>11.4f} {(ew / eg if eg else float('nan')):>11.4f}")

        last = max(windows)
        script = script_baseline(args.run, difficulty, maps, n, args)
        extra = [d for d in args.difficulties.split(",") if d and d != difficulty]
        extra_scripts = {d: script_baseline(args.run, d, maps, n, args) for d in extra}
        print(f"\nagent vs script per map ({difficulty}/{maps}; agent = wins/games over the latest window "
              f"and over the whole stage; script = one game per seed)")
        print(f"{'map':<18s} {'agent latest':>13s} {'agent stage':>13s} {'script':>8s}"
              + "".join(f" {'script ' + d[:4]:>12s}" for d in extra))
        seed_map = {int(r["seed"]): r["map"] for r in srows}
        def script_counts(sc: dict, m: str | None, drop=frozenset()) -> tuple[int, int]:
            games = [g for s, g in sc.items() if (m is None or seed_map.get(s) == m) and seed_map.get(s) not in drop]
            return sum(g["winner"] == "AGENT" for g in games), len(games)
        for m in all_maps:
            lw, lg = windows[last][m]
            sw, sg = script_counts(script, m)
            print(f"{m:<18s} {frac(lw, lg):>13s} {frac(*total[m]):>13s} {frac(sw, sg) if script else '-':>8s}"
                  + "".join(f" {frac(*script_counts(sc, m)):>12s}" for sc in extra_scripts.values()))
        for label, drop in (("ALL", frozenset()), (f"excl {'+'.join(sorted(excl))}", frozenset(excl))):
            lw = sum(v[0] for k, v in windows[last].items() if k not in drop)
            lg = sum(v[1] for k, v in windows[last].items() if k not in drop)
            tw = sum(v[0] for k, v in total.items() if k not in drop)
            tg = sum(v[1] for k, v in total.items() if k not in drop)
            print(f"{label:<18s} {frac(lw, lg):>13s} {frac(tw, tg):>13s} "
                  f"{frac(*script_counts(script, None, drop)) if script else '-':>8s}"
                  + "".join(f" {frac(*script_counts(sc, None, drop)):>12s}" for sc in extra_scripts.values()))

    out = args.run / "per_map_win_rates.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["difficulty", "maps", "window_start", "window_end", "map", "wins", "games"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
