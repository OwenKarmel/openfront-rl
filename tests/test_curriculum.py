"""Curriculum ladders promote on eval-suite win rate, one stage at a time.

Covers curriculum.py's CurriculumScheduler and LADDERS, plus train.py's
wilson_interval() (the CI printed beside every suite) and map_pool_for()
(how a stage's map key becomes the pool the envs draw from). No env needed.

usage: python tests/test_curriculum.py   (exit 1 on any failure)
"""

import argparse
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

from curriculum import DIFFICULTIES, LADDERS, CurriculumScheduler
from envs.openfront_env import CONNECTED_MAPS, MAP_POOLS, SMALL_MAPS
from train import map_pool_for, wilson_interval


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    return ok


def main() -> None:
    fails = 0

    print("ladders")
    fails += not check("'difficulty' is the original ladder, launch map pool throughout",
                       [(s.difficulty, s.maps) for s in LADDERS["difficulty"]] == [(d, None) for d in DIFFICULTIES])
    cf = LADDERS["connected-first"]
    fails += not check("'connected-first' starts on connected maps at easy",
                       (cf[0].difficulty, cf[0].maps) == ("easy", "connected"))
    fails += not check("'connected-first' ends at impossible on every small map",
                       (cf[-1].difficulty, cf[-1].maps) == ("impossible", "small"))
    one_change = all(
        (a.difficulty != b.difficulty) + (a.maps != b.maps) == 1 for a, b in zip(cf, cf[1:])
    )
    fails += not check("every promotion changes exactly one of difficulty / map pool", one_change)
    keys = {s.maps for ladder in LADDERS.values() for s in ladder} - {None}
    fails += not check("every stage's map key exists in MAP_POOLS", keys <= set(MAP_POOLS), str(keys))
    fails += not check("CONNECTED_MAPS is a subset of SMALL_MAPS", set(CONNECTED_MAPS) <= set(SMALL_MAPS))

    print("\npromotion")
    s = CurriculumScheduler(ladder="connected-first", promote_threshold=0.75)
    fails += not check("holds below the threshold", s.record_eval(47, 64) == "holding" and s.state()["index"] == 0,
                       f"47/64 = {47 / 64:.3f}")
    fails += not check("remembers the latest suite at this stage", s.last_eval == (47, 64))
    fails += not check("promotes at the threshold", s.record_eval(48, 64) == "promoted",
                       f"48/64 = {48 / 64:.3f}")
    fails += not check("promotion moves to the next stage and clears last_eval",
                       (s.difficulty, s.maps, s.last_eval) == ("easy", "small", None))
    for _ in range(10):
        s.record_eval(64, 64)
    fails += not check("never promotes past the last stage",
                       s.state()["index"] == len(cf) - 1 and s.record_eval(64, 64) == "holding")
    rejected = 0
    for bad in [(1, 0), (-1, 4), (5, 4)]:
        try:
            CurriculumScheduler().record_eval(*bad)
        except ValueError:
            rejected += 1
    fails += not check("rejects impossible suite results", rejected == 3)
    try:
        CurriculumScheduler(ladder="nope")
        ok = False
    except ValueError:
        ok = True
    fails += not check("rejects an unknown ladder", ok)

    print("\nwilson_interval")
    lo, hi = wilson_interval(0, 64)
    fails += not check("0/64 stays inside [0, 1] and is not degenerate", lo == 0.0 and 0.0 < hi < 0.1,
                       f"[{lo:.3f}, {hi:.3f}]")
    lo, hi = wilson_interval(48, 64)
    fails += not check("48/64 brackets 0.75 at roughly +/-0.1", lo < 0.75 < hi and 0.15 < hi - lo < 0.25,
                       f"[{lo:.3f}, {hi:.3f}]")
    lo, hi = wilson_interval(64, 64)
    fails += not check("64/64 tops out at exactly 1", hi == 1.0 and lo > 0.9, f"[{lo:.3f}, {hi:.3f}]")

    print("\nmap_pool_for")
    launch = argparse.Namespace(map=None)
    fixed = argparse.Namespace(map="onion")
    fails += not check("no stage pool, no --map -> SMALL_MAPS (the old default)", map_pool_for(launch, None) == SMALL_MAPS)
    fails += not check("no stage pool, --map onion -> fixed map (no pool)", map_pool_for(fixed, None) is None)
    fails += not check("stage pool 'connected' -> CONNECTED_MAPS", map_pool_for(launch, "connected") == CONNECTED_MAPS)

    print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
