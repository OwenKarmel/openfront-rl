"""Eval spacing stays fixed in EXPERIENCE while the rollout length varies.

analysis/README.md names eval index as the one deliberately-valid cross-run
comparison: eval cadence is chosen per run to hold experience-between-evals
at 3000 decision steps per env, so "eval n" means the same amount of
gameplay in every run. A horizon-matched rollout breaks that if evals stay
keyed on update count -- 60 updates is 3000 steps at T=50 but 60000 at
T=1000.

Drives train.py's real eval_due()/rollout_at()/steps_before(), the same way
the update loop does, including a mid-run resume.

usage: python tests/test_eval_cadence.py   (exit 1 on any failure)
"""

import argparse
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

from train import eval_due, rollout_at, steps_before

UPDATES = 6000
TARGET = 3000


def simulate(args, start=0, stop=UPDATES):
    """The update loop's bookkeeping, verbatim: returns {update: steps-at-eval}."""
    fired = {}
    steps_done = steps_before(start, args)
    for u in range(start, stop):
        steps_prev, steps_done = steps_done, steps_done + rollout_at(u, args)
        if eval_due(u, steps_prev, steps_done, args):
            fired[u] = steps_done
    return fired


def gaps(fired):
    s = [0] + [fired[u] for u in sorted(fired)]
    return [b - a for a, b in zip(s, s[1:])]


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    return ok


def main() -> None:
    fails = 0
    sched = dict(gamma=0.999, gamma_start=0.98, gamma_anneal_updates=3000,
                 gamma_anneal_start=0, rollout_length=50, rollout_match_horizon=True)

    print("update-keyed eval under a horizon-matched rollout (the bug): experience drifts")
    by_upd = gaps(simulate(argparse.Namespace(**sched, eval_every=60, eval_every_steps=0)))
    fails += not check("gap grows ~20x as the rollout grows", max(by_upd) >= 15 * min(by_upd),
                       f"first={by_upd[0]} last={by_upd[-1]}")

    print("\nstep-keyed eval (the fix): one eval per 3000 steps")
    a = argparse.Namespace(**sched, eval_every=60, eval_every_steps=TARGET)
    fired = simulate(a)
    total = steps_before(UPDATES, a)
    fails += not check("eval count == total steps // 3000", len(fired) == total // TARGET,
                       f"{len(fired)} evals over {total} steps")
    fails += not check("k-th eval is the first update reaching k*3000",
                       all(steps // TARGET == k + 1 for k, steps in
                           enumerate(fired[u] for u in sorted(fired))), "")
    g = gaps(fired)
    fails += not check("no gap exceeds 3000 + one max rollout", max(g) < TARGET + 1000, f"max={max(g)}")
    # Once T=1000 the grid aligns exactly: every 3rd update.
    late = [u for u in sorted(fired) if u > 3100]
    fails += not check("late-run evals land every 3 updates at T=1000",
                       all(b - a == 3 for a, b in zip(late, late[1:])), f"{late[:4]}...")

    print("\nresume mid-anneal fires exactly the evals an uninterrupted run would")
    for cut in (1, 1234, 2999, 3001, 4500):
        resumed = {**simulate(a, 0, cut), **simulate(a, cut, UPDATES)}
        fails += not check(f"resume at update {cut}", resumed == fired)

    print("\nconstant rollout: step-keyed and update-keyed agree exactly (live run's config)")
    live = dict(gamma=0.98, gamma_start=None, gamma_anneal_updates=0, gamma_anneal_start=0,
                rollout_length=50, rollout_match_horizon=False)
    s = simulate(argparse.Namespace(**live, eval_every=60, eval_every_steps=TARGET))
    u = simulate(argparse.Namespace(**live, eval_every=60, eval_every_steps=0))
    fails += not check("identical eval updates", sorted(s) == sorted(u), f"{len(s)} evals each")
    fails += not check("matched-horizon at gamma 0.98 is also identical",
                       sorted(simulate(argparse.Namespace(**{**live, "rollout_match_horizon": True},
                                                          eval_every=60, eval_every_steps=TARGET))) == sorted(u))

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILURE(S)'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
