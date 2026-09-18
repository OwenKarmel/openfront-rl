"""gamma_at / rollout_at: no-op defaults, anneal shape, boundaries, resume safety.

The property that matters most is that both are pure functions of the update
counter. Checkpoints store the update number but not hyperparameters, so a
resume reconstructs the schedule only by recomputing it; anything held in
mutable state would silently restart the anneal.

usage: python tests/test_schedules.py      (exit 1 on any failure)
"""

import argparse
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

from train import gamma_at, horizon_steps, rollout_at, steps_before


def mk(**kw) -> argparse.Namespace:
    base = dict(gamma=0.99, gamma_start=None, gamma_anneal_updates=0,
                gamma_anneal_start=0, rollout_length=64, rollout_match_horizon=False)
    base.update(kw)
    return argparse.Namespace(**base)


ANNEAL = dict(gamma=0.999, gamma_start=0.98, gamma_anneal_updates=3000)


def check(name, got, want):
    ok = got == want if isinstance(want, (bool, int)) else abs(got - want) < 1e-9
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    return ok


def main() -> None:
    fails = 0

    print("gamma_at -- no --gamma-start means constant, whatever else is set")
    for kw in (dict(gamma=0.98), dict(gamma=0.98, gamma_anneal_updates=3000, gamma_anneal_start=7)):
        a = mk(**kw)
        for u in (0, 1, 1000, 10**9):
            fails += not check(f"{kw} update {u}", gamma_at(u, a), 0.98)

    print("\ngamma_at -- anneal endpoints are exact")
    a = mk(**ANNEAL)
    fails += not check("update 0 == gamma_start", gamma_at(0, a), 0.98)
    fails += not check("update 3000 == gamma, bit-exact", gamma_at(3000, a) == 0.999, True)
    fails += not check("past the end stays pinned", gamma_at(99999, a) == 0.999, True)
    seq = [gamma_at(u, a) for u in range(3001)]
    fails += not check("strictly increasing across the anneal",
                       all(x < y for x, y in zip(seq, seq[1:])), True)

    print("\ngamma_at -- geometric in (1-gamma), not linear in gamma")
    mid = gamma_at(1500, a)
    fails += not check("midpoint (1-gamma) is the geometric mean",
                       1 - mid, ((1 - 0.98) * (1 - 0.999)) ** 0.5)
    fails += not check("midpoint is NOT the linear one",
                       abs(mid - (0.98 + 0.999) / 2) > 1e-3, True)
    growth = [(1 - seq[u]) / (1 - seq[u + 1]) for u in range(3000)]
    fails += not check("horizon grows by a constant factor per update",
                       max(growth) - min(growth) < 1e-9, True)
    fails += not check("... of 20**(1/3000)", growth[0], 20 ** (1 / 3000))

    print("\ngamma_at -- --gamma-anneal-start shifts the whole schedule (warm start)")
    b = mk(**ANNEAL, gamma_anneal_start=11300)
    fails += not check("before the start holds gamma_start", gamma_at(5000, b), 0.98)
    fails += not check("at the start == gamma_start", gamma_at(11300, b), 0.98)
    for u in (1, 750, 1500, 2999):
        fails += not check(f"offset {u} matches cold schedule", gamma_at(11300 + u, b), gamma_at(u, a))
    fails += not check("end of shifted anneal == gamma", gamma_at(14300, b) == 0.999, True)

    print("\nhorizon_steps -- float error must not knock off a step")
    for g, want in ((0.98, 50), (0.99, 100), (0.995, 200), (0.997, 333), (0.999, 1000)):
        fails += not check(f"1/(1-{g})", horizon_steps(g), want)

    print("\nrollout_at -- off by default: --rollout-length exactly, even while gamma anneals")
    for rl in (50, 64, 300):
        fails += not check(f"rollout_length={rl}", rollout_at(0, mk(rollout_length=rl)), rl)
        fails += not check(f"rollout_length={rl}, annealing, mid",
                           rollout_at(1500, mk(rollout_length=rl, **ANNEAL)), rl)

    print("\nrollout_at -- --rollout-match-horizon: exactly one discount horizon, uncapped")
    fails += not check("constant gamma 0.98 -> 50 (the live run's rollout)",
                       rollout_at(0, mk(gamma=0.98, rollout_match_horizon=True)), 50)
    c = mk(**ANNEAL, rollout_match_horizon=True, rollout_length=64)
    fails += not check("start: horizon of 0.98", rollout_at(0, c), 50)
    fails += not check("end: horizon of 0.999 (no cap)", rollout_at(3000, c), 1000)
    for u in (0, 1, 777, 1500, 2999, 3000, 5000):
        fails += not check(f"update {u}: == horizon_steps(gamma_at)",
                           rollout_at(u, c), horizon_steps(gamma_at(u, c)))
    ts = [rollout_at(u, c) for u in range(3001)]
    fails += not check("monotone non-decreasing", all(x <= y for x, y in zip(ts, ts[1:])), True)
    fails += not check("--rollout-length is ignored", rollout_at(10, c) != 64, True)

    print("\npurity / resume -- recomputing from scratch gives identical answers")
    fresh = mk(**ANNEAL, rollout_match_horizon=True, rollout_length=64)
    for u in (0, 1234, 2999, 4000):
        fails += not check(f"gamma_at({u}) across a restart", gamma_at(u, c) == gamma_at(u, fresh), True)
        fails += not check(f"rollout_at({u}) across a restart", rollout_at(u, c), rollout_at(u, fresh))
    fails += not check("steps_before(1234) == running sum", steps_before(1234, fresh), sum(ts[:1234]))
    fails += not check("steps_before is 50/update when unscheduled",
                       steps_before(11299, mk(gamma=0.98, rollout_length=50)), 11299 * 50)

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILURE(S)'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
