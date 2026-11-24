"""--attack-size-head, env-bridge side: troopFraction on the wire.

  1. reset echo: attack_size_head=True round-trips; an env-bridge reply
     without the echo (one that predates the option) fails at reset
  2. the bridge rejects troopFraction on a head-off episode, a missing or
     out-of-range one on a head-on attack step, and one sent with noop
  3. the fraction takes effect: in lockstep on one seed, a 75% expand leaves
     far fewer troops at home than a 10% expand
  4. fraction 0.20 vs the head off, in lockstep on fixed seeds through
     engine-ended games, with a deterministic scripted policy:
       - BOATS: exactly the same game, every step (observation, reward,
         done, winner). The head-off path already sized boats in intentFor()
         as boatAttackAmount() = floor(troops/5); the head sends
         floor(0.2 x troops()) at the same moment.
       - LAND ATTACKS: NOT bit-identical, by design of the engine's null
         path, and this test pins the difference down. troops: null makes
         AttackExecution.init() compute troops/5 at the END of the tick
         (GameImpl.executeNextTick() inits new executions after ticking the
         running ones), i.e. after one more tick of troop growth; an explicit
         count is fixed when the intent is built -- as the real client does
         (ClientGameRunner: troops() * attackRatio). So the games agree on
         every step before the first land attack, and on the step of it AGENT's
         troops differ by under 1% (the head-on attack is the smaller one).
         README "Action space" has the measured numbers.

usage: python tests/test_attack_size.py   (exit 1 on any failure; a few minutes)
"""

import os
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

import numpy as np  # noqa: E402

from envs.openfront_env import (  # noqa: E402
    ACTIONS,
    ATTACK_FRACTIONS,
    DEFAULT_ATTACK_FRACTION_IDX,
    SMALL_MAPS,
    OpenFrontEnv,
)

# The training launchers' env-bridge flags: test the configuration that trains.
for var in ("OPENFRONT_EXPAND_MASK", "OPENFRONT_BOAT_MASK", "OPENFRONT_NEUTRAL_BOATS"):
    os.environ[var] = "1"

NOOP, EXPAND, ATTACK, BOAT = (ACTIONS.index(a) for a in ("noop", "expand", "attack_opponent", "boat_attack"))
TROOPS = 1  # SELF_FEATURES index
D = DEFAULT_ATTACK_FRACTION_IDX
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")


def make(head, seed_prefix="attack-size", max_steps=3000):
    return OpenFrontEnv(map_pool=SMALL_MAPS, seed=seed_prefix, difficulty="hard", max_steps=max_steps,
                        attack_size_head=head)


def raises(fn) -> str | None:
    try:
        fn()
    except RuntimeError as e:
        return str(e)
    return None


print("1. reset echo")
env = make(True)
try:
    env.reset(seed=0)
    check("attack_size_head=True: reset succeeds (echo matches)", True)
    real_send = env._send
    env._send = lambda cmd: {k: v for k, v in real_send(cmd).items() if k != "attackSizeHead"}
    msg = raises(lambda: env.reset(seed=0))
    check("a reply without the echo (older env-bridge) fails at reset", msg is not None and "predates" in msg, msg or "")
    env._send = real_send

    print("2. bridge-side validation")
    env.reset(seed=1)
    for label, cmd in (
        ("attack without troopFraction", {"cmd": "step", "action": "expand"}),
        ("troopFraction 0", {"cmd": "step", "action": "expand", "troopFraction": 0}),
        ("troopFraction 1.5", {"cmd": "step", "action": "expand", "troopFraction": 1.5}),
        ("troopFraction with noop", {"cmd": "step", "action": "noop", "troopFraction": 0.2}),
    ):
        msg = raises(lambda cmd=cmd: env._send(cmd))
        check(f"head on, {label}: rejected", msg is not None, msg or "")
finally:
    env.close()
env = make(False)
try:
    env.reset(seed=1)
    msg = raises(lambda: env._send({"cmd": "step", "action": "expand", "troopFraction": 0.2}))
    check("head off, troopFraction sent: rejected", msg is not None and "without attackSizeHead" in msg, msg or "")
finally:
    env.close()

print("3. the fraction takes effect")
envs = [make(True), make(True)]
try:
    outs = [e.reset(seed=2) for e in envs]
    for t in range(200):
        if outs[0][1]["action_mask"][EXPAND]:
            break
        outs = [(r[0], r[4]) for r in (e.step([NOOP, 0, 0, D]) for e in envs)]
    before = [o[0]["self_features"][TROOPS] for o in outs]
    res = [e.step([EXPAND, 0, 0, i]) for e, i in zip(envs, (ATTACK_FRACTIONS.index(0.10), ATTACK_FRACTIONS.index(0.75)))]
    after = [r[0]["self_features"][TROOPS] for r in res]
    check(f"one expand at 10% vs 75% (troops {before[0]:.0f} -> {after[0]:.0f} vs {before[1]:.0f} -> {after[1]:.0f})",
          before[0] == before[1] and after[1] < 0.5 * after[0])
finally:
    for e in envs:
        e.close()


def policy(t, obs, mask, kinds):
    if "attack" in kinds and mask[ATTACK] and t % 7 == 0:
        return [ATTACK, 0, 0, D]
    if "boat" in kinds and mask[BOAT] and t % 13 == 0:
        cells = np.flatnonzero(obs["boat_target_mask"].reshape(-1))
        return [BOAT, int(cells[t % len(cells)]), 0, D]
    if "expand" in kinds and mask[EXPAND] and t % 3 == 0:
        return [EXPAND, 0, 0, D]
    return [NOOP, 0, 0, D]


def lockstep(seed, kinds, max_steps=3000):
    """Head off vs head on at 0.20, same seed and policy. Returns (steps,
    first divergence step or None, the action type at it, troops off/on just
    after it, winners, boats launched)."""
    pair = [make(False, "equiv", max_steps), make(True, "equiv", max_steps)]
    try:
        outs = [e.reset(seed=seed) for e in pair]
        first, first_kind, troops, boats = None, None, None, 0
        for t in range(max_steps):
            acts = [policy(t, o[0], o[1]["action_mask"], kinds) for o in outs]
            boats += acts[0][0] == BOAT
            for e, a in zip(pair, acts):
                e.step_send(a)
            res = [e.step_recv() for e in pair]
            (o0, r0, d0, tr0, i0), (o1, r1, d1, tr1, i1) = res
            same = (np.array_equal(o0["tile_grid"], o1["tile_grid"])
                    and np.array_equal(o0["self_features"], o1["self_features"])
                    and np.array_equal(o0["opponent_features"], o1["opponent_features"])
                    and r0 == r1 and d0 == d1 and i0["winner"] == i1["winner"])
            if not same and first is None:
                first, first_kind = t, acts[0][0]
                troops = (o0["self_features"][TROOPS], o1["self_features"][TROOPS])
            outs = [(o0, i0), (o1, i1)]
            if d0 or d1 or tr0 or tr1:
                break
        return t + 1, first, first_kind, troops, (i0["winner"], i1["winner"]), boats, pair[0].map_name
    finally:
        for e in pair:
            e.close()


print("4. fraction 0.20 vs head off")
for seed in (0, 2):
    steps, first, _, _, winners, boats, m = lockstep(seed, ("boat",))
    check(f"boats only, seed {seed} ({m}): identical for all {steps} steps to the engine's end "
          f"({boats} boat_attack steps, winner {winners[0]})", first is None and boats > 0)
for seed in (0, 1):
    steps, first, kind, troops, winners, _, m = lockstep(seed, ("expand", "attack"), max_steps=600)
    rel = abs(troops[1] - troops[0]) / troops[0] if troops else float("nan")
    check(f"land attacks, seed {seed} ({m}): identical until the first land attack (step {first}, "
          f"{ACTIONS[kind] if kind is not None else '-'}), then AGENT's troops differ by {rel:.2%} "
          f"(off {troops[0]:.0f} / on {troops[1]:.0f})" if troops else f"land attacks, seed {seed}: no divergence",
          first is not None and kind in (EXPAND, ATTACK) and rel < 0.01 and troops[1] > troops[0])

print("ALL PASSED" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
