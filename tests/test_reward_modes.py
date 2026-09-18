"""--reward-mode: the three reward modes pay exactly the terms they claim to.

Runs three envs -- one per mode -- in LOCKSTEP on the same seed and the same
scripted actions, through a real game the engine itself ends. Because the
simulation is deterministic per seed, the three see identical games, and each
step's rewards can be compared directly:

  1. dynamics identical: same observation, same done flag, same winner in
     every mode (the reward mode must not change the game);
  2. decomposition: shaping_and_terminal == shaping_only + terminal_only,
     exactly, at every step;
  3. terminal_only pays 0 on every step but the last, and +1/-1 there,
     matching the winner;
  4. shaping_only pays no terminal term (its last step is an ordinary
     tile-share delta), and its running sum is exactly the change in AGENT's
     tile count over the map's land-tile count, at every prefix -- i.e. it
     really is the tile-share potential delta and nothing else.

Plus the failure paths: an unknown mode is rejected by the Python env, by
argparse, and by the Node server, and a server that ignores the field (an
older env-bridge) is caught by the reset echo check rather than silently
paying the default reward.

A win (+1) is not produced here -- a scripted agent cannot reliably win a
game -- but +1 and -1 come from the same `winner === "AGENT" ? 1 : -1`
expression the -1 path exercises.

usage: python tests/test_reward_modes.py   (exit 1 on any failure; a few minutes)
"""

import subprocess
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

import numpy as np

from envs.openfront_env import REWARD_MODES, SELF_TILES_IDX, OpenFrontEnv
from train import SMALL_MAPS

EXPAND = 1
NOOP = 0
MAX_STEPS = 4000

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")


def make(mode, seed_prefix):
    return OpenFrontEnv(map_name="onion", map_pool=SMALL_MAPS, seed=seed_prefix,
                        difficulty="easy", ticks_per_step=10, max_steps=20000, reward_mode=mode)


def failure_paths():
    print("failure paths")
    try:
        OpenFrontEnv(reward_mode="terminal-only")
        check("Python env rejects an unknown mode", False)
    except ValueError as e:
        check("Python env rejects an unknown mode", True, str(e)[:70])

    r = subprocess.run([sys.executable, str(TRAINING / "train.py"), "--reward-mode", "win_loss"],
                       capture_output=True, text=True, cwd=str(TRAINING))
    check("argparse rejects an unknown mode", r.returncode == 2 and "invalid choice" in r.stderr,
          r.stderr.strip().splitlines()[-1][:90] if r.stderr else "")

    env = make("shaping_only", "reward-test-bad")
    env.reward_mode = "bogus"  # bypass Python validation to reach the server
    try:
        env.reset(seed=1)
        check("Node server rejects an unknown mode", False)
    except RuntimeError as e:
        check("Node server rejects an unknown mode", "unknown rewardMode" in str(e), str(e)[:90])
    finally:
        env.close()

    env = make("terminal_only", "reward-test-stale")
    real_send = env._send

    def stale_send(cmd):  # an env-bridge that predates rewardMode: no echo
        reply = real_send(cmd)
        reply.pop("rewardMode", None)
        return reply

    env._send = stale_send
    try:
        env.reset(seed=1)
        check("a server that ignores the mode is caught at reset", False)
    except RuntimeError as e:
        check("a server that ignores the mode is caught at reset", "not the requested" in str(e), str(e)[:90])
    finally:
        env.close()


def lockstep(label, seed_prefix, seed, script):
    print(f"\nlockstep: {label}")
    envs = {m: make(m, seed_prefix) for m in REWARD_MODES}
    obs0 = {m: e.reset(seed=seed) for m, e in envs.items()}
    maps = {e.map_name for e in envs.values()}
    check("all three envs on the same map", len(maps) == 1, f"{maps}")
    same_start = all(np.array_equal(obs0[m][0]["tile_grid"], obs0[REWARD_MODES[0]][0]["tile_grid"])
                     for m in REWARD_MODES)
    check("identical starting observation", same_start)

    rewards = {m: [] for m in REWARD_MODES}
    tiles = [float(obs0[REWARD_MODES[0]][0]["self_features"][SELF_TILES_IDX])]
    infos = obs0[REWARD_MODES[0]][1]
    diverged = None
    ended = False
    winner = None
    for t in range(MAX_STEPS):
        a = script(t, infos["action_mask"])
        out = {m: e.step([a, 0, 0]) for m, e in envs.items()}
        ref = out[REWARD_MODES[0]]
        for m in REWARD_MODES:
            o, r, term, trunc, info = out[m]
            rewards[m].append(r)
            if diverged is None and (
                not np.array_equal(o["tile_grid"], ref[0]["tile_grid"])
                or not np.array_equal(o["self_features"], ref[0]["self_features"])
                or term != ref[2] or info["winner"] != ref[4]["winner"]
            ):
                diverged = (t, m)
        tiles.append(float(ref[0]["self_features"][SELF_TILES_IDX]))
        infos = ref[4]
        if ref[2]:
            ended, winner = True, ref[4]["winner"]
            break
    for e in envs.values():
        e.close()

    n = len(rewards[REWARD_MODES[0]])
    full = np.array(rewards["shaping_and_terminal"])
    shp = np.array(rewards["shaping_only"])
    ter = np.array(rewards["terminal_only"])
    print(f"  ({n} steps, {'engine ended it, winner=' + str(winner) if ended else 'hit the test cap without an ending'}, "
          f"AGENT tiles {tiles[0]:.0f} -> max {max(tiles):.0f} -> {tiles[-1]:.0f})")

    check("dynamics identical in all modes (obs, done, winner)", diverged is None, f"first divergence {diverged}")
    check("shaping_and_terminal == shaping_only + terminal_only, exactly, every step",
          bool(np.all(full == shp + ter)), f"max |diff| {np.abs(full - (shp + ter)).max():.3e}")
    # Running shaping sum == (tiles_t - tiles_0) / L for the map's land-tile
    # count L, at EVERY prefix. Fit L from the largest change, require it to
    # be a whole number, then check every prefix against it. A game where
    # AGENT's tile count never changes (e.g. it holds its spawn until an
    # opponent's 80% ends the game) must pay exactly zero shaping instead.
    cum = np.cumsum(shp)
    dtiles = np.array(tiles[1:]) - tiles[0]
    if not np.any(dtiles):
        check("tile count never changed, so shaping_only paid exactly 0 every step",
              bool(np.all(shp == 0)), f"{np.count_nonzero(shp)} nonzero steps")
    else:
        k = int(np.argmax(np.abs(dtiles)))
        L = dtiles[k] / cum[k] if cum[k] != 0 else float("nan")
        whole = bool(np.isfinite(L)) and abs(L - round(L)) < 1e-3
        check("implied land-tile count is a whole number", whole, f"L = {L:.4f}")
        if whole:
            err = np.abs(cum - dtiles / round(L)).max()
            check("running shaping sum == tile change / land tiles, at every step", err < 1e-9,
                  f"max error {err:.2e}, {np.count_nonzero(shp)} nonzero steps")

    if ended:
        want = 1.0 if winner == "AGENT" else -1.0
        check("terminal_only pays 0 on every step before the last", bool(np.all(ter[:-1] == 0)),
              f"{np.count_nonzero(ter[:-1])} nonzero")
        check(f"terminal_only pays {want:+.0f} on the last step (winner={winner})", ter[-1] == want, f"got {ter[-1]}")
        check("shaping_only's last step has no terminal term", abs(shp[-1]) < 0.5, f"got {shp[-1]:+.6f}")
        check("shaping_and_terminal's last step is that delta plus the terminal term",
              full[-1] == shp[-1] + want, f"{full[-1]:+.6f}")
        check("totals: shaping_and_terminal = shaping_only + terminal_only",
              abs(full.sum() - (shp.sum() + ter.sum())) < 1e-9,
              f"{full.sum():+.5f} = {shp.sum():+.5f} + {ter.sum():+.0f}")
    else:
        check("terminal_only pays exactly 0 everywhere when nothing ended", bool(np.all(ter == 0)))
        print("  (terminal-step checks skipped: no engine ending within the test cap)")


def main():
    failure_paths()
    # Seeds reproduced from greedy evals whose all-noop agent was overrun in
    # 6-9k ticks, so the engine ends the game (winner=OPPONENT) quickly.
    lockstep("all noop, overrun at spawn", "eval-2526", 2526, lambda t, mask: NOOP)
    lockstep("expand for 40 steps, then noop", "eval-2466", 2466,
             lambda t, mask: EXPAND if t < 40 and mask[EXPAND] else NOOP)
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILURE(S)'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
