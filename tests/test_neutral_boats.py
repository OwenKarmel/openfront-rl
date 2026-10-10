"""OPENFRONT_NEUTRAL_BOATS=1 lets boat_attack land on unowned land.

Land attacks stop at water, so without this AGENT can only ever grow on the
landmass it spawned on (or by boating into an opponent). fourislands is the
clean case: four equal islands, so a spawn's landmass is 25% of the land and
the 80% domination threshold is unreachable without crossing water.

Same seed, two env workers:
  off -- OPENFRONT_NEUTRAL_BOATS unset: boat_target_mask covers opponent land only
  on  -- set: it also covers every cell with unowned land
then, with the flag on, AGENT expands until it owns shore, sends one boat to
a cell holding only unowned land on another island, and waits for it to land.

Also checks the reset reply's envFlags echo, and that OpenFrontEnv refuses a
worker whose flags no longer match the environment.

usage: python tests/test_neutral_boats.py   (exit 1 on any failure)
"""

import os
import sys
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

from envs.openfront_env import ACTIONS, MACRO_GRID, OpenFrontEnv

MAP = "fourislands"
SEED = 0
BOAT = ACTIONS.index("boat_attack")
EXPAND = ACTIONS.index("expand")
NOOP = ACTIONS.index("noop")
SETTLE_STEPS = 50        # the opponent holds no land yet at reset
MAX_EXPAND_STEPS = 400   # to reach shore from an inland spawn
MAX_SAIL_STEPS = 600     # for the boat to cross and land


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    return ok


def make_env(neutral_boats: bool) -> OpenFrontEnv:
    # Read by the Node worker at startup, which inherits this environment.
    os.environ["OPENFRONT_BOAT_MASK"] = "1"
    if neutral_boats:
        os.environ["OPENFRONT_NEUTRAL_BOATS"] = "1"
    else:
        os.environ.pop("OPENFRONT_NEUTRAL_BOATS", None)
    return OpenFrontEnv(map_name=MAP, seed="neutral-boats", difficulty="easy", max_steps=5000)


def cell_slices(grid: np.ndarray, cell: int) -> tuple[slice, slice]:
    """The native-grid rectangle a macro-cell covers: exactly the tiles
    tileGridAndBoatMask() assigns to it (tile x is in cell floor(x*G/w), so
    cell mx is [ceil(mx*w/G), ceil((mx+1)*w/G))) -- the bounds
    resolveBoatTarget() scans with the flag on."""
    h, w = grid.shape
    my, mx = divmod(cell, MACRO_GRID)
    ceil = lambda a, b: -(-a // b)  # noqa: E731
    return (slice(ceil(my * h, MACRO_GRID), ceil((my + 1) * h, MACRO_GRID)),
            slice(ceil(mx * w, MACRO_GRID), ceil((mx + 1) * w, MACRO_GRID)))


def landmass_of(grid: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Every land tile 4-connected over land to `seeds` -- what land attacks
    can reach. Iterative dilation; fine at 1500x1500."""
    land = grid != 3
    reach = seeds & land
    while True:
        grown = reach.copy()
        grown[1:] |= reach[:-1]
        grown[:-1] |= reach[1:]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= land
        if (grown == reach).all():
            return reach
        reach = grown


def main() -> None:
    fails = 0

    print(f"mask, flag off vs on (same seed, {SETTLE_STEPS} noop steps in so the opponent holds land)")
    env = make_env(neutral_boats=False)
    obs_off, _ = env.reset(seed=SEED)
    flags_off = env._last_obs.get("envFlags")
    for _ in range(SETTLE_STEPS):
        obs_off, *_ = env.step([NOOP, 0, 0])
    env.close()
    env = make_env(neutral_boats=True)
    obs, info = env.reset(seed=SEED)
    flags_on = env._last_obs.get("envFlags")
    for _ in range(SETTLE_STEPS):
        obs, _, _, _, info = env.step([NOOP, 0, 0])
    grid = obs["tile_grid"]
    mask_off = obs_off["boat_target_mask"].reshape(-1).astype(bool)
    mask_on = obs["boat_target_mask"].reshape(-1).astype(bool)

    fails += not check("reset echoes envFlags.neutralBoats", flags_off.get("neutralBoats") is False
                       and flags_on.get("neutralBoats") is True, f"off={flags_off} on={flags_on}")
    fails += not check("same seed and actions, same state", (obs_off["tile_grid"] == grid).all())
    off_ok = mask_off.any() and all((grid[cell_slices(grid, c)] == 2).any() for c in np.flatnonzero(mask_off))
    fails += not check("flag off: some legal cells, every one holding opponent land", off_ok,
                       f"{mask_off.sum()} cells")
    on_ok = all(np.isin(grid[cell_slices(grid, c)], (0, 2)).any() for c in np.flatnonzero(mask_on))
    fails += not check("flag on: every legal cell holds unowned or opponent land", on_ok, f"{mask_on.sum()} cells")
    fails += not check("flag on: a superset of flag off, and larger", (mask_on >= mask_off).all()
                       and mask_on.sum() > mask_off.sum(), f"{mask_off.sum()} -> {mask_on.sum()}")

    print("\na boat to another island's unowned land")
    for step in range(MAX_EXPAND_STEPS):
        if info["action_mask"][BOAT]:
            break
        obs, _, term, trunc, info = env.step([EXPAND, 0, 0])
    boat_legal = bool(info["action_mask"][BOAT])
    fails += not check("expanding reaches shore, making boat_attack legal", boat_legal, f"after {step} steps")
    if not boat_legal:
        env.close()
        sys.exit(1)

    grid = obs["tile_grid"]
    home = landmass_of(grid, grid == 1)
    ys, xs = np.nonzero(grid == 1)
    centre = np.array([ys.mean(), xs.mean()])
    candidates = []
    for c in np.flatnonzero(obs["boat_target_mask"].reshape(-1)):
        rect = grid[cell_slices(grid, c)]
        rect_home = home[cell_slices(grid, c)]
        # Only unowned land, none of it on AGENT's own landmass -- so the
        # one representative tile resolveBoatTarget() picks is off-island.
        if (rect == 0).any() and not (rect == 2).any() and not (rect_home & (rect != 3)).any():
            sl = cell_slices(grid, c)
            mid = np.array([(sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2])
            candidates.append((np.linalg.norm(mid - centre), c))
    fails += not check("some legal cell holds only off-island unowned land", bool(candidates),
                       f"{len(candidates)} candidates")
    if not candidates:
        env.close()
        sys.exit(1)
    _, target = min(candidates)

    troops_before = float(obs["self_features"][1])
    obs, _, term, trunc, info = env.step([BOAT, int(target), 0])
    troops_after = float(obs["self_features"][1])
    fails += not check("the boat launched (troops left with it)", troops_after < 0.95 * troops_before,
                       f"{troops_before:.0f} -> {troops_after:.0f}")

    landed_at = None
    for step in range(MAX_SAIL_STEPS):
        if ((obs["tile_grid"] == 1) & ~home).any():
            landed_at = step
            break
        if term or trunc:
            break
        obs, _, term, trunc, info = env.step([NOOP, 0, 0])
    off_island = int(((obs["tile_grid"] == 1) & ~home).sum())
    fails += not check("AGENT now owns land off its spawn landmass", landed_at is not None,
                       f"{off_island} tiles after {landed_at} steps")

    print("\nstale worker")
    os.environ.pop("OPENFRONT_NEUTRAL_BOATS")
    try:
        env.reset(seed=SEED + 1)
        refused = False
    except RuntimeError as e:
        refused = "OPENFRONT_NEUTRAL_BOATS" in str(e)
    fails += not check("reset refuses a worker spawned with a different flag", refused)
    env.close()

    print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
