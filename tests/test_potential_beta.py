"""--potential-beta: the convex shaping potential pays exactly what it claims.

Phi(p) = (exp(beta*p) - 1)/beta over AGENT's tile share p, so a tile gained
(or lost) at share p is worth exp(beta*p) times the same tile at spawn. This
runs three envs in LOCKSTEP on one seed and one scripted policy -- beta = 0,
2 and 5 -- through a game long enough for p to move a long way, and checks:

  1. beta does not change the game: identical observation, done flag and
     winner in every arm, at every step;
  2. beta = 0 is bit-identical to an env built without the parameter at all
     (the default really is the old reward);
  3. for beta > 0, the reward at every step equals Phi(p') - Phi(p) exactly,
     with p read off the engine's own tile counts -- not merely
     "bigger at high p", the actual closed form;
  4. the property the parameter exists for: the beta/linear reward ratio at
     high territory is exp(beta * dp) times the ratio at low territory, and
     it holds for LOSSES as symmetrically as for gains;
  5. Phi is monotone, so shaping_only keeps its ordering guarantee (the
     undiscounted return still telescopes to Phi(final) - Phi(start)).

Plus the failure paths: a negative or non-finite beta is rejected by the
Python env, by argparse and by the Node server, and a server that ignores
the field (an older env-bridge) is caught by the reset echo rather than
silently paying linear shaping.

usage: python tests/test_potential_beta.py   (exit 1 on any failure; a few minutes)
"""

import math
import subprocess
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

import numpy as np

from envs.openfront_env import SELF_TILES_IDX, OpenFrontEnv
from train import SMALL_MAPS

EXPAND = 1
NOOP = 0
MAX_STEPS = 1200
BETAS = [0.0, 2.0, 5.0]

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")


def phi(p, beta):
    return p if beta == 0 else np.expm1(beta * p) / beta


def make(seed_prefix, **kw):
    return OpenFrontEnv(map_name="onion", map_pool=SMALL_MAPS, seed=seed_prefix,
                        difficulty="easy", ticks_per_step=10, max_steps=20000,
                        reward_mode="shaping_only", **kw)


def failure_paths():
    print("failure paths")
    for bad in (-1.0, float("nan"), float("inf")):
        try:
            OpenFrontEnv(potential_beta=bad)
            check(f"Python env rejects potential_beta={bad}", False)
        except ValueError as e:
            check(f"Python env rejects potential_beta={bad}", True, str(e)[:60])

    r = subprocess.run([sys.executable, str(TRAINING / "train.py"), "--potential-beta", "-2"],
                       capture_output=True, text=True, cwd=str(TRAINING))
    check("train.py rejects a negative --potential-beta", r.returncode != 0,
          (r.stdout + r.stderr).strip().splitlines()[-1][:90])

    env = make("beta-test-bad", potential_beta=1.0)
    env.potential_beta = -3.0  # bypass Python validation to reach the server
    try:
        env.reset(seed=1)
        check("Node server rejects a negative potentialBeta", False)
    except RuntimeError as e:
        check("Node server rejects a negative potentialBeta", "potentialBeta must be" in str(e), str(e)[:80])
    finally:
        env.close()

    env = make("beta-test-stale", potential_beta=2.0)
    real_send = env._send

    def stale_send(cmd):  # an env-bridge that predates potentialBeta: no echo
        reply = real_send(cmd)
        reply.pop("potentialBeta", None)
        return reply

    env._send = stale_send
    try:
        env.reset(seed=1)
        check("a server that ignores potentialBeta is caught at reset", False)
    except RuntimeError as e:
        check("a server that ignores potentialBeta is caught at reset",
              "potentialBeta" in str(e), str(e)[:80])
    finally:
        env.close()


def policy_actions():
    """Actions sampled from a FIXED archived checkpoint, so the lockstep visits
    the high-territory, tile-LOSING states that a scripted `expand` cannot:
    `expand` only takes neutral land, so it stalls at ~9% once the spawn
    island is full, and never attacks.

    Deliberately NOT checkpoints/latest.pt: that moves with whatever run is
    live, so the states this arm covers would silently change (a freshly
    cold-started policy reaches neither high territory nor sustained losses,
    and the checks would still pass while covering nothing). Optional --
    skipped if the archive is absent, since the closed-form checks do not
    depend on it."""
    ckpt = (TRAINING / "checkpoints" / "archive"
            / "run_2026-09-17_gamma-anneal098-0999_shaping-only" / "latest.pt")
    if not ckpt.exists():
        return None
    import torch

    from models.network import ActorCritic
    from train import net_inputs, obs_to_batch

    net = ActorCritic()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    net.eval()
    torch.manual_seed(0)

    def act(obs, info):
        # Sampled once from the reference arm and replayed into every arm, so
        # all arms still see the identical action sequence.
        inp = net_inputs(obs_to_batch([obs]), info["action_mask"][None], torch.device("cpu"))
        return net.act(*inp)[0][0].tolist()

    return act


def lockstep(label, seed, prefix, policy, max_steps):
    print(f"\nlockstep: {label}, seed {seed}")
    global fails
    envs = {b: make(prefix, potential_beta=b) for b in BETAS}
    envs["default"] = make(prefix)  # no potential_beta argument at all
    obs0 = {k: e.reset(seed=seed) for k, e in envs.items()}
    check("identical starting observation in every arm",
          all(np.array_equal(obs0[k][0]["tile_grid"], obs0[0.0][0]["tile_grid"]) for k in envs))

    rew = {k: [] for k in envs}
    tiles = [float(obs0[0.0][0]["self_features"][SELF_TILES_IDX])]
    infos = obs0[0.0][1]
    diverged = None
    for t in range(max_steps):
        a = policy(obs0[0.0][0] if t == 0 else ref[0], infos)
        out = {k: e.step(a) for k, e in envs.items()}
        ref = out[0.0]
        for k in envs:
            o, r, term, trunc, info = out[k]
            rew[k].append(r)
            if diverged is None and (
                not np.array_equal(o["tile_grid"], ref[0]["tile_grid"])
                or term != ref[2] or info["winner"] != ref[4]["winner"]
            ):
                diverged = (t, k)
        tiles.append(float(ref[0]["self_features"][SELF_TILES_IDX]))
        infos = ref[4]
        if ref[2] or ref[3]:
            break
    for e in envs.values():
        e.close()

    lin = np.array(rew[0.0])
    tiles = np.array(tiles)
    print(f"  ({len(lin)} steps, tiles {tiles[0]:.0f} -> max {tiles.max():.0f} -> {tiles[-1]:.0f})")

    check("beta does not change the game (obs, done, winner)", diverged is None, f"first divergence {diverged}")
    check("beta=0 is bit-identical to an env built without the parameter",
          bool(np.all(lin == np.array(rew["default"]))),
          f"max |diff| {np.abs(lin - np.array(rew['default'])).max():.3e}")

    # Absolute tile share needs the map's land-tile count, which the wire
    # format does not carry; recover it from the linear arm, where the reward
    # IS the tile delta over that count, then require it to be a whole number.
    dtiles = tiles[1:] - tiles[0]
    cum = np.cumsum(lin)
    k = int(np.argmax(np.abs(dtiles)))
    L = dtiles[k] / cum[k]
    check("implied land-tile count is a whole number", abs(L - round(L)) < 1e-3, f"L = {L:.4f}")
    L = round(L)
    p = tiles / L
    check("recovered tile share matches the linear arm at every step",
          float(np.abs(np.cumsum(lin) - (p[1:] - p[0])).max()) < 1e-9)

    for b in BETAS[1:]:
        want = phi(p[1:], b) - phi(p[:-1], b)
        got = np.array(rew[b])
        err = float(np.abs(got - want).max())
        check(f"beta={b}: reward == Phi(p') - Phi(p) at every step", err < 1e-12,
              f"max error {err:.2e} over {len(got)} steps, p in [{p.min():.3f}, {p.max():.3f}]")

        # The point of the parameter, measured on real steps: same-sized tile
        # change, weighted by where it happened. Compare the realized
        # beta/linear ratio against exp(beta*p) at the midpoint of the step.
        moved = np.abs(lin) > 0
        ratio = got[moved] / lin[moved]
        mid = (p[:-1] + p[1:])[moved] / 2
        check(f"beta={b}: per-tile weight tracks exp(beta*p)",
              float(np.abs(ratio - np.exp(b * mid)).max()) < 1e-3,
              f"weight {ratio.min():.2f} at p={mid[ratio.argmin()]:.3f} -> "
              f"{ratio.max():.2f} at p={mid[ratio.argmax()]:.3f}")

        losses = moved & (lin < 0)
        if losses.any():
            rl = got[losses] / lin[losses]
            ml = (p[:-1] + p[1:])[losses] / 2
            check(f"beta={b}: losses are weighted by the same exp(beta*p)",
                  float(np.abs(rl - np.exp(b * ml)).max()) < 1e-3,
                  f"{losses.sum()} losing steps, weight up to {rl.max():.2f}")
        else:
            print(f"  (beta={b}: no losing step in this episode; gain path checked above)")

        check(f"beta={b}: Phi is monotone, so shaping_only keeps its ordering",
              bool(np.all(np.diff(phi(np.linspace(0, 1, 10001), b)) > 0)))
        check(f"beta={b}: undiscounted return telescopes to Phi(final) - Phi(start)",
              abs(got.sum() - (phi(p[-1], b) - phi(p[0], b))) < 1e-12,
              f"{got.sum():+.6f} vs {phi(p[-1], b) - phi(p[0], b):+.6f}")

    check("beta=0 matches the closed form too (Phi(p) = p)",
          float(np.abs(lin - (p[1:] - p[:-1])).max()) < 1e-12)
    check("Phi'(0) = 1 for every beta (spawn weight is never reduced)",
          all(abs((phi(1e-9, b) / 1e-9) - 1.0) < 1e-6 for b in BETAS),
          f"{[round(phi(1e-9, b) / 1e-9, 9) for b in BETAS]}")


def main():
    failure_paths()
    # seed 7022 reached 52% territory under the trained policy; scripted
    # expand-always on the same map stalls at ~9%.
    lockstep("expand-always", 7022, "probe-7022",
             lambda obs, info: [EXPAND if info["action_mask"][EXPAND] else NOOP, 0, 0], 1200)
    act = policy_actions()
    if act is None:
        print("\n  SKIP: archived checkpoint absent -- high-territory and tile-loss "
              "steps not exercised")
    else:
        # This one rises to ~37% and is then eaten back down to 0, so it
        # covers both the high-p weighting and the symmetric loss weighting.
        lockstep("trained policy (sampled)", 7032, "probe-7032", act, 2000)

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILURE(S)'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
