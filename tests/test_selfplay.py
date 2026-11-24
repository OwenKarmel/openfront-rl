"""Self-play opponents mixed with the curriculum's expert (Nation-AI) ones.

Pure parts (no env):
  - parse_expert_frac: one number or one entry per ladder stage, rejections;
  - OpponentPool: FIFO eviction, latest_prob 0/1, an evicted key still plays,
    save/load round trip;
  - OpponentChooser: expert_frac 1 is all-Nation and draws no randomness,
    0 is all-policy, and the fraction follows the curriculum stage.

Env-backed (opponentKind "policy" through a real env-bridge):
  - the two seats' observations mirror each other exactly at reset and after
    play (tile classes 1<->2 swapped, self/opponent features crossed);
  - random legal play by both seats ends (an elimination, or the engine's own
    80% win) with the right `winner`, and the dumped GameRecord lists both humans and
    replays IN SYNC under OpenFrontIO's own `npm run replay:game`;
  - the step-protocol guards: a self-play step without opponent_action and a
    Nation-AI step with one both refuse; the same env goes back to a
    Nation-AI episode whose info carries no opponent fields.

usage: python tests/test_selfplay.py   (exit 1 on any failure)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

from curriculum import LADDERS
from envs.openfront_env import OpenFrontEnv
from models.network import ActorCritic
from selfplay import LATEST, OpponentChooser, OpponentPool, parse_expert_frac

OPENFRONTIO = TRAINING.parent / "OpenFrontIO"
MAP = "caucasus"      # one connected landmass: the two seats meet overland
MAX_STEPS = 6000


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    return ok


def mirrored(a_obs, b_obs) -> bool:
    """b's observation is a's seen from the other seat."""
    g = a_obs["tile_grid"]
    swapped = np.where(g == 1, 2, np.where(g == 2, 1, g))
    a_self, b_self = a_obs["self_features"], b_obs["self_features"]
    # OPPONENT_FEATURES = (alive, tiles, troops, troops_ratio, gold, shares_border)
    return (
        np.array_equal(swapped, b_obs["tile_grid"])
        and np.allclose(a_self, b_obs["opponent_features"][0][1:5])
        and np.allclose(b_self, a_obs["opponent_features"][0][1:5])
        and a_obs["opponent_features"][0][5] == b_obs["opponent_features"][0][5]  # shares_border is symmetric
    )


def random_action(rng, mask, obs):
    t = int(rng.choice(np.flatnonzero(mask)))
    cells = np.flatnonzero(obs["boat_target_mask"].ravel())
    return [t, int(rng.choice(cells)) if len(cells) else 0, 0]


class FakeEnv:
    opponent_kind = None


def pure_checks() -> int:
    fails = 0
    print("parse_expert_frac")
    stages = len(LADDERS["connected-first"])
    fails += not check("one number fills every stage", parse_expert_frac("0.5", stages) == [0.5] * stages)
    fails += not check("per-stage list", parse_expert_frac("1,1,.75,.5,.25", stages) == [1, 1, 0.75, 0.5, 0.25])
    for bad in ("1,0.5", "1.5", "-0.1", "x"):
        try:
            parse_expert_frac(bad, stages)
            fails += not check(f"rejects {bad!r}", False)
        except ValueError:
            fails += not check(f"rejects {bad!r}", True)

    print("OpponentPool")
    live = ActorCritic()
    pool = OpponentPool(make_net=ActorCritic, device=torch.device("cpu"), max_size=3, latest_prob=0.0)
    rng = np.random.default_rng(0)
    fails += not check("empty pool always plays the live net", all(pool.sample(rng) == LATEST for _ in range(50)))
    for u in (10, 20, 30, 40):
        with torch.no_grad():
            for p in live.parameters():
                p.fill_(u)
        pool.add(u, live)
    fails += not check("FIFO eviction keeps the newest max_size", pool.keys() == [20, 30, 40], str(pool.keys()))
    draws = {pool.sample(rng) for _ in range(300)}
    fails += not check("latest_prob 0 draws only snapshots, all of them", draws == {20, 30, 40}, str(draws))
    pool.latest_prob = 1.0
    fails += not check("latest_prob 1 draws only the live net", all(pool.sample(rng) == LATEST for _ in range(50)))
    first = next(pool.net(30, live).parameters())
    fails += not check("a snapshot plays its own frozen weights", bool((first == 30).all()) and not first.requires_grad)
    fails += not check("LATEST is the live net itself", pool.net(LATEST, live) is live)
    evicted = pool.net(10, live)
    fails += not check("an evicted key falls back to the oldest snapshot",
                       bool((next(evicted.parameters()) == 20).all()))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "pool.pt"
        pool.save(path)
        other = OpponentPool(make_net=ActorCritic, device=torch.device("cpu"), max_size=3)
        other.load(path)
        fails += not check("save/load round trip", other.keys() == pool.keys()
                           and bool((next(other.net(40, live).parameters()) == 40).all()))

    print("OpponentChooser")
    stage = [0]
    chooser = OpponentChooser([1.0, 0.0], lambda: stage[0], pool, num_slots=2, rng=np.random.default_rng(1))
    state_before = chooser.rng.bit_generator.state
    env = FakeEnv()
    kinds = []
    for _ in range(20):
        chooser(0, env)
        kinds.append(env.opponent_kind)
    fails += not check("expert_frac 1: every episode is the Nation AI", set(kinds) == {"nation"}
                       and chooser.slot_keys[0] is None)
    fails += not check("expert_frac 1 consumes no randomness", chooser.rng.bit_generator.state == state_before)
    stage[0] = 1  # a promotion changes the mix at the next reset
    chooser(1, env)
    fails += not check("expert_frac 0 (next stage): a policy opponent", env.opponent_kind == "policy"
                       and chooser.slot_keys[1] is not None)
    chooser.expert_fracs = [0.25, 0.25]
    n_nation = 0
    for _ in range(2000):
        chooser(0, env)
        n_nation += env.opponent_kind == "nation"
    fails += not check("expert_frac 0.25 gives ~25% Nation episodes", abs(n_nation / 2000 - 0.25) < 0.04,
                       f"{n_nation / 2000:.3f}")
    return fails


def env_checks() -> int:
    fails = 0
    print("self-play episode through the env-bridge")
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as replays:
        env = OpenFrontEnv(map_name=MAP, seed="test-selfplay", difficulty="easy", opponent_kind="policy",
                           max_steps=MAX_STEPS)
        try:
            env.dump_game_record_dir = replays
            obs, info = env.reset(seed=0)
            fails += not check("reset carries the opponent seat", info["opponent_kind"] == "policy"
                               and "opponent_obs" in info and "opponent_action_mask" in info)
            fails += not check("seats mirror each other at reset", mirrored(obs, info["opponent_obs"]))
            try:
                env.step([0, 0, 0])
                fails += not check("self-play step without opponent_action refuses", False)
            except ValueError:
                fails += not check("self-play step without opponent_action refuses", True)

            mirror_ok, term, trunc, step = True, False, False, 0
            while not (term or trunc):
                a = random_action(rng, info["action_mask"], obs)
                b = random_action(rng, info["opponent_action_mask"], info["opponent_obs"])
                obs, _, term, trunc, info = env.step(a, b)
                step += 1
                if step % 100 == 0:
                    mirror_ok &= mirrored(obs, info["opponent_obs"])
            fails += not check("seats mirror each other throughout play", mirror_ok)
            agent_tiles = obs["self_features"][0]
            opp_tiles = info["opponent_obs"]["self_features"][0]
            # Two ways to end: an elimination (Episode.isDone()'s self-play
            # rule) or the engine's own 80%-of-land win with both alive.
            fails += not check("episode ends by the engine or an elimination, not truncated",
                               term and not trunc, f"step {step}, tiles {agent_tiles:.0f} vs {opp_tiles:.0f}")
            expected = "AGENT" if agent_tiles > opp_tiles else "OPPONENT"
            fails += not check("winner is the survivor / the larger player", info["winner"] == expected,
                               f"{info['winner']} vs {expected} (tiles {agent_tiles:.0f} vs {opp_tiles:.0f})")

            env.dump_game_record_dir = None
            env.opponent_kind = "nation"
            obs, info = env.reset(seed=1)
            fails += not check("same env back to a Nation-AI episode, no opponent fields",
                               info["opponent_kind"] == "nation" and "opponent_obs" not in info)
            try:
                env.step([0, 0, 0], [0, 0, 0])
                fails += not check("Nation-AI step with opponent_action refuses", False)
            except ValueError:
                fails += not check("Nation-AI step with opponent_action refuses", True)
        finally:
            env.close()

        records = sorted(Path(replays).glob("*.json"))
        fails += not check("one GameRecord dumped", len(records) == 1, str([r.name for r in records]))
        if records:
            import json
            players = [p["clientID"] for p in json.loads(records[0].read_text())["info"]["players"]]
            fails += not check("record lists both humans", players == ["AGENTAAA", "OPPONENT"], str(players))
            out = subprocess.run(["npm", "run", "-s", "replay:game", "--", str(records[0])],
                                 cwd=OPENFRONTIO, capture_output=True, text=True, timeout=900)
            text = out.stdout + out.stderr
            fails += not check("replays IN SYNC under OpenFrontIO's replay:game", "IN SYNC" in text,
                               "" if "IN SYNC" in text else text[-800:])
    return fails


def main() -> None:
    fails = pure_checks() + env_checks()
    print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
