"""train.py's eval suite: fixed seeds, sampled policy, batched across workers.

Runs run_eval_suite() twice with an untrained network and tiny settings (6
episodes on CONNECTED_MAPS, 3 workers, capped at 40 steps each) and checks:

  - one result per seed, in seed order, every map drawn from the stage's pool
  - the SAME maps both times -- the seeds are fixed, so successive suites
    differ only in the policy
  - workers are reused (6 episodes on 3 envs) and every step is one batched
    forward across the live envs, through net.act() (the sampled policy)
  - a step-capped episode is truncated, not won, and summarize_suite()'s
    win rate / Wilson CI reflect that
  - GameRecords are dumped for seeds < --eval-replays only, and a repeated
    seed repeats its gameID (why ReplayServer serves the newest match)

usage: python tests/test_eval_suite.py   (exit 1 on any failure)
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

import torch

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

from envs.openfront_env import CONNECTED_MAPS, REWARD_MODES
from models.network import ActorCritic
from train import run_eval_suite, summarize_suite

EPISODES, WORKERS, CAP, REPLAYS = 6, 3, 40, 2


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    return ok


def main() -> None:
    fails = 0
    torch.manual_seed(0)
    net = ActorCritic()
    calls = {"act": 0, "batch": []}
    act = net.act

    def counting_act(*inputs):
        calls["act"] += 1
        calls["batch"].append(inputs[0].shape[0])
        return act(*inputs)

    net.act = counting_act

    with tempfile.TemporaryDirectory() as replays:
        args = argparse.Namespace(
            map=None, ticks_per_step=10, max_episode_steps=20000,
            reward_mode=REWARD_MODES[0], potential_beta=0.0,
            eval_episodes=EPISODES, eval_envs=WORKERS, eval_max_steps=CAP,
            eval_replays=REPLAYS, eval_dir=replays,
        )
        first = run_eval_suite(args, net, torch.device("cpu"), "easy", "connected")
        acts_first, batches_first = calls["act"], list(calls["batch"])
        second = run_eval_suite(args, net, torch.device("cpu"), "easy", "connected")
        dumped = sorted(p.name for p in Path(replays).glob("*.json"))
        args.eval_episodes = 1
        run_eval_suite(args, net, torch.device("cpu"), "impossible", "connected")
        other_stage = sorted(p.name for p in Path(replays).glob("*.json") if p.name not in dumped)

    print("results")
    fails += not check("one episode per seed, in seed order", [e["seed"] for e in first] == list(range(EPISODES)))
    maps = [e["map"] for e in first]
    fails += not check("every map from the stage's pool", set(maps) <= set(CONNECTED_MAPS), str(maps))
    fails += not check("fixed seeds -> the same maps every suite", maps == [e["map"] for e in second])
    fails += not check("seeds spread across more than one map", len(set(maps)) > 1, str(sorted(set(maps))))

    print("\nexecution")
    steps = sum(e["ticks"] for e in first) // args.ticks_per_step
    fails += not check("sampled policy: net.act() drives every step", acts_first > 0, f"{acts_first} calls")
    fails += not check("batched: fewer forwards than env steps", acts_first < EPISODES * CAP,
                       f"{acts_first} forwards for ~{steps} steps")
    fails += not check("never more live envs than workers", max(batches_first) <= WORKERS,
                       f"batch sizes {sorted(set(batches_first))}")

    print("\noutcomes")
    fails += not check("capped episodes are truncated, with no winner",
                       all(e["truncated"] and e["winner"] is None for e in first))
    s = summarize_suite(first)
    fails += not check("summary: 0 wins, honest CI", s["wins"] == 0 and s["win_rate"] == 0.0
                       and s["ci_low"] == 0.0 and 0.0 < s["ci_high"] < 1.0,
                       f"{s['wins']}/{s['episodes']} CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}]")

    print("\nreplays")
    fails += not check(f"{REPLAYS} dumps per suite (seeds < --eval-replays)", len(dumped) == 2 * REPLAYS, str(dumped))
    game_ids = [re.sub(r"^.*_", "", name) for name in dumped]
    fails += not check("a repeated seed repeats its gameID", len(set(game_ids)) == REPLAYS, str(sorted(set(game_ids))))
    other_ids = {re.sub(r"^.*_", "", name) for name in other_stage}
    fails += not check("the same seed at another stage gets its own gameID",
                       len(other_stage) == 1 and not other_ids & set(game_ids), str(sorted(other_ids)))

    print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
