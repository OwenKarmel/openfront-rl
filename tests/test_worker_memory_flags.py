"""--recycle-every and --canary-envs change memory and wall time, never results.

  - make_env() passes --recycle-every to OpenFrontEnv, and a worker built
    with recycle_every=1 really respawns its Node process at the next reset
  - run_eval_suite() with num_envs=1 vs 3, and with workers recycled every
    episode vs every 25, plays the same games: same seeds, maps, winners,
    ticks and territory (a deterministic all-noop policy, short games)

usage: python tests/test_worker_memory_flags.py   (exit 1 on any failure; ~1 min)
"""

import argparse
import sys
import tempfile
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

import torch  # noqa: E402

import train  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")


class NoopPolicy:
    """Duck-types ActorCritic.act(): always noop, so every game is fixed by its seed."""

    def act(self, *inputs):
        b = inputs[-1].shape[0]
        return torch.zeros(b, 4, dtype=torch.long), None, None


def make_args(recycle_every):
    return argparse.Namespace(
        map=None, ticks_per_step=10, max_episode_steps=60, eval_max_steps=60, eval_episodes=4, eval_envs=3,
        eval_replays=0, eval_dir=tempfile.mkdtemp(), recycle_every=recycle_every,
    )


env = train.make_env(make_args(1), "hard", "recycle-check", maps="small")
try:
    first_pid = env._proc.pid
    check("make_env passes --recycle-every", env.recycle_every == 1)
    env.reset(seed=1)
    check("recycle_every=1 respawns the worker at the next reset", env._proc.pid != first_pid)
finally:
    env.close()

keys = ("seed", "map", "winner", "ticks", "truncated", "peak_territory_frac", "final_territory_frac")
runs = {}
for label, recycle, num_envs in (("3 workers, recycle 25", 25, None), ("1 worker, recycle 25", 25, 1),
                                 ("3 workers, recycle 1", 1, None)):
    eps = train.run_eval_suite(make_args(recycle), NoopPolicy(), torch.device("cpu"), "hard", "small",
                               num_envs=num_envs)
    runs[label] = [tuple(e[k] for k in keys) for e in eps]
base = runs["3 workers, recycle 25"]
check(f"{len(base)} games played, seeds {[g[0] for g in base]}", len(base) == 4)
for label, games in runs.items():
    check(f"{label}: same games as 3 workers / recycle 25", games == base)

print("ALL PASSED" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
