"""Demo: CurriculumScheduler driving OpenFrontEnv's difficulty across
several episodes of a random (legal-action-masked) policy.

This is a mechanism demo, not a training result: a random policy has no
reason to beat even Easy reliably, so the expected (and correct) outcome
here is the scheduler staying at/near "easy" rather than climbing the
ladder -- promotions only happen once Phase 3's real policy can actually
win consistently. What this proves is that reset(difficulty=...) and the
scheduler's promote/demote bookkeeping work end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from curriculum import CurriculumScheduler
from envs.openfront_env import OpenFrontEnv

NUM_EPISODES = 12


def main() -> None:
    rng = np.random.default_rng(0)
    scheduler = CurriculumScheduler(window=4, promote_threshold=0.6, demote_threshold=0.2)

    for episode in range(NUM_EPISODES):
        difficulty = scheduler.difficulty
        env = OpenFrontEnv(
            map_name="onion",
            seed=f"curriculum-{episode}",
            difficulty=difficulty,
            max_steps=40,
            ticks_per_step=15,
        )
        try:
            obs, info = env.reset(seed=episode)
            terminated = truncated = False
            while not (terminated or truncated):
                legal = np.flatnonzero(info["action_mask"])
                action = int(rng.choice(legal))
                obs, reward, terminated, truncated, info = env.step(action)
            won = info["winner"] == "AGENT"
        finally:
            env.close()

        outcome = scheduler.record_episode(won)
        state = scheduler.state()
        print(
            f"episode {episode:2d} vs {difficulty:10s}: "
            f"{'WIN ' if won else 'loss'} (ticks={info['ticks']:4d}) -> "
            f"{outcome:9s} win_rate={state['win_rate']} next={scheduler.difficulty}"
        )

    print(f"\nFinal state: {scheduler.state()}")


if __name__ == "__main__":
    main()
