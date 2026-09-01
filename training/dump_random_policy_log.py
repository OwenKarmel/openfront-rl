"""One-off: run the Phase-2 stack (gymnasium env + subprocess env-bridge)
with a random policy and dump the resulting turn log for inspection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from envs.openfront_env import OpenFrontEnv

OUT_PATH = "/tmp/random-policy-record.json"


def main() -> None:
    rng = np.random.default_rng(0)
    env = OpenFrontEnv(
        map_name="plains",
        seed="random-policy-log",
        max_steps=30,
        ticks_per_step=10,
        dump_record=OUT_PATH,
    )
    try:
        obs, info = env.reset(seed=1)
        print(f"reset: tile_grid={obs['tile_grid'].shape} self_tiles={obs['self_tiles']}")
        total_reward = 0.0
        for i in range(30):
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(
                f"step {i}: action={['noop','expand'][action]} reward={reward:.4f} "
                f"self_tiles={obs['self_tiles'][0]:.0f} opp_tiles={obs['opp_tiles'][0]:.0f} "
                f"ticks={info['ticks']} terminated={terminated} truncated={truncated}"
            )
            if terminated or truncated:
                break
        print(f"\ntotal_reward={total_reward:.4f}")
    finally:
        env.close()  # flushes the turn-log record even if the episode didn't terminate


if __name__ == "__main__":
    main()
