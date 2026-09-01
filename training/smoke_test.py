"""Phase-2 round-trip smoke test: random policy through OpenFrontEnv.

Confirms the full stack (gymnasium API -> subprocess -> Node env-bridge ->
headless OpenFrontIO simulation) works end to end before any real policy
network or PPO training loop exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from envs.openfront_env import OpenFrontEnv


def main() -> None:
    rng = np.random.default_rng(0)
    env = OpenFrontEnv(map_name="plains", seed="smoke-py", max_steps=10, ticks_per_step=10)
    try:
        obs, info = env.reset(seed=1)
        print(f"reset: tile_grid={obs['tile_grid'].shape} self_tiles={obs['self_tiles']} "
              f"legal_actions={info['legal_actions']}")

        total_reward = 0.0
        for i in range(10):
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(
                f"step {i}: action={action} reward={reward:.4f} "
                f"self_tiles={obs['self_tiles'][0]:.0f} opp_tiles={obs['opp_tiles'][0]:.0f} "
                f"ticks={info['ticks']} terminated={terminated} truncated={truncated}"
            )
            if terminated or truncated:
                break

        print(f"\nSmoke test passed. total_reward={total_reward:.4f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
