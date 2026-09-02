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
from envs.openfront_env import ACTIONS, OpenFrontEnv


def main() -> None:
    rng = np.random.default_rng(0)
    env = OpenFrontEnv(
        map_name="onion", seed="smoke-py", difficulty="hard", max_steps=15, ticks_per_step=10
    )
    try:
        obs, info = env.reset(seed=1)
        print(f"reset: tile_grid={obs['tile_grid'].shape} self_tiles={obs['self_tiles']} "
              f"legal_actions={info['legal_actions']}")

        total_reward = 0.0
        boat_attacks_taken = 0
        for i in range(15):
            legal = np.flatnonzero(info["action_mask"])
            action_type = int(rng.choice(legal))
            tile_idx = 0
            if ACTIONS[action_type] == "boat_attack":
                legal_tiles = np.flatnonzero(obs["boat_target_mask"].reshape(-1))
                tile_idx = int(rng.choice(legal_tiles)) if len(legal_tiles) else 0
                boat_attacks_taken += 1
            obs, reward, terminated, truncated, info = env.step([action_type, tile_idx])
            total_reward += reward
            print(
                f"step {i}: action={ACTIONS[action_type]} tile_idx={tile_idx} reward={reward:.4f} "
                f"self_tiles={obs['self_tiles'][0]:.0f} opp_tiles={obs['opp_tiles'][0]:.0f} "
                f"ticks={info['ticks']} terminated={terminated} truncated={truncated}"
            )
            if terminated or truncated:
                break

        print(f"boat_attack sampled {boat_attacks_taken} time(s) in this run.")

        print(f"\nSmoke test passed. total_reward={total_reward:.4f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
