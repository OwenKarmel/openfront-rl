"""One-off: run the env-bridge stack (gymnasium env + subprocess) with a
random policy against a real Nation-AI opponent, and dump a full GameRecord
-- watchable in the real OpenFrontIO client (ReplayServer.ts) and
verifiable headlessly with OpenFrontIO's own `npm run replay:game`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from envs.openfront_env import ACTIONS, OpenFrontEnv

OUT_DIR = "/home/developer/openfront-rl/training/replays"


def main() -> None:
    rng = np.random.default_rng(0)
    env = OpenFrontEnv(
        map_name="onion",
        seed="random-policy-log",
        difficulty="hard",
        max_steps=30,
        ticks_per_step=10,
        dump_game_record_dir=OUT_DIR,
    )
    try:
        obs, info = env.reset(seed=1)
        print(f"reset: tile_grid={obs['tile_grid'].shape} self_tiles={obs['self_tiles']}")
        total_reward = 0.0
        for i in range(30):
            legal = np.flatnonzero(info["action_mask"])
            action = int(rng.choice(legal))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(
                f"step {i}: action={ACTIONS[action]} reward={reward:.4f} "
                f"self_tiles={obs['self_tiles'][0]:.0f} opp_tiles={obs['opp_tiles'][0]:.0f} "
                f"ticks={info['ticks']} terminated={terminated} truncated={truncated}"
            )
            if terminated or truncated:
                break
        print(f"\ntotal_reward={total_reward:.4f}")
    finally:
        env.close()  # flushes the GameRecord dump even if the episode didn't terminate


if __name__ == "__main__":
    main()
