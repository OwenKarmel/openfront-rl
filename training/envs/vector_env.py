"""Minimal synchronous vector-env wrapper: N independent OpenFrontEnv
instances (each its own Node subprocess), stepped and reset in a Python
loop and batched into numpy arrays for the network.

Not OS-parallel (stepping env i+1 waits for env i's subprocess round-trip
to finish) -- true parallelism would need multiprocessing/async IPC, which
is a reasonable follow-up once this loop's throughput is actually profiled
and found wanting. What this already buys over a single env: more diverse,
less-correlated rollout data per PPO update, and auto-reset on episode end
so the training loop never has to special-case a finished env.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from envs.openfront_env import OpenFrontEnv


class VecEnv:
    def __init__(self, env_fns: list[Callable[[], OpenFrontEnv]]):
        self.envs = [fn() for fn in env_fns]
        self.n = len(self.envs)

    def reset(self) -> tuple[list[dict], list[dict]]:
        obs, infos = [], []
        for i, env in enumerate(self.envs):
            o, info = env.reset(seed=i)
            obs.append(o)
            infos.append(info)
        return obs, infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """actions: (n,) int array. Auto-resets any env that just terminated/truncated."""
        obs, rewards, terms, truncs, infos = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, info = env.step(int(action))
            if term or trunc:
                # `winner`/final `ticks` from the just-finished episode are
                # worth keeping (e.g. for win-rate logging), so preserve them
                # under `episode_info` -- but `obs`/`legal_actions`/
                # `action_mask` must describe the FRESH episode about to be
                # acted on next, since that's what the training loop picks
                # the next action from.
                episode_info = info
                o, info = env.reset(seed=None)
                info["episode_info"] = episode_info
            obs.append(o)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
        return obs, np.array(rewards, dtype=np.float32), np.array(terms), np.array(truncs), infos

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def set_difficulty(self, difficulty: str) -> None:
        for env in self.envs:
            env.difficulty = difficulty
