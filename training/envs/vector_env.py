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
        # One counter per env slot, strided by self.n so no two (env, episode)
        # pairs across the vector ever collide on a seed. Must keep advancing
        # on every reset -- including auto-resets in step() -- otherwise each
        # slot replays a single fixed seed for its entire lifetime (same
        # terrain/opponent spawn every episode; the agent only ever sees `n`
        # distinct scenarios across the whole run instead of a fresh one each
        # episode).
        self._seed_counters = list(range(self.n))

    def reset(self) -> tuple[list[dict], list[dict]]:
        obs, infos = [], []
        for i, env in enumerate(self.envs):
            o, info = env.reset(seed=self._seed_counters[i])
            obs.append(o)
            infos.append(info)
        return obs, infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """actions: (n,) int array. Auto-resets any env that just terminated/truncated.

        Dispatches all N envs' step commands before blocking on any reply
        (step_send/step_recv, see OpenFrontEnv) so their ticks_per_step
        simulation ticks -- each env is a separate Node subprocess -- run
        concurrently on separate cores instead of one env's whole round
        trip finishing before the next one is even asked to start. This was
        a real, visible bottleneck: nvtop showed long flat stretches
        between GPU forward/backward pulses, matching this loop running
        envs strictly sequentially.
        """
        for env, action in zip(self.envs, actions):
            env.step_send(int(action))

        obs, rewards, terms, truncs, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step_recv()
            if term or trunc:
                # `winner`/final `ticks` from the just-finished episode are
                # worth keeping (e.g. for win-rate logging), so preserve them
                # under `episode_info` -- but `obs`/`legal_actions`/
                # `action_mask` must describe the FRESH episode about to be
                # acted on next, since that's what the training loop picks
                # the next action from.
                episode_info = info
                self._seed_counters[i] += self.n
                o, info = env.reset(seed=self._seed_counters[i])
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
