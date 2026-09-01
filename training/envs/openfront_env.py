"""Gymnasium wrapper around the env-bridge Node process (EnvServer.ts).

Speaks the newline-delimited JSON protocol documented at the top of
env-bridge/src/EnvServer.ts over a persistent subprocess: one `reset` line
per episode, one `step` line per decision. This is intentionally the
smallest possible wrapper needed to prove the whole loop -- Node simulation,
subprocess IPC, gymnasium API -- works, before any real network/PPO code is
written.

The env controls a single player, "AGENT". The other player, "OPPONENT", is
driven by `opponent_policy(obs) -> action:str`, defaulting to a simple
scripted "always expand" policy. Later curriculum phases swap this for a
policy that steers a real Nation/bot difficulty, or for a frozen copy of the
agent's own network (self-play).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openfront_env requires gymnasium: pip install gymnasium numpy"
    ) from exc

ENV_BRIDGE_DIR = Path(__file__).resolve().parents[2] / "env-bridge"
OPENFRONTIO_TSCONFIG = ENV_BRIDGE_DIR / ".." / "OpenFrontIO" / "tsconfig.json"

ACTIONS = ["noop", "expand"]
AGENT_ID = "AGENT"
OPPONENT_ID = "OPPONENT"


def always_expand(_obs: dict[str, Any]) -> str:
    return "expand"


class OpenFrontEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str = "plains",
        seed: str = "train",
        spawn_turns: int = 3,
        ticks_per_step: int = 10,
        max_steps: int = 200,
        opponent_policy: Callable[[dict[str, Any]], str] = always_expand,
        node_bin: str = "node",
        dump_record: str | None = None,
    ) -> None:
        super().__init__()
        self.map_name = map_name
        self.default_seed = seed
        self.spawn_turns = spawn_turns
        self.ticks_per_step = ticks_per_step
        self.max_steps = max_steps
        self.opponent_policy = opponent_policy
        self._node_bin = node_bin
        self.dump_record = dump_record

        self._proc: subprocess.Popen | None = None
        self._step_count = 0
        self._last_obs: dict[str, Any] | None = None

        self.action_space = spaces.Discrete(len(ACTIONS))
        # Populated with real dimensions on the first reset(); Box shape must
        # be static, so it's declared for map_name's known geometry via a
        # throwaway reset here.
        obs = self.reset()[0]
        h, w = obs["tile_grid"].shape
        self.observation_space = spaces.Dict(
            {
                "tile_grid": spaces.Box(low=0, high=2, shape=(h, w), dtype=np.int8),
                "self_tiles": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "self_troops": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "self_gold": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "opp_tiles": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "opp_troops": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "opp_gold": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )

    def _ensure_process(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [
                self._node_bin,
                str(ENV_BRIDGE_DIR / "node_modules" / ".bin" / "tsx"),
                "--tsconfig",
                str(OPENFRONTIO_TSCONFIG),
                str(ENV_BRIDGE_DIR / "src" / "EnvServer.ts"),
            ],
            cwd=str(ENV_BRIDGE_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )

    def _send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(cmd) + "\n")
        self._proc.stdin.flush()
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                raise RuntimeError("env-bridge process exited unexpectedly")
            line = line.strip()
            if not line:
                continue
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                continue  # stray console.log from Node, not a protocol line
            if "error" in reply:
                raise RuntimeError(f"env-bridge error: {reply['error']}")
            return reply

    def _to_gym_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
        w, h = raw["obs"]["width"], raw["obs"]["height"]
        grid = np.array(raw["obs"]["tileGrid"], dtype=np.int8).reshape(h, w)
        p = raw["obs"]["players"]
        return {
            "tile_grid": grid,
            "self_tiles": np.array([p[AGENT_ID]["tiles"]], dtype=np.float32),
            "self_troops": np.array([p[AGENT_ID]["troops"]], dtype=np.float32),
            "self_gold": np.array([p[AGENT_ID]["gold"]], dtype=np.float32),
            "opp_tiles": np.array([p[OPPONENT_ID]["tiles"]], dtype=np.float32),
            "opp_troops": np.array([p[OPPONENT_ID]["troops"]], dtype=np.float32),
            "opp_gold": np.array([p[OPPONENT_ID]["gold"]], dtype=np.float32),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ensure_process()
        episode_seed = f"{self.default_seed}-{seed}" if seed is not None else self.default_seed
        reset_cmd: dict[str, Any] = {
            "cmd": "reset",
            "seed": episode_seed,
            "map": self.map_name,
            "spawnTurns": self.spawn_turns,
            "ticksPerStep": self.ticks_per_step,
        }
        if self.dump_record is not None:
            reset_cmd["dumpRecord"] = self.dump_record
        raw = self._send(reset_cmd)
        self._step_count = 0
        self._last_obs = raw
        return self._to_gym_obs(raw), {"legal_actions": raw["legalActions"]}

    def step(self, action: int):
        assert self._last_obs is not None, "call reset() before step()"
        opp_action = self.opponent_policy(self._last_obs["obs"])
        raw = self._send(
            {
                "cmd": "step",
                "actions": {AGENT_ID: ACTIONS[action], OPPONENT_ID: opp_action},
            }
        )
        self._step_count += 1
        self._last_obs = raw
        obs = self._to_gym_obs(raw)
        reward = raw["reward"][AGENT_ID]
        terminated = bool(raw["done"])
        truncated = self._step_count >= self.max_steps
        info = {"legal_actions": raw["legalActions"], "ticks": raw["info"]["ticks"]}
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                    self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            self._proc.terminate()
            self._proc = None
