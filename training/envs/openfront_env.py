"""Gymnasium wrapper around the env-bridge Node process (EnvServer.ts).

Speaks the newline-delimited JSON protocol documented at the top of
env-bridge/src/EnvServer.ts over a persistent subprocess: one `reset` line
per episode, one `step` line per decision.

Single-agent: this env controls "AGENT". The opponent, "OPPONENT", is a real
built-in Nation-AI (NationExecution -- the same code driving Nation bots in
production games) at `difficulty`, making its own decisions every tick --
there is nothing to steer it with from Python. Curriculum is just "which
difficulty this episode uses"; self-play against a frozen copy of the
agent's own network is a separate, later mode (Phase 3+), not this env.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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

# Must match ACTIONS in EnvServer.ts.
ACTIONS = ["noop", "expand", "attack_opponent"]
DIFFICULTIES = ["easy", "medium", "hard", "impossible"]
AGENT_ID = "AGENT"
OPPONENT_ID = "OPPONENT"


class OpenFrontEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str = "onion",
        seed: str = "train",
        difficulty: str = "medium",
        ticks_per_step: int = 10,
        # A safety cap against a genuine stalemate hanging forever (e.g. an
        # AGENT that never attacks and an OPPONENT too weak/passive to reach
        # it), NOT meant to be hit in normal play -- real episodes should end
        # via Episode.isDone() (AGENT or OPPONENT eliminated), not truncation.
        # A too-small cap was previously cutting games short before either
        # side actually won or lost.
        max_steps: int = 20000,
        node_bin: str = "node",
        dump_game_record_dir: str | None = None,
    ) -> None:
        super().__init__()
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}")
        self.map_name = map_name
        self.default_seed = seed
        self.difficulty = difficulty
        self.ticks_per_step = ticks_per_step
        self.max_steps = max_steps
        self._node_bin = node_bin
        self.dump_game_record_dir = dump_game_record_dir

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

    def _write(self, cmd: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(cmd) + "\n")
        self._proc.stdin.flush()

    def _read_reply(self) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
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

    def _send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        self._write(cmd)
        return self._read_reply()

    def step_send(self, action: int) -> None:
        """Write half of step() only -- lets a caller (VecEnv) dispatch a
        step to every env's Node subprocess before blocking on any of their
        replies, so the ticks_per_step simulation ticks across N envs run
        concurrently (separate OS processes/cores) instead of one at a
        time. Must be paired with a later step_recv()."""
        self._write({"cmd": "step", "action": ACTIONS[action]})

    def step_recv(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Read half of step() -- see step_send()."""
        raw = self._read_reply()
        self._step_count += 1
        self._last_obs = raw
        obs = self._to_gym_obs(raw)
        reward = raw["reward"]
        terminated = bool(raw["done"])
        truncated = self._step_count >= self.max_steps
        info = {
            "legal_actions": raw["legalActions"],
            "action_mask": self._legal_action_mask(raw["legalActions"]),
            "ticks": raw["info"]["ticks"],
            "winner": raw["info"]["winner"],
        }
        return obs, reward, terminated, truncated, info

    def _to_gym_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
        w, h = raw["obs"]["width"], raw["obs"]["height"]
        # Wire format is base64 of raw Uint8Array bytes, not a JSON number
        # array -- see EnvServer.ts's tileGrid()/observation() comment.
        # frombuffer + a single b64decode call is far cheaper than parsing
        # tens/hundreds of thousands of individual JSON tokens per step.
        grid_bytes = base64.b64decode(raw["obs"]["tileGridB64"])
        grid = np.frombuffer(grid_bytes, dtype=np.uint8).reshape(h, w).astype(np.int8)
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

    def _legal_action_mask(self, legal_actions: list[str]) -> np.ndarray:
        return np.array([a in legal_actions for a in ACTIONS], dtype=bool)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ensure_process()
        episode_seed = f"{self.default_seed}-{seed}" if seed is not None else self.default_seed
        reset_cmd: dict[str, Any] = {
            "cmd": "reset",
            "seed": episode_seed,
            "map": self.map_name,
            "difficulty": self.difficulty,
            "ticksPerStep": self.ticks_per_step,
        }
        if self.dump_game_record_dir is not None:
            reset_cmd["dumpGameRecordDir"] = self.dump_game_record_dir
        raw = self._send(reset_cmd)
        self._step_count = 0
        self._last_obs = raw
        return self._to_gym_obs(raw), {
            "legal_actions": raw["legalActions"],
            "action_mask": self._legal_action_mask(raw["legalActions"]),
        }

    def step(self, action: int):
        assert self._last_obs is not None, "call reset() before step()"
        self.step_send(action)
        return self.step_recv()

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
