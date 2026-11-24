"""Gymnasium wrapper around the env-bridge Node process (EnvServer.ts).

Speaks the newline-delimited JSON protocol documented at the top of
env-bridge/src/EnvServer.ts over a persistent subprocess: one `reset` line
per episode, one `step` line per decision.

This env controls "AGENT". Who plays OPPONENT is `opponent_kind`, read at
every reset:

- "nation" (default): a real built-in Nation AI (NationExecution -- the same
  code driving Nation bots in production games) at `difficulty`, making its
  own decisions every tick; nothing to steer from Python.
- "policy": self-play. OPPONENT is a second Human player; step_send() then
  needs its action too, and every reset/step info carries its observation
  and action mask (`opponent_obs`, `opponent_action_mask`), encoded from its
  own seat exactly the way AGENT's are, so the same network can play it.
  Reward and `winner` stay AGENT's. Choosing per episode between the two --
  and which network plays the policy seat -- is train.py's job (see
  selfplay.py).

Opponents arrive as a fixed-width, masked array of MAX_OPPONENTS slots
(`opponent_features` + `opponent_mask`) rather than a flat opp_tiles/
opp_troops/opp_gold triple, so the observation shape -- and therefore the
network's input shape -- is the same whether an episode has one opponent or
ten. MAX_OPPONENTS is 1 today, i.e. the same 1v1 matchup as before.
"""

from __future__ import annotations

import base64
import json
import math
import os
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
# resolve(): tsx applies the tsconfig only to files under its REAL directory,
# so a symlinked OpenFrontIO/ (a worktree sharing the main checkout's
# submodule) would otherwise compile the engine without it -- breaking its
# class-field semantics (NationNukeBehavior reads this.game before it is set).
# A no-op in a normal checkout.
OPENFRONTIO_TSCONFIG = (ENV_BRIDGE_DIR / ".." / "OpenFrontIO" / "tsconfig.json").resolve()

# Must match ACTIONS in EnvServer.ts.
ACTIONS = ["noop", "expand", "attack_opponent", "boat_attack"]
DIFFICULTIES = ["easy", "medium", "hard", "impossible"]
# Which terms make up the step reward -- mirrors EnvServer.ts's REWARD_MODES
# (see its docstring). The first entry is the default and reproduces the
# reward from before this option existed.
REWARD_MODES = ["shaping_and_terminal", "shaping_only", "terminal_only"]
# Who plays OPPONENT -- mirrors OPPONENT_KINDS in GameSetup.ts. The first
# entry is the default and is the only opponent before self-play existed.
OPPONENT_KINDS = ["nation", "policy"]

# Must match MAX_OPPONENTS in EnvServer.ts: the fixed number of opponent
# slots every opponent-indexed observation/mask/action is padded to.
MAX_OPPONENTS = 1

# Feature order within `self_features` / each row of `opponent_features` --
# must match the field order playerObs()/opponentBlock() are unpacked in
# below, and (for opponents) the input layout network.py's shared
# per-opponent MLP is trained on. Named rather than positional so a caller
# that needs one feature (e.g. tile counts for logging) can index it by
# name instead of a bare literal.
SELF_FEATURES = ("tiles", "troops", "troops_ratio", "gold")
OPPONENT_FEATURES = ("alive", "tiles", "troops", "troops_ratio", "gold", "shares_border")
SELF_TILES_IDX = SELF_FEATURES.index("tiles")
OPPONENT_TILES_IDX = OPPONENT_FEATURES.index("tiles")
# The same fields spelled the way EnvServer.ts puts them on the wire.
_WIRE_FIELD = {"troops_ratio": "troopsRatio", "shares_border": "sharesBorder"}
_SELF_WIRE_FIELDS = tuple(_WIRE_FIELD.get(f, f) for f in SELF_FEATURES)
_OPPONENT_WIRE_FIELDS = tuple(_WIRE_FIELD.get(f, f) for f in OPPONENT_FEATURES)

# Must match MACRO_GRID in EnvServer.ts. A boat_attack action's tile
# parameter is a flat index into this MACRO_GRID x MACRO_GRID grid
# (tile_idx = macroY * MACRO_GRID + macroX, matching boatTargetMacroMask's
# bit-packing order on the TS side).
MACRO_GRID = 32
BOAT_ATTACK_IDX = ACTIONS.index("boat_attack")
ATTACK_OPPONENT_IDX = ACTIONS.index("attack_opponent")
EXPAND_IDX = ACTIONS.index("expand")

# The attack-size head's choices (attack_size_head=True / train.py's
# --attack-size-head): the fraction of AGENT's CURRENT troops an expand,
# attack_opponent or boat_attack sends -- action slot 3 indexes this list,
# and the env sends the fraction itself as `troopFraction`. 0.20 is what
# every attack sent before the head existed (Config.attackAmount() /
# boatAttackAmount() = troops/5). network.py's NUM_ATTACK_FRACTIONS must
# match its length.
ATTACK_FRACTIONS = (0.10, 0.20, 0.35, 0.50, 0.75)
DEFAULT_ATTACK_FRACTION_IDX = ATTACK_FRACTIONS.index(0.20)
ATTACK_SIZED_ACTIONS = (EXPAND_IDX, ATTACK_OPPONENT_IDX, BOAT_ATTACK_IDX)

# Every map in OpenFrontIO/resources/maps/ with both width and height <=1500
# (checked against each map's manifest.json "map" dimensions) -- the other
# ~100 maps run 1500-6000px per side, which would make the resize below
# (see RESIZE_DIM) stretch a much larger native resolution down much
# further, losing more detail than these already do. Regenerate by walking
# every manifest.json and filtering on map.width/map.height if the map pool
# ever changes.
SMALL_MAPS = [
    "baltics", "beringstrait", "blacksea", "bosphorusstraits", "caucasus",
    "danishstraits", "didier", "fingerlakes", "fourislands", "irishsea",
    "italia", "labyrinth", "morethanluck", "onion", "pangaea", "sierpinski",
    "tourney1", "tourney2", "tourney3", "tourney4", "yellowsea", "yenisei",
]

# The SMALL_MAPS where a random spawn's own landmass holds at least half the
# map's land -- i.e. where most of the game can be played without a boat.
# Land attacks stop at water, so on the rest of the pool the agent must cross
# water to grow past its spawn landmass at all. Measured with
# env-bridge/src/mapLandmasses.ts (spawnShare = expected share of the land in
# a random spawn's landmass):
#
#   yellowsea 99.3%   caucasus 97.0%   fingerlakes 80.1%   italia 66.7%   pangaea 54.3%
#
# Only the first three reach the 80% domination threshold overland. Every
# other SMALL_MAPS entry is under 50%, as low as 2.0% (labyrinth); the pool
# averages ~36%. Used by the "connected-first" curriculum ladder (see
# curriculum.py) to learn the land game before island maps are added.
CONNECTED_MAPS = ["yellowsea", "caucasus", "fingerlakes", "italia", "pangaea"]

# Named map pools a curriculum stage can select (see curriculum.py's Stage).
MAP_POOLS = {"small": SMALL_MAPS, "connected": CONNECTED_MAPS}

# Env-var gated behaviours of EnvServer.ts, keyed by the name its reset reply
# echoes them under ("envFlags"). Each is off unless the variable is exactly
# "1" when the Node worker starts.
ENV_FLAGS = {
    "OPENFRONT_EXPAND_MASK": "expandMask",
    "OPENFRONT_BOAT_MASK": "boatMask",
    "OPENFRONT_NEUTRAL_BOATS": "neutralBoats",
}


def requested_env_flags() -> dict[str, bool]:
    """Which ENV_FLAGS this process's environment turns on -- what every Node
    worker it spawns will inherit."""
    return {var: os.environ.get(var) == "1" for var in ENV_FLAGS}

# Fixed canvas every episode's tile_grid is resized to (nearest-neighbor)
# when map_pool is used. Deliberately NOT "pad with water to the pool's
# largest map" -- padding leaves onion-sized episodes just as cheap as
# today, but breaks a single shared observation_space (the real content
# would sit at a different sub-region depending on the map, and more
# importantly the whole point of a fixed canvas is one batched
# net.act()/net.evaluate_actions() call across envs on different maps,
# which needs an identical tile_grid shape regardless of which map any
# given env is currently on).
#
# Originally set to 1500 (the pool's max native size, so tourney1-4/
# fourislands needed no downscaling) -- reverted to 512 (onion's original
# size) after that setting produced confirmed OOM kills in real training
# (cgroup memory.events' oom_kill counter incremented on every crash;
# rollout_length=300 x num_envs=10 x 1500x1500 int8 tile_grids alone is
# ~6.75GB in the rollout buffer, before GPU tensors or the Node
# subprocesses' own memory). 512 matches this project's original,
# long-proven-stable memory footprint exactly (the same size every
# onion-only run before today used) at the cost of downscaling every
# larger pool map (most of them, up to ~3x per side for the 1500x1500
# ones) rather than preserving native resolution.
RESIZE_DIM = 512


def _resize_nearest(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resize for a categorical (tile-class) 2D array --
    not cv2/PIL interpolation, which would invent fractional/invalid class
    values between e.g. "land" and "water". Works for both up- and
    downscaling; dependency-free (plain numpy fancy indexing)."""
    h, w = grid.shape
    row_idx = np.arange(out_h) * h // out_h
    col_idx = np.arange(out_w) * w // out_w
    return grid[row_idx[:, None], col_idx]


class OpenFrontEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str = "onion",
        map_pool: list[str] | None = None,
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
        recycle_every: int = 25,
        reward_mode: str = REWARD_MODES[0],
        # Convexity of the shaping potential: Phi(p) = (exp(beta*p)-1)/beta
        # for beta > 0, Phi(p) = p at beta == 0 (the original linear
        # tile-share delta, bit-identical). See shapePotential() in
        # EnvServer.ts -- the server owns the arithmetic; this is only the
        # wire field and its validation.
        potential_beta: float = 0.0,
        # See the module docstring. Like difficulty and map_pool, a change
        # takes effect at the next reset().
        opponent_kind: str = OPPONENT_KINDS[0],
        attack_size_head: bool = False,
    ) -> None:
        super().__init__()
        if opponent_kind not in OPPONENT_KINDS:
            raise ValueError(f"opponent_kind must be one of {OPPONENT_KINDS}, got {opponent_kind!r}")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        if not math.isfinite(potential_beta) or potential_beta < 0:
            raise ValueError(f"potential_beta must be a finite number >= 0, got {potential_beta!r}")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}")
        self.map_name = map_name
        # When set, reset() picks a fresh random map from this pool every
        # episode (independent per env -- no cross-env synchronization
        # needed, since _to_gym_obs() resizes every map to the same
        # RESIZE_DIM canvas regardless of which one was picked) instead of
        # always using map_name.
        self.map_pool = map_pool
        self.default_seed = seed
        self.difficulty = difficulty
        self.ticks_per_step = ticks_per_step
        self.max_steps = max_steps
        self._node_bin = node_bin
        self.dump_game_record_dir = dump_game_record_dir
        self.reward_mode = reward_mode
        self.potential_beta = float(potential_beta)
        self.opponent_kind = opponent_kind
        # Off (default): the troopFraction field is never sent and the
        # env-bridge sizes attacks exactly as before (troops/5). On: action
        # slot 3 picks ATTACK_FRACTIONS[i] for every attack-type step.
        self.attack_size_head = bool(attack_size_head)
        # The kind of the episode actually in progress -- opponent_kind can
        # be changed for the NEXT episode while this one is still running.
        self._episode_opponent_kind: str | None = None

        # The env-bridge Node worker leaks ~2.5 MB per episode (measured:
        # a single worker grew 95 -> 157 MB over 25 resets, while its tsx
        # wrapper stayed flat). The retention is somewhere in OpenFrontIO's
        # own per-episode object graph, not in anything this project owns --
        # each episode deliberately builds a fresh GameMapImpl (see
        # resolveMap() in GameSetup.ts) and something holds onto it.
        #
        # Rather than chase that through a large third-party engine, bound it
        # here: respawn the subprocess every `recycle_every` episodes, which
        # caps a worker's growth at roughly recycle_every * 2.5 MB no matter
        # what the underlying cause is. Left this un-fixed, a real run grew
        # its 10 workers from 3.2 GB to 4.5 GB over 6.5 hours (+239 MB/hour)
        # and was OOM-killed at update 464.
        #
        # 25 keeps per-worker growth under ~60 MB and costs one ~2s Node
        # startup per 25 episodes -- at observed episode rates (~8 episodes
        # per env per hour) that is a couple of seconds every few hours,
        # far below the noise in per-update timings. Set to 0 to disable.
        self.recycle_every = recycle_every
        self._episodes_since_spawn = 0

        self._proc: subprocess.Popen | None = None
        self._step_count = 0
        self._last_obs: dict[str, Any] | None = None

        # [action_type, macro_tile_idx, player_idx] -- macro_tile_idx only
        # matters when action_type selects boat_attack, and player_idx only
        # when it selects attack_opponent (see step_send()); ignored/legal
        # for any value otherwise, same "cheap to sample, server validates
        # anyway" spirit as the existing type-level legality mask.
        self.action_space = spaces.MultiDiscrete(
            [len(ACTIONS), MACRO_GRID * MACRO_GRID, MAX_OPPONENTS, len(ATTACK_FRACTIONS)]
        )
        # Populated with real dimensions on the first reset(); Box shape must
        # be static, so it's declared for map_name's known geometry via a
        # throwaway reset here. With map_pool set, _to_gym_obs() resizes
        # every episode's tile_grid to (RESIZE_DIM, RESIZE_DIM) regardless
        # of which map was picked, so the shape this throwaway reset
        # observes is already the fixed one every future reset will match.
        obs = self.reset()[0]
        h, w = obs["tile_grid"].shape
        self.observation_space = spaces.Dict(
            {
                # 0=neutral land, 1=self, 2=ANY opponent, 3=water. Class 2
                # deliberately carries no per-opponent identity -- see
                # tileGridAndBoatMask() in EnvServer.ts.
                "tile_grid": spaces.Box(low=0, high=3, shape=(h, w), dtype=np.int8),
                # See SELF_FEATURES for the field order.
                "self_features": spaces.Box(
                    low=0, high=np.inf, shape=(len(SELF_FEATURES),), dtype=np.float32
                ),
                # One row per opponent slot, padded to MAX_OPPONENTS; see
                # OPPONENT_FEATURES for the per-row field order. Padding rows
                # are all-zero and flagged False in opponent_mask.
                "opponent_features": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(MAX_OPPONENTS, len(OPPONENT_FEATURES)),
                    dtype=np.float32,
                ),
                # 1 where the slot is a real opponent rather than padding --
                # what the network's set encoder pools over.
                "opponent_mask": spaces.Box(
                    low=0, high=1, shape=(MAX_OPPONENTS,), dtype=np.int8
                ),
                # 1 where attack_opponent may legally target that slot.
                # Unlike boat_target_mask this one is exact, not a heuristic:
                # it comes from real alive/sharesBorderWith checks, so the
                # env-bridge rejects (rather than no-ops) an unmasked pick.
                "attack_target_mask": spaces.Box(
                    low=0, high=1, shape=(MAX_OPPONENTS,), dtype=np.int8
                ),
                # Per-macro-cell boat_attack destination legality -- see
                # boatTargetMacroMask() in EnvServer.ts. Doubles as the tile
                # parameter's action mask: the network shouldn't need a
                # separate wire field to know which macro-cells are legal
                # boat targets when this is already in the observation.
                "boat_target_mask": spaces.Box(
                    low=0, high=1, shape=(MACRO_GRID, MACRO_GRID), dtype=np.int8
                ),
            }
        )

    def _recycle_process(self) -> None:
        """Tear the Node worker down so the next _ensure_process() spawns a
        fresh one, releasing everything the old one leaked. Only ever called
        between episodes (from reset(), before the new episode is started),
        so no in-flight game state is lost."""
        if self._proc is None:
            return
        # No "close" command first: that would make the worker flush a
        # GameRecord dump for the episode that just ended, which reset()
        # does not want and which close() only does because it is the real
        # end of the env's life. Just terminate.
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10)
        self._proc = None
        self._episodes_since_spawn = 0

    def _ensure_process(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._episodes_since_spawn = 0
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

    def _encode_action(self, action) -> dict[str, Any]:
        """[action_type_idx, macro_tile_idx, player_idx(, attack_frac_idx)]
        -> the wire's {"action", "macroX"/"macroY" | "playerIdx",
        "troopFraction"} (see EnvServer.ts's ActionChoice). Slot 3 is read
        only with attack_size_head and only for an attack-type action; a
        3-slot action is accepted with the head off, as before it existed."""
        type_idx = int(action[0])
        choice: dict[str, Any] = {"action": ACTIONS[type_idx]}
        if type_idx == BOAT_ATTACK_IDX:
            tile_idx = int(action[1])
            macro_y, macro_x = divmod(tile_idx, MACRO_GRID)
            choice["macroX"] = macro_x
            choice["macroY"] = macro_y
        elif type_idx == ATTACK_OPPONENT_IDX:
            choice["playerIdx"] = int(action[2])
        if self.attack_size_head and type_idx in ATTACK_SIZED_ACTIONS:
            choice["troopFraction"] = ATTACK_FRACTIONS[int(action[3])]
        return choice

    def step_send(self, action, opponent_action=None) -> None:
        """Write half of step() only -- lets a caller (VecEnv) dispatch a
        step to every env's Node subprocess before blocking on any of their
        replies, so the ticks_per_step simulation ticks across N envs run
        concurrently (separate OS processes/cores) instead of one at a
        time. Must be paired with a later step_recv().

        action: [action_type_idx, macro_tile_idx, player_idx] (see
        action_space). opponent_action: the same for OPPONENT -- required in
        a self-play ("policy") episode, and must be None otherwise."""
        is_selfplay = self._episode_opponent_kind == "policy"
        if is_selfplay != (opponent_action is not None):
            raise ValueError(
                f"opponent_action is {'required' if is_selfplay else 'not allowed'} in a "
                f"{self._episode_opponent_kind!r}-opponent episode"
            )
        cmd: dict[str, Any] = {"cmd": "step", **self._encode_action(action)}
        if opponent_action is not None:
            cmd["opponentAction"] = self._encode_action(opponent_action)
        self._write(cmd)

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
            "opponent_kind": self._episode_opponent_kind,
            **self._opponent_info(raw),
        }
        return obs, reward, terminated, truncated, info

    def _opponent_info(self, raw: dict[str, Any]) -> dict[str, Any]:
        """OPPONENT's own-seat observation and action mask, in a self-play
        episode; nothing otherwise."""
        opp = raw.get("opponent")
        if opp is None:
            return {}
        return {
            "opponent_obs": self._to_gym_obs(opp),
            "opponent_action_mask": self._legal_action_mask(opp["legalActions"]),
        }

    def _to_gym_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
        w, h = raw["obs"]["width"], raw["obs"]["height"]
        # Wire format is base64 of raw Uint8Array bytes, not a JSON number
        # array -- see EnvServer.ts's tileGrid()/observation() comment.
        # frombuffer + a single b64decode call is far cheaper than parsing
        # tens/hundreds of thousands of individual JSON tokens per step.
        grid_bytes = base64.b64decode(raw["obs"]["tileGridB64"])
        grid = np.frombuffer(grid_bytes, dtype=np.uint8).reshape(h, w).astype(np.int8)
        if self.map_pool:
            grid = _resize_nearest(grid, RESIZE_DIM, RESIZE_DIM)
        # Packed bits, LSB-first within each byte (matches EnvServer.ts's
        # `out[bitIdx >> 3] |= 1 << (bitIdx & 7)`); bitIdx = macroY*MACRO_GRID
        # + macroX, so a row-major reshape lines up with (macroY, macroX).
        boat_mask_bytes = base64.b64decode(raw["obs"]["boatTargetMacroMaskB64"])
        boat_mask = (
            np.unpackbits(np.frombuffer(boat_mask_bytes, dtype=np.uint8), bitorder="little")[
                : MACRO_GRID * MACRO_GRID
            ]
            .reshape(MACRO_GRID, MACRO_GRID)
            .astype(np.int8)
        )
        obs = raw["obs"]
        self_features = np.array(
            [obs["self"][f] for f in _SELF_WIRE_FIELDS], dtype=np.float32
        )
        opponent_features = np.array(
            [[o[f] for f in _OPPONENT_WIRE_FIELDS] for o in obs["opponents"]],
            dtype=np.float32,
        )
        return {
            "tile_grid": grid,
            "self_features": self_features,
            "opponent_features": opponent_features,
            "opponent_mask": np.array(obs["opponentMask"], dtype=np.int8),
            "attack_target_mask": np.array(obs["attackTargetMask"], dtype=np.int8),
            "boat_target_mask": boat_mask,
        }

    def _legal_action_mask(self, legal_actions: list[str]) -> np.ndarray:
        return np.array([a in legal_actions for a in ACTIONS], dtype=bool)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Bound the worker's per-episode leak (see recycle_every in __init__).
        # Checked before _ensure_process() so the respawn and the new
        # episode's startup are the same round trip.
        if self.recycle_every and self._episodes_since_spawn >= self.recycle_every:
            self._recycle_process()
        self._ensure_process()
        self._episodes_since_spawn += 1
        episode_seed = f"{self.default_seed}-{seed}" if seed is not None else self.default_seed
        # Picked independently per env, per episode -- safe because
        # _to_gym_obs() resizes every map to the same fixed canvas, so
        # different envs (or the same env across episodes) being on
        # different maps never produces a batching-shape mismatch.
        if self.map_pool:
            self.map_name = str(self.np_random.choice(self.map_pool))
        reset_cmd: dict[str, Any] = {
            "cmd": "reset",
            "seed": episode_seed,
            "map": self.map_name,
            "difficulty": self.difficulty,
            "ticksPerStep": self.ticks_per_step,
            "rewardMode": self.reward_mode,
            "potentialBeta": self.potential_beta,
        }
        # Sent only when it is not the default, so a Nation-AI reset is the
        # same command it was before self-play existed.
        if self.opponent_kind != OPPONENT_KINDS[0]:
            reset_cmd["opponentKind"] = self.opponent_kind
        if self.attack_size_head:
            # Sent only when on, so a default reset is byte-identical to one
            # from before the option existed.
            reset_cmd["attackSizeHead"] = True
        if self.dump_game_record_dir is not None:
            reset_cmd["dumpGameRecordDir"] = self.dump_game_record_dir
        raw = self._send(reset_cmd)
        if raw.get("rewardMode") != self.reward_mode:
            # An env-bridge that predates rewardMode ignores the field and pays
            # the default reward -- which would be indistinguishable from the
            # requested one in every log. Fail instead.
            raise RuntimeError(
                f"env-bridge is using reward mode {raw.get('rewardMode')!r}, "
                f"not the requested {self.reward_mode!r}"
            )
        if raw.get("potentialBeta") != self.potential_beta:
            # Same trap as rewardMode: an env-bridge that predates
            # potentialBeta silently pays the linear shaping instead.
            raise RuntimeError(
                f"env-bridge is using potentialBeta {raw.get('potentialBeta')!r}, "
                f"not the requested {self.potential_beta!r}"
            )
        # Same trap for self-play: an env-bridge that predates opponentKind
        # would quietly play the Nation AI instead.
        if raw.get("opponentKind", OPPONENT_KINDS[0]) != self.opponent_kind:
            raise RuntimeError(
                f"env-bridge is using opponentKind {raw.get('opponentKind')!r}, "
                f"not the requested {self.opponent_kind!r}"
            )
        # Same trap again for the env-var gated behaviours: a worker that
        # predates a flag ignores the variable, and one spawned before the
        # variable changed is running the old setting. Either way the run
        # would not be what its launcher says.
        if bool(raw.get("attackSizeHead", False)) != self.attack_size_head:
            raise RuntimeError(
                f"env-bridge reports attackSizeHead={raw.get('attackSizeHead')!r}, but this env "
                f"was built with attack_size_head={self.attack_size_head}"
                + (" (it predates the option)" if "attackSizeHead" not in raw else "")
            )
        server_flags = raw.get("envFlags")
        for var, wanted in requested_env_flags().items():
            running = None if server_flags is None else server_flags.get(ENV_FLAGS[var])
            if bool(running) != wanted:
                raise RuntimeError(
                    f"{var} is {'1' if wanted else 'unset'} here, but the env-bridge reports "
                    f"{ENV_FLAGS[var]}={running!r}"
                    + (" (it predates envFlags)" if server_flags is None else "")
                )
        self._step_count = 0
        self._last_obs = raw
        self._episode_opponent_kind = self.opponent_kind
        return self._to_gym_obs(raw), {
            "legal_actions": raw["legalActions"],
            "action_mask": self._legal_action_mask(raw["legalActions"]),
            "opponent_kind": self._episode_opponent_kind,
            **self._opponent_info(raw),
        }

    def step(self, action, opponent_action=None):
        assert self._last_obs is not None, "call reset() before step()"
        self.step_send(action, opponent_action)
        return self.step_recv()

    def close(self):
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                    self._proc.stdin.flush()
                # "close" makes the worker write any pending GameRecord dump
                # and exit by itself; terminating it straight away could kill
                # it mid-write and lose the dump.
                self._proc.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                pass
            if self._proc.poll() is None:
                self._proc.terminate()
            self._proc = None
