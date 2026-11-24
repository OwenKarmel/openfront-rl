"""Self-play opponents, mixed with the curriculum's expert (Nation-AI) ones.

Every training episode's OPPONENT is chosen at that episode's reset:

- with probability `expert_frac` of the current curriculum stage, the
  **expert**: the built-in Nation AI at the stage's difficulty -- exactly the
  opponent every run before self-play trained against;
- otherwise a **policy** opponent: a second Human player steered from Python
  (env-bridge opponentKind "policy") by a copy of the agent's own network,
  drawn from an OpponentPool -- the live network with probability
  `latest_prob`, else a uniformly chosen frozen snapshot of it.

The map always comes from the stage's pool, for both kinds. Only AGENT's seat
is trained on: the opponent's transitions are never added to the rollout
buffer, so the PPO update is the same on-policy update as before, against a
mix of opponents. Eval suites and curriculum promotion are unchanged: they
are always played against the Nation AI, so "stage" keeps meaning "beats
<difficulty> on <maps>" and stays comparable with every earlier run.

Why a pool rather than only the live network: an opponent that is always the
current policy moves every update, and a policy can cycle (A beats B beats C
beats A) without ever getting better against anything fixed. Older snapshots
keep the strategies it already beat in the mix. Why keep the expert at all:
the Nation AI is the target, and a pool seeded from a weak policy has nothing
to teach early on -- per-stage `expert_frac` lets a ladder start almost
all-expert and shift toward self-play as the agent climbs (see
parse_expert_frac()).

Rewards are AGENT's, with the run's own --reward-mode/--potential-beta, in
both kinds of episode. A self-play episode ends at the first elimination (see
Episode.isDone() in GameSetup.ts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# Pool key for "whatever the live network is right now" -- it is not a
# snapshot, so it is never frozen and never stored.
LATEST = -1

POOL_FILENAME = "selfplay_pool.pt"


def parse_expert_frac(spec: str, num_stages: int) -> list[float]:
    """--expert-frac -> one probability per curriculum stage.

    Either one number for every stage ("0.5") or a comma-separated list with
    exactly one entry per stage of the ladder ("1,1,0.75,0.5,0.5"). 1 means
    every episode is against the Nation AI -- the behaviour from before
    self-play existed -- and 0 means none is."""
    parts = [p.strip() for p in spec.split(",")]
    try:
        fracs = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"--expert-frac {spec!r}: expected a number or a comma-separated list of them")
    if len(fracs) == 1:
        fracs = fracs * num_stages
    if len(fracs) != num_stages:
        raise ValueError(
            f"--expert-frac {spec!r} has {len(fracs)} entries; the curriculum ladder has {num_stages} stages"
        )
    if not all(0.0 <= f <= 1.0 for f in fracs):
        raise ValueError(f"--expert-frac {spec!r}: every entry must lie in [0, 1]")
    return fracs


@dataclass
class OpponentPool:
    """Frozen snapshots of the learner, keyed by the update they were taken at.

    Snapshots are kept as CPU state_dicts (the whole network is ~0.2-0.5M
    parameters, so a full pool is tens of MB) and materialised as eval-mode
    networks on `device` the first time they are asked to act. FIFO: once
    `max_size` is reached the oldest snapshot is dropped."""

    make_net: Callable[[], torch.nn.Module]
    device: torch.device
    max_size: int = 20
    latest_prob: float = 0.5
    _snapshots: dict[int, dict] = field(default_factory=dict)
    _nets: dict[int, torch.nn.Module] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_size < 1:
            raise ValueError("pool max_size must be at least 1")
        if not 0.0 <= self.latest_prob <= 1.0:
            raise ValueError("latest_prob must lie in [0, 1]")

    def __len__(self) -> int:
        return len(self._snapshots)

    def keys(self) -> list[int]:
        return sorted(self._snapshots)

    def add(self, update: int, net: torch.nn.Module) -> None:
        self._snapshots[update] = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        self._nets.pop(update, None)
        while len(self._snapshots) > self.max_size:
            oldest = min(self._snapshots)
            del self._snapshots[oldest]
            self._nets.pop(oldest, None)

    def sample(self, rng: np.random.Generator) -> int:
        """LATEST with probability latest_prob (always, while the pool is
        empty), else a uniformly drawn snapshot key."""
        if not self._snapshots or rng.random() < self.latest_prob:
            return LATEST
        keys = self.keys()
        return keys[int(rng.integers(len(keys)))]

    def net(self, key: int, live_net: torch.nn.Module) -> torch.nn.Module:
        """The network that plays `key`. A key that has been evicted since it
        was drawn (an episode can outlive several snapshots) keeps its cached
        net if it still has one, and otherwise falls back to the oldest
        snapshot still held -- the nearest thing to what was drawn."""
        if key == LATEST:
            return live_net
        if key in self._nets:
            return self._nets[key]
        if key not in self._snapshots:
            key = min(self._snapshots) if self._snapshots else LATEST
            if key == LATEST:
                return live_net
            if key in self._nets:
                return self._nets[key]
        net = self.make_net().to(self.device)
        net.load_state_dict(self._snapshots[key])
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self._nets[key] = net
        return net

    def prune_cache(self, in_use: set[int]) -> None:
        """Drop materialised nets for evicted snapshots no episode still uses."""
        for key in list(self._nets):
            if key not in self._snapshots and key not in in_use:
                del self._nets[key]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"snapshots": self._snapshots, "max_size": self.max_size}, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self._snapshots = dict(state["snapshots"])
        self._nets.clear()
        while len(self._snapshots) > self.max_size:
            del self._snapshots[min(self._snapshots)]


class OpponentChooser:
    """Decides, at each env slot's reset, who that episode's OPPONENT is.

    Installed as VecEnv's before_reset hook. `stage_index` is read at every
    reset rather than captured, so a curriculum promotion changes the mix for
    every episode that starts after it (like the difficulty and map pool)."""

    def __init__(
        self,
        expert_fracs: list[float],
        stage_index: Callable[[], int],
        pool: OpponentPool,
        num_slots: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.expert_fracs = expert_fracs
        self.stage_index = stage_index
        self.pool = pool
        self.rng = rng if rng is not None else np.random.default_rng()
        # The pool key playing each slot's current episode; None for a
        # Nation-AI episode.
        self.slot_keys: list[int | None] = [None] * num_slots

    @property
    def expert_frac(self) -> float:
        return self.expert_fracs[self.stage_index()]

    def __call__(self, slot: int, env) -> None:
        frac = self.expert_frac
        # frac == 1 draws nothing, so an all-expert run consumes no
        # randomness and builds the exact same episodes as before.
        if frac >= 1.0 or self.rng.random() < frac:
            env.opponent_kind = "nation"
            self.slot_keys[slot] = None
        else:
            env.opponent_kind = "policy"
            self.slot_keys[slot] = self.pool.sample(self.rng)

    def in_use(self) -> set[int]:
        return {k for k in self.slot_keys if k is not None}


def opponent_actions(
    infos: list[dict],
    chooser: OpponentChooser,
    live_net: torch.nn.Module,
    obs_to_batch: Callable,
    net_inputs: Callable,
    device: torch.device,
) -> list[np.ndarray | None]:
    """OPPONENT's action for every self-play slot, None for the rest.

    One batched forward per distinct network in play this step (the live one
    and each snapshot drawn), each over its slots' own-seat observations. The
    opponent samples its actions exactly like AGENT does -- the policy, not
    its argmax (see run_eval_suite() in train.py for why argmax is a different,
    much worse policy here)."""
    out: list[np.ndarray | None] = [None] * len(infos)
    groups: dict[int, list[int]] = {}
    for i, info in enumerate(infos):
        if info.get("opponent_kind") == "policy":
            key = chooser.slot_keys[i]
            groups.setdefault(LATEST if key is None else key, []).append(i)
    for key, slots in groups.items():
        net = chooser.pool.net(key, live_net)
        batch = obs_to_batch([infos[i]["opponent_obs"] for i in slots])
        mask = np.stack([infos[i]["opponent_action_mask"] for i in slots])
        actions, _, _ = net.act(*net_inputs(batch, mask, device))
        for i, a in zip(slots, actions.cpu().numpy()):
            out[i] = a
    return out

