"""Curriculum scheduler: a ladder of (difficulty, map pool) stages, promoted
on the win rate of a sampled-policy eval suite.

This is deliberately just the scheduling policy: it doesn't run episodes or
train anything, which keeps it independently testable without standing up
an environment. train.py's run_eval_suite() produces the numbers it is fed.

Monotonic (promotion only, no demotion): an earlier version demoted back
down once the trailing win rate fell to <=20%, on a catastrophic-forgetting
rationale. In practice this produced a persistent oscillation, not a
safety net -- promotion is gated on mastery of the *current* (easier) tier,
not readiness for the next one, so crossing the promote threshold reliably
triggered an initial losing streak against the harder opponent, which then
hit the demote threshold before enough medium-specific experience had
accumulated to actually close the gap; demoting cleared the window and
dropped back to a difficulty the policy hadn't forgotten (weights aren't
reset on demotion), so it re-promoted quickly on unchanged skill and
repeated. Measured on a real run: 25 easy<->medium round-trips over ~2,500
updates, 73% of that time spent back at easy, hard never reached even
once. Removing demotion trades the (theoretical, not observed here)
catastrophic-forgetting risk for guaranteed forward progress and lets a
harder tier accumulate the sustained time-on-task it needs to actually be
learnable.

What promotion is gated on, and why it changed twice:

1. Stochastic training-rollout outcomes. Dropped because training episodes
   are whatever happened to finish inside a rollout -- a population that
   shifts with the rollout length and the map mix, not a fixed test.
2. Greedy (argmax) eval episodes over a rolling window of 12. Dropped
   because the greedy policy is a different policy from the one being
   trained: argmax turns the near-50/50 noop/expand split the policy holds
   at spawn into "never act" (34/78 greedy evals in the 2026-09-13 gamma
   anneal never acted after spawning). Greedy eval won 3/198 there while
   the sampled policy won ~24% of training episodes, so the 9/12 gate could
   never open and every run since has trained against easy only.
3. Now: one eval SUITE per eval -- N episodes (train.py's --eval-episodes,
   default 64) played by the sampled policy, the one training optimizes,
   on a FIXED set of seeds, so every suite plays the same maps, spawns and
   opponents and successive suites differ only in the policy. At 64
   episodes the win rate's standard error near the 0.75 threshold is ~5pp,
   against ~12pp for the old 12-episode window.

Stages also choose the map pool, so a ladder can hold back maps whose
difficulty is a different skill rather than a stronger opponent -- see
LADDERS["connected-first"].
"""

from __future__ import annotations

from dataclasses import dataclass

DIFFICULTIES = ["easy", "medium", "hard", "impossible"]


@dataclass(frozen=True)
class Stage:
    difficulty: str
    # Key into openfront_env.MAP_POOLS, or None for whatever the run was
    # launched with (--map if given, else SMALL_MAPS) -- the behaviour from
    # before stages had a map pool at all.
    maps: str | None = None


LADDERS: dict[str, list[Stage]] = {
    # The original ladder: difficulty only, map pool left to the launcher.
    "difficulty": [Stage(d) for d in DIFFICULTIES],
    # Learn the land game first, on the five maps where a spawn's own
    # landmass is at least half the map (CONNECTED_MAPS), then add the island
    # maps, then raise the difficulty. On most of SMALL_MAPS land attacks run
    # out of map long before the 80% domination threshold, so growing at all
    # needs a boat -- a separate skill. One variable changes per promotion,
    # never two at once, so a regression after a promotion is attributable.
    "connected-first": [
        Stage("easy", "connected"),
        Stage("easy", "small"),
        Stage("medium", "small"),
        Stage("hard", "small"),
        Stage("impossible", "small"),
    ],
}


@dataclass
class CurriculumScheduler:
    ladder: str = "difficulty"
    promote_threshold: float = 0.75
    start_index: int = 0

    def __post_init__(self) -> None:
        if self.ladder not in LADDERS:
            raise ValueError(f"unknown curriculum ladder {self.ladder!r}: expected one of {sorted(LADDERS)}")
        self.stages = LADDERS[self.ladder]
        if not 0 <= self.start_index < len(self.stages):
            raise ValueError(f"start_index {self.start_index} outside ladder {self.ladder!r}")
        self._index = self.start_index
        # (wins, episodes) of the latest suite at the CURRENT stage; cleared on
        # promotion, so it never describes a stage the agent has left.
        self.last_eval: tuple[int, int] | None = None

    @property
    def stage(self) -> Stage:
        return self.stages[self._index]

    @property
    def difficulty(self) -> str:
        return self.stage.difficulty

    @property
    def maps(self) -> str | None:
        return self.stage.maps

    def record_eval(self, wins: int, episodes: int) -> str:
        """Feed one eval suite's result in; returns "promoted"/"holding" (no
        demotion -- see module docstring)."""
        if episodes <= 0 or not 0 <= wins <= episodes:
            raise ValueError(f"bad eval suite result: {wins}/{episodes}")
        self.last_eval = (wins, episodes)
        if wins / episodes >= self.promote_threshold and self._index < len(self.stages) - 1:
            self._index += 1
            self.last_eval = None
            return "promoted"
        return "holding"

    def state(self) -> dict:
        return {
            "ladder": self.ladder,
            "index": self._index,
            "difficulty": self.difficulty,
            "maps": self.maps,
            "last_eval_win_rate": None if self.last_eval is None else self.last_eval[0] / self.last_eval[1],
        }
