"""Windowed win-rate curriculum scheduler.

Tracks the agent's recent win rate against the current difficulty and
advances (or backs off) through OpenFrontEnv's DIFFICULTIES ladder --
Easy -> Medium -> Hard -> Impossible -- accordingly. This is deliberately
just the scheduling policy: it doesn't run episodes or train anything,
so it's usable as-is once Phase 3 wires up a real PPO training loop, and
independently testable/demoable before that loop exists (see
curriculum_demo.py).

Demotion exists because a fixed-difficulty curriculum risks catastrophic
forgetting: an agent promoted on a lucky streak that then can't hold Medium
should drop back rather than grind uselessly against an opponent it isn't
beating, per the self-play/league guidance in the project plan.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

DIFFICULTIES = ["easy", "medium", "hard", "impossible"]


@dataclass
class CurriculumScheduler:
    difficulties: list[str] = field(default_factory=lambda: list(DIFFICULTIES))
    window: int = 20
    promote_threshold: float = 0.6
    demote_threshold: float = 0.2
    start_index: int = 0

    def __post_init__(self) -> None:
        if not self.difficulties:
            raise ValueError("difficulties must be non-empty")
        self._index = self.start_index
        self._results: deque[bool] = deque(maxlen=self.window)

    @property
    def difficulty(self) -> str:
        return self.difficulties[self._index]

    @property
    def win_rate(self) -> float | None:
        """None until the window has enough episodes at the current level to be meaningful."""
        if len(self._results) < self.window:
            return None
        return sum(self._results) / len(self._results)

    def record_episode(self, won: bool) -> str:
        """Feed one episode's outcome in; returns "promoted"/"demoted"/"holding"."""
        self._results.append(won)
        rate = self.win_rate
        if rate is None:
            return "holding"
        if rate >= self.promote_threshold and self._index < len(self.difficulties) - 1:
            self._index += 1
            self._results.clear()
            return "promoted"
        if rate <= self.demote_threshold and self._index > 0:
            self._index -= 1
            self._results.clear()
            return "demoted"
        return "holding"

    def state(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "index": self._index,
            "episodes_at_level": len(self._results),
            "win_rate": self.win_rate,
        }
