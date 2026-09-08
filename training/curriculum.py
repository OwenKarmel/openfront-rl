"""Windowed win-rate curriculum scheduler.

Tracks the agent's recent win rate against the current difficulty and
advances through OpenFrontEnv's DIFFICULTIES ladder -- Easy -> Medium ->
Hard -> Impossible -- accordingly. This is deliberately just the
scheduling policy: it doesn't run episodes or train anything, which keeps
it independently testable without standing up an environment.

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

Fed from eval (greedy/deterministic) episodes, not training-rollout
(stochastic/sampled) ones -- train.py calls record_episode() from
run_eval_episode()'s result, not from the stochastic episode_outcomes
collected during rollout. This matters a lot more now that promotion is
one-way: on a real run, the stochastic training win rate at medium sat
around 15-27% while the greedy eval win rate was a flat 0% across 105
medium evals -- promoting on the stochastic number alone risks locking in
a permanent promotion the actual (deployed, watched) policy hasn't earned.
window=12 and promote_threshold=0.75 (9/12) are deliberately conservative
given there's no more demotion to correct a premature promotion; the
tradeoff is a much slower feedback loop (evals run once every
--eval-every updates, not several times per update like training
episodes), accepted deliberately since training now runs indefinitely.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

DIFFICULTIES = ["easy", "medium", "hard", "impossible"]


@dataclass
class CurriculumScheduler:
    difficulties: list[str] = field(default_factory=lambda: list(DIFFICULTIES))
    window: int = 12
    promote_threshold: float = 0.75
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
        """Feed one episode's outcome in; returns "promoted"/"holding" (no demotion -- see module docstring)."""
        self._results.append(won)
        rate = self.win_rate
        if rate is None:
            return "holding"
        if rate >= self.promote_threshold and self._index < len(self.difficulties) - 1:
            self._index += 1
            self._results.clear()
            return "promoted"
        return "holding"

    def state(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "index": self._index,
            "episodes_at_level": len(self._results),
            "win_rate": self.win_rate,
        }
