"""Rules that composite guards follow.

Two kinds, both `Rule`s:

- `EnsembleRule` (`EnsembleGuard`): how members' results for one category become one score.
  Built in: `Votes(k)`, `AnyOf()`, `AllOf()`, `Weighted(weights, cutoff)`.
- `EscalationRule` (`CascadeGuard`): whether a guard's answer stands or the text goes to the
  next guard. Built in: `Margin(below, above)`.

Both are protocols, so your own rule is any object with the right methods; inheriting is optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from .result import CategoryResult, GuardResult


@runtime_checkable
class Rule(Protocol):
    """The family: a rule a composite guard follows. Use `EnsembleRule` or `EscalationRule`."""


@runtime_checkable
class EnsembleRule(Rule, Protocol):
    """How an EnsembleGuard turns its members' results for one category into one score.

    `members` maps each member's label (LayaGuard, JevGuard, JevGuard#2, ...) to its
    CategoryResult for that category (`.score`, `.flagged`, `.threshold`, `.by_mode`). The
    category is flagged when `score(members) >= threshold(n)`, n being the number of members.

    Optional, for `EnsembleGuard(lazy=True)`: `decided(members, n) -> bool`, True once the members
    seen so far settle the verdict whatever the remaining ones say. Then `score` is called on those
    members only and must still land on the same side of `threshold(n)`. Without it, a lazy
    ensemble runs every member.
    """

    def score(self, members: Mapping[str, CategoryResult]) -> float: ...

    def threshold(self, n: int) -> float: ...


@runtime_checkable
class EscalationRule(Rule, Protocol):
    """Whether a CascadeGuard passes the text on: True sends it to the next guard."""

    def escalate(self, result: GuardResult) -> bool: ...


# ── ensemble rules ─────────────────────────────────────────────────────────
def _flag_share(members: Mapping[str, CategoryResult]) -> float:
    return sum(bool(c.flagged) for c in members.values()) / len(members)


@dataclass(frozen=True)
class Votes(EnsembleRule):
    """Flag when at least `k` members flag (default: a majority). Score: the share flagging."""

    k: int | None = None

    def score(self, members: Mapping[str, CategoryResult]) -> float:
        return _flag_share(members)

    def threshold(self, n: int) -> float:
        k = n // 2 + 1 if self.k is None else self.k
        if not 1 <= k <= n:
            raise ValueError(f"k must be between 1 and {n}, got {k}")
        return k / n

    def decided(self, members: Mapping[str, CategoryResult], n: int) -> bool:
        k = round(self.threshold(n) * n)
        flags = sum(bool(c.flagged) for c in members.values())
        return flags >= k or flags + (n - len(members)) < k


class AnyOf(Votes):
    """Flag when any member flags."""

    def threshold(self, n: int) -> float:
        return 1 / n


class AllOf(Votes):
    """Flag only when every member flags."""

    def threshold(self, n: int) -> float:
        return 1.0


@dataclass(frozen=True)
class Weighted(EnsembleRule):
    """Flag when the flagging members carry at least `cutoff` of the total weight."""

    weights: Mapping[str, float]
    cutoff: float = 0.5

    def score(self, members: Mapping[str, CategoryResult]) -> float:
        missing = set(members) - set(self.weights)
        if missing:
            raise ValueError(f"Weighted weights must name every member; missing {sorted(missing)}")
        total = sum(self.weights[label] for label in members)
        return sum(self.weights[label] for label, c in members.items() if c.flagged) / total

    def threshold(self, n: int) -> float:
        return self.cutoff

    def decided(self, members: Mapping[str, CategoryResult], n: int) -> bool:
        total = sum(self.weights.values())
        flagged = sum(self.weights[label] for label, c in members.items() if c.flagged)
        remaining = sum(w for label, w in self.weights.items() if label not in members)
        return flagged >= self.cutoff * total or flagged + remaining < self.cutoff * total


# ── escalation rules ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class Margin(EscalationRule):
    """Escalate when the verdict sits within a margin of the guard's own threshold t.

    `Margin(0.2)` is symmetric, [t - 0.2, t + 0.2); `Margin(0.3, 0.1)` is wider below t, so
    near-misses get a second look more often than near-flags. Relative to t, one margin works
    across models that score on different scales. Per category: an unflagged result escalates
    when some category is within `below` of its threshold; a flagged one when every flagged
    category is within `above` of its threshold (with one threshold: the top score is in the band).
    """

    below: float
    above: float | None = None  # None: same as below

    def __post_init__(self) -> None:
        if self.below < 0 or (self.above is not None and self.above < 0):
            raise ValueError(f"Margin widths must be >= 0, got {self.below}, {self.above}")

    def escalate(self, result: GuardResult) -> bool:
        scored = [(c.score, c.threshold) for c in result.ranked if c.threshold is not None]
        if len(scored) != len(result.ranked):
            raise ValueError("Margin needs a guard with a threshold")
        above = self.below if self.above is None else self.above
        flagged = [(s, t) for s, t in scored if s >= t]
        if flagged:
            return all(s < t + above for s, t in flagged)
        return any(s >= t - self.below for s, t in scored)
