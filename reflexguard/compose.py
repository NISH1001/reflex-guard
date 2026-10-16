"""Guards made of guards: EnsembleGuard (concurrent, votes) and CascadeGuard (sequential, escalates)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .guard import Guard
from .result import CategoryResult, GuardResult


# ── ensemble rules ─────────────────────────────────────────────────────────
@runtime_checkable
class Rule(Protocol):
    """How an EnsembleGuard turns its members' results for one category into one score.

    `members` maps each member's label (LayaGuard, JevGuard, JevGuard#2, ...) to its
    CategoryResult for that category (`.score`, `.flagged`, `.threshold`, `.by_mode`). The
    category is flagged when `score(members) >= threshold(n)`, n being the number of members.
    """

    def score(self, members: Mapping[str, CategoryResult]) -> float: ...

    def threshold(self, n: int) -> float: ...


def _flag_share(members: Mapping[str, CategoryResult]) -> float:
    return sum(bool(c.flagged) for c in members.values()) / len(members)


@dataclass(frozen=True)
class Votes:
    """Flag when at least `k` members flag (default: a majority). Score: the share flagging."""

    k: int | None = None

    def score(self, members: Mapping[str, CategoryResult]) -> float:
        return _flag_share(members)

    def threshold(self, n: int) -> float:
        k = n // 2 + 1 if self.k is None else self.k
        if not 1 <= k <= n:
            raise ValueError(f"k must be between 1 and {n}, got {k}")
        return k / n


class Any(Votes):
    """Flag when any member flags."""

    def threshold(self, n: int) -> float:
        return 1 / n


class All(Votes):
    """Flag only when every member flags."""

    def threshold(self, n: int) -> float:
        return 1.0


@dataclass(frozen=True)
class Weighted:
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


RULES = {"any": Any, "all": All, "votes": Votes}


def _labels(guards: Sequence[Guard]) -> list[str]:
    """Class names, numbered on repeats: LayaGuard, JevGuard, JevGuard#2."""
    seen: dict[str, int] = {}
    labels = []
    for g in guards:
        name = type(g).__name__
        seen[name] = seen.get(name, 0) + 1
        labels.append(name if seen[name] == 1 else f"{name}#{seen[name]}")
    return labels


def _check(guards: Sequence[Guard], kind: str) -> None:
    if len(guards) < 2:
        raise ValueError(f"{kind} needs at least two guards, got {len(guards)}")
    names = set(guards[0].categories)
    for g in guards[1:]:
        if set(g.categories) != names:
            raise ValueError(f"{kind} members must use the same categories: {sorted(names)} vs {sorted(g.categories)}")


class EnsembleGuard(Guard):
    """Run every member concurrently and combine their results per category with a `Rule`.

    `rule` is a Rule (Votes(k), Any(), All(), Weighted(...), or your own object with `score` and
    `threshold`) or a shortcut: "any", "all", "votes" (with `k`, default a majority). The built-in
    rules vote on members' decisions, since models score on different scales, so members need a
    `threshold`. `by_mode` holds each member's own score, `raw` each member's CategoryResult, and
    re-thresholding the result (`res.at(...)`) moves the cut-off on the rule's score.
    """

    def __init__(self, guards: Sequence[Guard], rule: str | Rule = "votes", k: int | None = None,
                 debug: bool = False) -> None:
        if isinstance(rule, str):
            if rule not in RULES:
                raise ValueError(f"rule must be one of {sorted(RULES)} or a Rule, got {rule!r}")
            rule = Votes(k) if rule == "votes" else RULES[rule]()
        elif k is not None:
            raise ValueError("k only applies to rule='votes'; pass Votes(k) instead")
        if not isinstance(rule, Rule):
            raise TypeError(f"rule must be a Rule (an object with score() and threshold()), got {rule!r}")
        _check(guards, "EnsembleGuard")
        labels = _labels(guards)
        if isinstance(rule, Votes | Weighted):
            missing = [label for label, g in zip(labels, guards) if g.threshold is None]
            if missing:
                raise ValueError(f"members need a threshold for {type(rule).__name__} (it uses their decisions): {missing}")
        if isinstance(rule, Weighted) and set(rule.weights) != set(labels):
            raise ValueError(f"Weighted weights must name exactly the members {labels}, got {sorted(rule.weights)}")
        super().__init__(categories=dict(guards[0].categories), threshold=rule.threshold(len(guards)), debug=debug)
        self.guards, self.labels, self.rule = list(guards), labels, rule
        self.concurrency = min(g.concurrency for g in guards)

    async def guard(self, context: str) -> GuardResult:
        results = await asyncio.gather(*(g.guard(context) for g in self.guards))
        categories = []
        for name in self.categories:
            members = {label: res[name] for label, res in zip(self.labels, results)}
            categories.append(CategoryResult(name, self.rule.score(members), {lb: c.score for lb, c in members.items()},
                                             members))
        self._log("{}: {}", self.rule, {c.name: round(c.score, 3) for c in categories})
        ranked = sorted(categories, key=lambda c: c.score, reverse=True)
        return GuardResult(ranked=ranked, raw=dict(zip(self.labels, results)), chunks=max(r.chunks for r in results),
                           source=type(self).__name__).at(self.threshold)


class CascadeGuard(Guard):
    """Try members in order; the first one that answers confidently wins.

    A member that raises (an API error, a timeout, `InputTooLong`) passes the text to the next
    one, so `CascadeGuard([api_guard, local_guard])` is a fallback. With `uncertain=(lo, hi)` a
    member also passes it on when any category's score lands in [lo, hi): a cheap guard can
    answer the clear cases and leave the rest to a stronger one. The last member always answers
    (or its error propagates). `res.source` names the member that answered.
    """

    def __init__(self, guards: Sequence[Guard], uncertain: tuple[float, float] | None = None,
                 debug: bool = False) -> None:
        _check(guards, "CascadeGuard")
        if uncertain is not None and not 0 <= uncertain[0] < uncertain[1] <= 1:
            raise ValueError(f"uncertain must be (lo, hi) with 0 <= lo < hi <= 1, got {uncertain!r}")
        super().__init__(categories=dict(guards[0].categories), debug=debug)
        self.guards, self.labels, self.uncertain = list(guards), _labels(guards), uncertain
        self.concurrency = min(g.concurrency for g in guards)

    async def guard(self, context: str) -> GuardResult:
        last = len(self.guards) - 1
        for i, (label, g) in enumerate(zip(self.labels, self.guards)):
            try:
                res = await g.guard(context)
            except Exception as e:
                if i == last:
                    raise
                self._log("{} failed ({}: {}); trying the next guard", label, type(e).__name__, e)
                continue
            if i < last and self.uncertain and any(self.uncertain[0] <= c.score < self.uncertain[1] for c in res.ranked):
                self._log("{} is unsure; escalating", label)
                continue
            return replace(res, source=label)
        raise AssertionError("unreachable")
