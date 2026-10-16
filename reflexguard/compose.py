"""Guards made of guards: EnsembleGuard (concurrent, votes) and CascadeGuard (sequential, escalates)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Sequence

from .guard import Guard
from .result import CategoryResult, GuardResult

RULES = ("any", "all", "votes")


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
    """Run every member concurrently and vote per category.

    Members score on different scales, so the ensemble combines their decisions: each member
    needs a `threshold`, and a category's ensemble score is the share of members that flag it.
    `rule="any"` flags when one member does, `"all"` when every member does, `"votes"` when at
    least `k` do (default: a majority). `by_mode` holds each member's own score, `raw` each
    member's `CategoryResult`. Re-thresholding the result (`res.at(0.5)`) changes the vote share.
    """

    def __init__(self, guards: Sequence[Guard], rule: str = "votes", k: int | None = None, debug: bool = False) -> None:
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
        _check(guards, "EnsembleGuard")
        missing = [label for label, g in zip(_labels(guards), guards) if g.threshold is None]
        if missing:
            raise ValueError(f"EnsembleGuard members need a threshold (it votes on their decisions): {missing}")
        n = len(guards)
        k = {"any": 1, "all": n}.get(rule, n // 2 + 1 if k is None else k)
        if not 1 <= k <= n:
            raise ValueError(f"k must be between 1 and {n}, got {k}")
        super().__init__(categories=dict(guards[0].categories), threshold=k / n, debug=debug)
        self.guards, self.labels, self.rule, self.k = list(guards), _labels(guards), rule, k
        self.concurrency = min(g.concurrency for g in guards)

    async def guard(self, context: str) -> GuardResult:
        results = await asyncio.gather(*(g.guard(context) for g in self.guards))
        categories = []
        for name in self.categories:
            members = {label: res[name] for label, res in zip(self.labels, results)}
            votes = sum(bool(c.flagged) for c in members.values())
            categories.append(CategoryResult(name, votes / len(self.guards), {lb: c.score for lb, c in members.items()},
                                             members))
        self._log("{} votes={}", self.rule, {c.name: round(c.score * len(self.guards)) for c in categories})
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
