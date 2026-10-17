"""Guards made of guards: EnsembleGuard (concurrent, votes) and CascadeGuard (sequential, escalates)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Sequence

from .guard import Guard
from .result import CategoryResult, GuardResult
from .rules import EnsembleRule, EscalationRule, Margin, Votes, Weighted


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
    """Run every member concurrently and combine their results per category with an `EnsembleRule`.

    `rule` is any `EnsembleRule`: the built-in `Votes(k)` (default: a majority), `AnyOf()`, `AllOf()`,
    `Weighted(weights, cutoff)`, or your own object with `score` and `threshold`. The built-in
    rules vote on members' decisions, since models score on different scales, so members need a
    `threshold`. `by_mode` holds each member's own score, `raw` each member's CategoryResult, and
    re-thresholding the result (`res.at(...)`) moves the cut-off on the rule's score.
    """

    def __init__(self, guards: Sequence[Guard], rule: EnsembleRule = Votes(), debug: bool = False) -> None:
        if not isinstance(rule, EnsembleRule):
            raise TypeError(f"EnsembleGuard needs an EnsembleRule (score() and threshold()), got {rule!r}")
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
    """Try members in order; the first one whose answer stands wins.

    A member that raises (an API error, a timeout, `InputTooLong`) always passes the text to the
    next one, so `CascadeGuard([api_guard, local_guard])` is a fallback. With an `EscalationRule`
    a member also passes it on when `rule.escalate(result)` is True: with `Margin(0.2)` a cheap
    guard answers the clear cases and leaves the ones near its threshold to a stronger one.
    `rule` is one rule for every handoff or a list with one per handoff (len(guards) - 1). The
    last member always answers (or its error propagates). `res.source` names the member that
    answered. `a >> b >> c` builds a cascade without a rule.
    """

    def __init__(self, guards: Sequence[Guard], rule: EscalationRule | Sequence[EscalationRule] | None = None,
                 debug: bool = False) -> None:
        _check(guards, "CascadeGuard")
        handoffs = len(guards) - 1
        rules = list(rule) if isinstance(rule, Sequence) else [rule] * handoffs
        if len(rules) != handoffs:
            raise ValueError(f"CascadeGuard needs one rule per handoff ({handoffs}), got {len(rules)}")
        for r in rules:
            if r is not None and not isinstance(r, EscalationRule):
                raise TypeError(f"CascadeGuard needs an EscalationRule (escalate()), got {r!r}")
        labels = _labels(guards)
        missing = [lb for lb, g, r in zip(labels, guards, rules) if isinstance(r, Margin) and g.threshold is None]
        if missing:
            raise ValueError(f"Margin escalates relative to a threshold; these guards have none: {missing}")
        super().__init__(categories=dict(guards[0].categories), debug=debug)
        self.guards, self.labels, self.rules = list(guards), labels, rules
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
            if i < last and self.rules[i] is not None and self.rules[i].escalate(res):
                self._log("{}: {} escalates", label, self.rules[i])
                continue
            return replace(res, source=label)
        raise AssertionError("unreachable")
