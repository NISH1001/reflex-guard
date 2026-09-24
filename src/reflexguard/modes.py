"""Guard modes and the `|` / `&` expressions that combine them."""

from __future__ import annotations

from dataclasses import dataclass


class Expr:
    """A mode or a combination of modes. Reduces per-mode scores to one score."""

    def __or__(self, other: Expr) -> AnyOf:
        return AnyOf(_parts(self, AnyOf) + _parts(_check(other), AnyOf))

    def __and__(self, other: Expr) -> AllOf:
        return AllOf(_parts(self, AllOf) + _parts(_check(other), AllOf))

    def modes(self) -> tuple[Mode, ...]:
        """Every distinct mode in the expression, in first-seen order."""
        raise NotImplementedError

    def combine(self, scores: dict[str, float]) -> float:
        """Reduce `{mode name: score}` to one score."""
        raise NotImplementedError


@dataclass(frozen=True)
class Mode(Expr):
    name: str

    def modes(self) -> tuple[Mode, ...]:
        return (self,)

    def combine(self, scores: dict[str, float]) -> float:
        return scores[self.name]

    def __repr__(self) -> str:
        return f"Mode.{self.name.upper()}"


@dataclass(frozen=True)
class AnyOf(Expr):
    """At least `k` of `parts` (k=1: any). Score is the k-th highest part score."""

    parts: tuple[Expr, ...]
    k: int = 1

    def modes(self) -> tuple[Mode, ...]:
        return _unique(self.parts)

    def combine(self, scores: dict[str, float]) -> float:
        return sorted((p.combine(scores) for p in self.parts), reverse=True)[self.k - 1]


@dataclass(frozen=True)
class AllOf(Expr):
    """Every part. Score is the lowest part score."""

    parts: tuple[Expr, ...]

    def modes(self) -> tuple[Mode, ...]:
        return _unique(self.parts)

    def combine(self, scores: dict[str, float]) -> float:
        return min(p.combine(scores) for p in self.parts)


Mode.NOUL = Mode("noul")
Mode.CHOICE = Mode("choice")
Mode.SCORE = Mode("score")
MODES = (Mode.NOUL, Mode.CHOICE, Mode.SCORE)


def with_votes(expr: Expr, votes: int) -> AnyOf:
    """Require at least `votes` parts of a top-level `|` group."""
    if not isinstance(expr, AnyOf) or expr.k != 1:
        raise ValueError("votes needs a plain `|` group of modes, e.g. NOUL | CHOICE | SCORE")
    if isinstance(votes, bool) or not isinstance(votes, int) or not 1 <= votes <= len(expr.parts):
        raise ValueError(f"votes must be an int from 1 to {len(expr.parts)}, got {votes!r}")
    return AnyOf(expr.parts, votes)


def _check(other: object) -> Expr:
    if not isinstance(other, Expr):
        raise TypeError(f"can only combine modes with modes, got {other!r}")
    return other


def _parts(expr: Expr, kind: type) -> tuple[Expr, ...]:
    # Flatten a | b | c into one group; keep groups with votes (k > 1) intact.
    if isinstance(expr, kind) and getattr(expr, "k", 1) == 1:
        return expr.parts
    return (expr,)


def _unique(parts: tuple[Expr, ...]) -> tuple[Mode, ...]:
    return tuple(dict.fromkeys(m for p in parts for m in p.modes()))
