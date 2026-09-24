"""Guard results: every category ranked by score, plus violations when thresholded."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

Threshold = float | dict[str, float] | None


@dataclass(frozen=True)
class CategoryResult:
    name: str
    score: float
    by_mode: dict[str, float]
    raw: dict[str, Any] = field(default_factory=dict)
    threshold: float | None = None

    @property
    def flagged(self) -> bool | None:
        return None if self.threshold is None else self.score >= self.threshold


@dataclass(frozen=True)
class GuardResult:
    ranked: list[CategoryResult]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def violations(self) -> list[CategoryResult] | None:
        if not self.ranked or self.ranked[0].threshold is None:
            return None
        return [c for c in self.ranked if c.flagged]

    @property
    def flagged(self) -> bool | None:
        v = self.violations
        return None if v is None else bool(v)

    @property
    def scores(self) -> dict[str, float]:
        return {c.name: c.score for c in self.ranked}

    @property
    def top(self) -> CategoryResult:
        return self.ranked[0]

    def __getitem__(self, name: str) -> CategoryResult:
        for c in self.ranked:
            if c.name == name:
                return c
        raise KeyError(name)

    def at(self, threshold: Threshold) -> GuardResult:
        """The same scores under a different threshold; the model is not re-run."""
        per_cat = resolve_thresholds(threshold, [c.name for c in self.ranked])
        return replace(self, ranked=[replace(c, threshold=per_cat[c.name]) for c in self.ranked])


def resolve_thresholds(threshold: Threshold, names: list[str]) -> dict[str, float | None]:
    """Validate `threshold` and expand it to one value (or None) per category."""
    if threshold is None:
        return dict.fromkeys(names)
    if isinstance(threshold, dict):
        missing = [n for n in names if n not in threshold]
        unknown = [n for n in threshold if n not in names]
        if missing or unknown:
            raise ValueError(f"threshold dict must cover exactly the categories; missing={missing}, unknown={unknown}")
        return {n: _check(threshold[n]) for n in names}
    value = _check(threshold)
    return dict.fromkeys(names, value)


def _check(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"threshold must be a number in [0, 1], got {value!r}")
    return float(value)
