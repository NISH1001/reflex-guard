"""The Guard protocol. Implementations subclass it and write only `predict`."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Protocol, runtime_checkable

from .modes import Expr, Mode, with_votes
from .questions import build_questions, mode_score, question_id
from .result import CategoryResult, GuardResult, Threshold, resolve_thresholds

Categories = list[str] | dict[str, str]


@runtime_checkable
class Guard(Protocol):
    NOUL: ClassVar[Mode] = Mode.NOUL
    CHOICE: ClassVar[Mode] = Mode.CHOICE
    SCORE: ClassVar[Mode] = Mode.SCORE

    def __init__(
        self,
        categories: Categories,
        mode: Expr = Mode.NOUL,
        threshold: Threshold = None,
        votes: int | None = None,
    ) -> None:
        self.categories = _normalize_categories(categories)
        if not isinstance(mode, Expr):
            raise TypeError(f"mode must be a Mode or a combination of modes, got {mode!r}")
        self.mode = mode if votes is None else with_votes(mode, votes)
        resolve_thresholds(threshold, list(self.categories))  # fail fast on a bad threshold
        self.threshold = threshold

    async def predict(self, context: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Ask the model `questions` about `context`; return answers keyed by question id."""
        raise NotImplementedError(f"{type(self).__name__} must implement predict(); use e.g. LayaGuard")

    async def aguard(self, context: str) -> GuardResult:
        if not isinstance(context, str):
            raise TypeError(f"context must be a str, got {type(context).__name__}")
        if not context.strip():
            raise ValueError("context is empty")
        modes = self.mode.modes()
        answers = await self.predict(context, build_questions(self.categories, modes))
        results = []
        for name in self.categories:
            by_mode, raw = {}, {}
            for m in modes:
                qid = question_id(m, name)
                if qid not in answers:
                    raise KeyError(f"model returned no answer for question {qid!r}")
                raw[qid] = answers[qid]
                by_mode[m.name] = mode_score(m, name, answers[qid])
            results.append(CategoryResult(name, self.mode.combine(by_mode), by_mode, raw))
        ranked = sorted(results, key=lambda c: c.score, reverse=True)
        return GuardResult(ranked=ranked, raw=answers).at(self.threshold)

    def guard(self, context: str) -> GuardResult:
        """Sync `aguard`. Inside a running event loop (Jupyter, marimo) it runs on a worker thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aguard(context))
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.aguard(context)).result()


def _normalize_categories(categories: Categories) -> dict[str, str]:
    if isinstance(categories, str) or not isinstance(categories, (list, tuple, dict)):
        raise TypeError("categories must be a list of names or a {name: description} dict")
    cats = dict(categories) if isinstance(categories, dict) else dict.fromkeys(categories, "")
    if not cats:
        raise ValueError("categories is empty")
    for name, desc in cats.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"category names must be non-empty strings, got {name!r}")
        if not isinstance(desc, str):
            raise TypeError(f"description for {name!r} must be a str, got {desc!r}")
    return cats
