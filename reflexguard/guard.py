"""The Guard protocol. Implementations subclass it and write only `predict`."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Protocol, runtime_checkable

from loguru import logger

from .modes import Expr, Mode, with_votes
from .questions import build_questions, mode_score, question_id
from .result import CategoryResult, GuardResult, Threshold, resolve_thresholds

Categories = list[str] | dict[str, str]


@runtime_checkable
class Guard(Protocol):
    NOUL: ClassVar[Mode] = Mode.NOUL
    CHOICE: ClassVar[Mode] = Mode.CHOICE
    SCORE: ClassVar[Mode] = Mode.SCORE
    # guard_many's default number of texts in flight. Local models serialize their model calls, so
    # 2 only overlaps one text's pre/post-processing with another's model call; API guards raise it.
    concurrency: ClassVar[int] = 2

    def __init__(
        self,
        categories: Categories,
        mode: Expr = Mode.NOUL,
        threshold: Threshold = None,
        votes: int | None = None,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.categories = _normalize_categories(categories)
        if not isinstance(mode, Expr):
            raise TypeError(f"mode must be a Mode or a combination of modes, got {mode!r}")
        self.mode = mode if votes is None else with_votes(mode, votes)
        resolve_thresholds(threshold, list(self.categories))  # fail fast on a bad threshold
        self.threshold = threshold

    async def predict(
        self, context: str, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Ask the model `questions` about `context`; return answers keyed by question id.

        A guard that splits long text returns one such dict per chunk. Each category is then
        judged by its worst chunk, with its modes combined within that chunk.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement predict(); use e.g. LayaGuard")

    async def guard(self, context: str) -> GuardResult:
        if not isinstance(context, str):
            raise TypeError(f"context must be a str, got {type(context).__name__}")
        if not context.strip():
            raise ValueError("context is empty")
        modes = self.mode.modes()
        questions = build_questions(self.categories, modes)
        self._log("{} questions for {} categories, mode={!r}", len(questions), len(self.categories), self.mode)
        t0 = time.perf_counter()
        answers = await self.predict(context, questions)
        per_chunk = answers if isinstance(answers, list) else [answers]
        if not per_chunk:
            raise ValueError("model returned no chunks")
        self._log("predict took {:.1f} ms over {} chunk(s)", (time.perf_counter() - t0) * 1000, len(per_chunk))
        results = []
        for name in self.categories:
            # Combine the modes within each chunk, then keep the chunk where the category scores
            # highest: modes must agree about the same passage, and a category is as bad as its
            # worst chunk. Different categories may peak in different chunks (multi-label).
            best = None
            for i, chunk_answers in enumerate(per_chunk):
                by_mode, raw = {}, {}
                for m in modes:
                    qid = question_id(m, name)
                    if qid not in chunk_answers:
                        raise KeyError(f"model returned no answer for question {qid!r}")
                    raw[qid] = chunk_answers[qid]
                    by_mode[m.name] = mode_score(m, name, chunk_answers[qid])
                score = self.mode.combine(by_mode)
                if best is None or score > best.score:
                    best = CategoryResult(name, score, by_mode, raw, chunk=i)
            results.append(best)
        ranked = sorted(results, key=lambda c: c.score, reverse=True)
        raw = {qid: a for c in ranked for qid, a in c.raw.items()}
        result = GuardResult(ranked=ranked, raw=raw, chunks=len(per_chunk),
                             source=type(self).__name__).at(self.threshold)
        for c in result.ranked:
            self._log("{} score={:.3f} by_mode={}", c.name, c.score, c.by_mode)
        self._log("flagged={} violations={}", result.flagged, [c.name for c in result.violations or []])
        return result

    def _log(self, message: str, *args: Any) -> None:
        if self.debug:
            logger.opt(depth=1).debug(f"{type(self).__name__}: {message}", *args)

    async def guard_many(self, contexts: list[str], concurrency: int | None = None) -> list[GuardResult]:
        """`guard` over many texts, `concurrency` at a time (default: the guard's); results in input order."""
        limit = self.concurrency if concurrency is None else concurrency
        if limit < 1:
            raise ValueError(f"concurrency must be >= 1, got {limit}")
        gate = asyncio.Semaphore(limit)

        async def one(context: str) -> GuardResult:
            async with gate:
                return await self.guard(context)

        return list(await asyncio.gather(*(one(c) for c in contexts)))

    def __or__(self, other: object) -> Guard:
        """`a | b`: an EnsembleGuard that flags when any member does (AnyOf); chains flatten."""
        return _combine(self, other, any_of=True)

    def __and__(self, other: object) -> Guard:
        """`a & b`: an EnsembleGuard that flags only when every member does (AllOf); chains flatten."""
        return _combine(self, other, any_of=False)

    def guard_sync(self, context: str) -> GuardResult:
        """Sync `guard`. Inside a running event loop (Jupyter, marimo) it runs on a worker thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.guard(context))
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.guard(context)).result()


def _combine(left: Guard, right: object, any_of: bool) -> Guard:
    if not isinstance(right, Guard):
        return NotImplemented
    from .compose import AllOf, AnyOf, EnsembleGuard  # compose imports this module

    rule = AnyOf if any_of else AllOf

    def members(g: Guard) -> list[Guard]:  # a | b | c is one ensemble, a | (b & c) nests
        return list(g.guards) if isinstance(g, EnsembleGuard) and type(g.rule) is rule else [g]

    return EnsembleGuard(members(left) + members(right), rule=rule())


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
