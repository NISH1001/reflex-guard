"""Shared plumbing for guards on hosted typed-decision APIs (Jev, OpenAI Decisions).

These models answer reflexguard's question types natively over HTTP, so a guard only translates
questions and answers; this base does the rest: chunking text below the API's comfortable size,
concurrent requests, retries, snapshot pinning, usage, and a policy for refused questions.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from ..chunking import byte_chunks, check_chunks
from ..guard import Categories, Guard
from ..modes import Expr, Mode
from ..result import Threshold

# A dated snapshot ends in -20261014 or -2026-10-14; only those are checked against what served.
_DATED = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2})$")
REFUSAL_POLICIES = ("flag", "ignore", "raise")


class DecisionAPIGuard(Guard):
    """Subclasses set `url` / `key_env` and implement `_payload` and `_answers`.

    Use it as an async context manager to share one connection pool across calls
    (`async with JevGuard(...) as g: await g.guard_many(texts)`); otherwise each call opens its
    own client. `guard_many` shares one automatically.
    """

    concurrency = 5  # I/O-bound: guard_many keeps 5 texts in flight by default

    url: str = ""
    key_env: str = ""
    extra: str = ""  # the pip extra that installs httpx for this guard
    max_questions: int | None = None  # per request; larger question sets are split

    def __init__(
        self,
        categories: Categories,
        mode: Expr = Mode.NOUL,
        threshold: Threshold = None,
        votes: int | None = None,
        debug: bool = False,
        model: str = "",
        api_key: str | None = None,
        url: str | None = None,
        max_chunk_bytes: int = 30_000,
        overlap: int = 128,
        max_chunks: int | None = 64,
        max_concurrency: int = 8,
        timeout: float = 60.0,
        retries: int = 3,
        retry_wait: float = 1.0,
        on_refusal: str = "flag",
        client: Any = None,
    ) -> None:
        super().__init__(categories, mode, threshold, votes, debug)
        self.api_key = api_key or os.environ.get(self.key_env)
        if not self.api_key:
            raise ValueError(f"{type(self).__name__} needs an API key: pass api_key= or set {self.key_env}")
        if max_concurrency < 1 or retries < 0 or overlap < 0:
            raise ValueError("need max_concurrency >= 1, retries >= 0 and overlap >= 0")
        if on_refusal not in REFUSAL_POLICIES:
            raise ValueError(f"on_refusal must be one of {REFUSAL_POLICIES}, got {on_refusal!r}")
        self.model, self.url = model, url or self.url
        self.max_chunk_bytes, self.overlap, self.max_chunks = max_chunk_bytes, overlap, max_chunks
        self.max_concurrency, self.timeout, self.retries, self.retry_wait = max_concurrency, timeout, retries, retry_wait
        self.on_refusal = on_refusal
        self.usage = {"requests": 0, "input_tokens": 0, "cost": 0.0, "refusals": 0}
        self._client = client
        self._opened: list[bool] = []  # one entry per active `async with`: did it open the client?

    async def __aenter__(self) -> DecisionAPIGuard:
        import httpx

        self._opened.append(self._client is None)
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._opened.pop():
            await self._client.aclose()
            self._client = None

    async def guard_many(self, contexts: list[str], concurrency: int | None = None) -> list:
        async with self:  # one connection pool for the whole batch
            return await super().guard_many(contexts, concurrency)

    # ── to implement ────────────────────────────────────────────────────────
    def _payload(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    def _answers(self, data: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Answers keyed by question id in reflexguard's shape; a refused question maps to None."""
        raise NotImplementedError

    # ── shared ──────────────────────────────────────────────────────────────
    async def predict(self, context: str, questions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            import httpx
        except ImportError as e:
            raise ImportError(f"{type(self).__name__} needs httpx: pip install 'reflexguard[{self.extra}]'") from e
        chunks = byte_chunks(context, self.max_chunk_bytes, self.overlap)
        check_chunks(len(chunks), self.max_chunks, "the text")
        self._log("{} chunk(s), {} questions per request", len(chunks), len(questions))
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        limit = asyncio.Semaphore(self.max_concurrency)
        groups = [questions] if not self.max_questions else [
            dict(list(questions.items())[i:i + self.max_questions]) for i in range(0, len(questions), self.max_questions)]

        async def chunk_answers(text: str) -> dict[str, Any]:
            parts = await asyncio.gather(*(self._ask(client, limit, text, group) for group in groups))
            return {qid: a for part in parts for qid, a in part.items()}

        try:
            return list(await asyncio.gather(*(chunk_answers(c) for c in chunks)))
        finally:
            if self._client is None:
                await client.aclose()

    async def _ask(self, client: Any, limit: asyncio.Semaphore, text: str,
                   questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        body = self._payload(text, questions)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with limit:
            for attempt in range(self.retries + 1):
                response = await client.post(self.url, json=body, headers=headers)
                retry = response.status_code == 429 or response.status_code >= 500
                if not retry or attempt == self.retries:
                    break
                self._log("HTTP {}, retrying", response.status_code)
                await asyncio.sleep(self.retry_wait * 2**attempt)
        response.raise_for_status()
        data = response.json()
        served = data.get("model", "")
        if _DATED.search(self.model) and served != self.model:
            raise RuntimeError(f"asked for {self.model} but the API served {served}; results would not be comparable")
        usage = data.get("usage") or {}
        self.usage["requests"] += 1
        self.usage["input_tokens"] += usage.get("input_tokens", 0)
        self.usage["cost"] += usage.get("cost") or 0.0
        answers = self._answers(data, questions)
        return {qid: (a if a is not None else self._refused(qid)) for qid, a in answers.items()}

    def _refused(self, qid: str) -> dict[str, Any]:
        """A refused question: fail closed by default (`flag`), or score it 0 (`ignore`), or raise."""
        self.usage["refusals"] += 1
        if self.on_refusal == "raise":
            raise RuntimeError(f"the model refused question {qid!r}")
        mode, _, category = qid.partition(":")
        p = 1.0 if self.on_refusal == "flag" else 0.0
        return {"refusal": True, "noul": p, "probabilities": {category: p, f"not_{category}": 1 - p}, "score": 3 * p}


def openrouter_answers(data: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """OpenRouter's Decisions API answers in reflexguard's own shape, keyed by question id."""
    answers = data.get("answers") or {}
    return {qid: (None if answers.get(qid, {}).get("type") == "refusal" else answers[qid]) for qid in questions}
