"""JevGuard: runs guard questions on TypeSafe's Jev through OpenRouter's Decisions API."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from ..chunking import byte_chunks, check_chunks
from ..guard import Categories, Guard
from ..modes import Expr, Mode
from ..result import Threshold

# A dated snapshot, so results stay comparable; an undated id (typesafe/jev-1.14) follows releases.
DEFAULT_MODEL = "typesafe/jev-1.14-20261014"
DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
_DATED = re.compile(r"-\d{8}$")


class JevGuard(Guard):
    """Jev answers the guard's typed questions directly: every question of every mode goes in one
    request, with the text sent once. Jev reads 262k tokens but switches to a slower, less
    accurate long-context mode past ~32 KB of text, so longer text is split into overlapping
    chunks of at most `max_chunk_bytes`, sent concurrently (`max_concurrency`), and each
    category keeps its highest-scoring chunk. More than `max_chunks` chunks raises
    `InputTooLong`. A pinned (dated) `model` must be the snapshot that answers, or the call
    raises. `usage` accumulates requests, input tokens and cost in USD.
    """

    def __init__(
        self,
        categories: Categories,
        mode: Expr = Mode.NOUL,
        threshold: Threshold = None,
        votes: int | None = None,
        debug: bool = False,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        url: str = DEFAULT_URL,
        max_chunk_bytes: int = 30_000,
        overlap: int = 128,
        max_chunks: int | None = 64,
        max_concurrency: int = 8,
        timeout: float = 60.0,
        retries: int = 3,
        retry_wait: float = 1.0,
        client: Any = None,
    ) -> None:
        super().__init__(categories, mode, threshold, votes, debug)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("JevGuard needs an OpenRouter key: pass api_key= or set OPENROUTER_API_KEY")
        if max_concurrency < 1 or retries < 0 or overlap < 0:
            raise ValueError("need max_concurrency >= 1, retries >= 0 and overlap >= 0")
        self.model, self.url = model, url
        self.max_chunk_bytes, self.overlap, self.max_chunks = max_chunk_bytes, overlap, max_chunks
        self.max_concurrency, self.timeout, self.retries, self.retry_wait = max_concurrency, timeout, retries, retry_wait
        self.usage = {"requests": 0, "input_tokens": 0, "cost": 0.0}
        self._client = client

    async def predict(self, context: str, questions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            import httpx
        except ImportError as e:
            raise ImportError("JevGuard needs httpx: pip install 'reflexguard[jev]'") from e
        chunks = byte_chunks(context, self.max_chunk_bytes, self.overlap)
        check_chunks(len(chunks), self.max_chunks, "the text")
        self._log("{} chunk(s), {} questions per request", len(chunks), len(questions))
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        limit = asyncio.Semaphore(self.max_concurrency)
        try:
            return list(await asyncio.gather(*(self._ask(client, limit, c, questions) for c in chunks)))
        finally:
            if self._client is None:
                await client.aclose()

    async def _ask(self, client: Any, limit: asyncio.Semaphore, text: str,
                   questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        body = {"model": self.model, "state": {"text": text}, "questions": questions}
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
            raise RuntimeError(f"asked for {self.model} but Jev served {served}; results would not be comparable")
        usage = data.get("usage") or {}
        self.usage["requests"] += 1
        self.usage["input_tokens"] += usage.get("input_tokens", 0)
        self.usage["cost"] += usage.get("cost", 0.0)
        return data["answers"]
