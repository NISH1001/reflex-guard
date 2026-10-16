"""JevGuard: runs guard questions on TypeSafe's Jev through OpenRouter's Decisions API."""

from __future__ import annotations

from typing import Any

from ._decision_api import DecisionAPIGuard, openrouter_answers

# A dated snapshot, so results stay comparable; an undated id (typesafe/jev-1.14) follows releases.
DEFAULT_MODEL = "typesafe/jev-1.14-20261014"
DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"


class JevGuard(DecisionAPIGuard):
    """Jev answers the guard's typed questions directly: every question of every mode goes in one
    request, with the text sent once. Jev reads 262k tokens but switches to a slower, less
    accurate long-context mode past ~32 KB of text, so longer text is split into overlapping
    chunks of at most `max_chunk_bytes`, sent concurrently (`max_concurrency`), and each
    category keeps its highest-scoring chunk. More than `max_chunks` chunks raises
    `InputTooLong`. A pinned (dated) `model` must be the snapshot that answers, or the call
    raises. `usage` accumulates requests, input tokens, cost in USD and refusals; a refused
    question follows `on_refusal` ("flag" by default: it scores 1.0).
    """

    url = DEFAULT_URL
    key_env = "OPENROUTER_API_KEY"
    extra = "jev"

    def __init__(self, categories, model: str = DEFAULT_MODEL, **kwargs: Any) -> None:
        super().__init__(categories, model=model, **kwargs)

    def _payload(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {"model": self.model, "state": {"text": text}, "questions": questions}

    def _answers(self, data: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return openrouter_answers(data, questions)
