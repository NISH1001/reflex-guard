"""OpenAIDecisionGuard: runs guard questions on OpenAI's Decisions API (GPT-6 Luna)."""

from __future__ import annotations

from typing import Any

from ..questions import SCORE_LEVELS
from ._decision_api import DecisionAPIGuard, openrouter_answers

VIA = {
    # OpenAI's own endpoint and schema: predicate / choice / score questions, answers as a list.
    "openai": ("https://api.openai.com/v1/decisions", "OPENAI_API_KEY", "gpt-6-luna"),
    # The same model through OpenRouter's Decisions API (Jev's schema), as a dated snapshot.
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY",
                   "openai/gpt-6-luna-decisions-20261006"),
}


class OpenAIDecisionGuard(DecisionAPIGuard):
    """OpenAI's Decisions API answers typed questions natively: `noul` maps to a `predicate`,
    `choice` to a `choice` and `score` to a `score` question, all in one request (up to 200
    questions; more are split). `via="openai"` (default) calls api.openai.com with
    OPENAI_API_KEY; `via="openrouter"` calls the same model through OpenRouter with
    OPENROUTER_API_KEY. Chunking, concurrency, retries, pinning, `usage` and `on_refusal` work as
    in `JevGuard`. A dated `model` (e.g. gpt-6-luna-2026-10-06) must be the one that answers.
    """

    extra = "openai"

    def __init__(self, categories, model: str | None = None, via: str = "openai", **kwargs: Any) -> None:
        if via not in VIA:
            raise ValueError(f"via must be one of {sorted(VIA)}, got {via!r}")
        self.via = via
        self.url, self.key_env, default_model = VIA[via]
        self.max_questions = 200 if via == "openai" else None
        super().__init__(categories, model=model or default_model, **kwargs)

    def _payload(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if self.via == "openrouter":
            return {"model": self.model, "state": {"text": text}, "questions": questions}
        return {"model": self.model, "input": text, "questions": [_native(qid, q) for qid, q in questions.items()]}

    def _answers(self, data: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if self.via == "openrouter":
            return openrouter_answers(data, questions)
        by_name = {a.get("name"): a for a in data.get("answers") or []}
        out: dict[str, Any] = {}
        for qid in questions:
            a = by_name.get(qid)
            if a is None:
                continue  # Guard reports the missing answer
            if a["type"] == "refusal":
                out[qid] = None
            elif a["type"] == "predicate":
                out[qid] = {"noul": a["probability"]}
            elif a["type"] == "choice":
                out[qid] = {"probabilities": {str(p["value"]): p["probability"] for p in a["probabilities"]}}
            else:
                out[qid] = {"score": a["score"], "probabilities": {str(p["value"]): p["probability"]
                                                                   for p in a["probabilities"]}}
        return out


def _native(qid: str, q: dict[str, Any]) -> dict[str, Any]:
    """reflexguard's question as OpenAI's: the input is a plain string, not a `text` field."""
    instructions = q["instructions"].replace("`text`", "the input")
    if q["type"] == "noul":
        return {"type": "predicate", "name": qid, "instructions": instructions}
    if q["type"] == "choice":
        return {"type": "choice", "name": qid, "instructions": instructions,
                "choices": [{"value": k, "description": v} for k, v in q["criteria"].items()]}
    levels = q.get("criteria") or SCORE_LEVELS
    return {"type": "score", "name": qid, "instructions": instructions, "levels": [{"label": lv} for lv in levels]}
