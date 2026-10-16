"""LayaGuard: runs guard questions on a local Laya router."""

from __future__ import annotations

import asyncio
import threading
import warnings
from typing import Any

from ..chunking import check_chunks, windows
from ..guard import Categories, Guard
from ..modes import Expr, Mode
from ..result import Threshold

warnings.filterwarnings("ignore", message="laya: this checkpoint ships invalid temperatures")

# Tokens Laya's sequence spends besides the question head and the text: [CLS], [SEP]s, and the
# "text:" key of the serialized state. Generous, so a chunk never gets cut by Laya itself.
_RESERVED = 16


class LayaGuard(Guard):
    """Laya answers every question in one call, but only reads the first `max_len - head_max_len`
    tokens of the text (about 300 on its defaults) and silently drops the rest. Longer text is
    split into overlapping chunks with Laya's own tokenizer, scored in one batched call, and each
    category keeps its highest-scoring chunk. Needing more than `max_chunks` chunks raises
    `InputTooLong` instead of scoring part of the text.
    """

    def __init__(
        self,
        categories: Categories,
        mode: Expr = Mode.NOUL,
        threshold: Threshold = None,
        votes: int | None = None,
        debug: bool = False,
        router: Any = None,
        overlap: int = 32,
        max_chunks: int | None = 64,
    ) -> None:
        super().__init__(categories, mode, threshold, votes, debug)
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        self._router = router
        self.overlap, self.max_chunks = overlap, max_chunks
        self._lock = threading.Lock()

    async def predict(self, context: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return await asyncio.to_thread(self._predict_sync, context, questions)

    def _predict_sync(self, context: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        # One lock: the router is created once and calls into the model are serialized.
        with self._lock:
            if self._router is None:
                self._log("creating Laya router")
                self._router = _make_router()
            model, chunks = self._chunks(context, questions)
            if len(chunks) == 1:
                return self._router.predict({"text": context}, questions)["answers"]
            self._log("{} chunks of long text in one batched call", len(chunks))
            results = self._router.predict_batch(
                [{"state": {"text": c}, "questions": questions, "model": model} for c in chunks]
            )
            return [r["answers"] for r in results]  # one per chunk; Guard combines them

    def _chunks(self, context: str, questions: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
        """The routed checkpoint, and the text cut into pieces that fit its window."""
        model = self._router.route({"text": context}, questions)["model"]
        agent = self._router.load(model)
        budget = agent.cfg.get("max_len", 512) - agent.cfg.get("head_max_len", 192) - _RESERVED
        # verbose=False: this only measures the text, so the tokenizer's "longer than the maximum
        # sequence length" warning does not apply.
        enc = agent.tok(context, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        offsets = enc["offset_mapping"]
        if len(offsets) <= budget:
            return model, [context]
        spans = windows(len(offsets), budget, min(self.overlap, budget // 4))
        check_chunks(len(spans), self.max_chunks, "the text")
        # Cut the original string at token boundaries, so every chunk is the text as written.
        return model, [context[offsets[s][0]:offsets[e - 1][1]] for s, e in spans]


def _make_router() -> Any:
    try:
        from laya import Router
    except ImportError as e:
        raise ImportError("LayaGuard needs laya: pip install 'reflexguard[laya]'") from e
    return Router()
