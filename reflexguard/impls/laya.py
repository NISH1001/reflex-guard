"""LayaGuard: runs guard questions on a local Laya router."""

from __future__ import annotations

import asyncio
import threading
import warnings
from typing import Any

from ..guard import Categories, Guard
from ..modes import Expr, Mode
from ..result import Threshold

warnings.filterwarnings("ignore", message="laya: this checkpoint ships invalid temperatures")


class LayaGuard(Guard):
    def __init__(
        self,
        categories: Categories,
        mode: Expr = Mode.NOUL,
        threshold: Threshold = None,
        votes: int | None = None,
        debug: bool = False,
        router: Any = None,
    ) -> None:
        super().__init__(categories, mode, threshold, votes, debug)
        self._router = router
        self._lock = threading.Lock()

    async def predict(self, context: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return await asyncio.to_thread(self._predict_sync, context, questions)

    def _predict_sync(self, context: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        # One lock: the router is created once and calls into the model are serialized.
        with self._lock:
            if self._router is None:
                self._log("creating Laya router")
                self._router = _make_router()
            return self._router.predict({"text": context}, questions)["answers"]


def _make_router() -> Any:
    try:
        from laya import Router
    except ImportError as e:
        raise ImportError("LayaGuard needs laya: pip install 'reflexguard[laya]'") from e
    return Router()
