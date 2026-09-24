"""JevGuard: not implemented yet."""

from __future__ import annotations

from typing import Any

from ..guard import Guard


class JevGuard(Guard):
    async def predict(self, context: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError("JevGuard is not implemented yet")
