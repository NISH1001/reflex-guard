"""VonGuard: not implemented yet."""

from __future__ import annotations

from typing import Any

from ..guard import Guard


class VonGuard(Guard):
    async def predict(self, context: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError("VonGuard is not implemented yet")
