"""Token windows for text longer than a model's window, shared by the guards that chunk."""

from __future__ import annotations


class InputTooLong(ValueError):
    """The text needs more chunks than the guard's `max_chunks`. Nothing was scored.

    A guard raises instead of truncating, so a caller that must fail closed can block on it.
    """


def windows(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """`[start, end)` windows of at most `size` over `n` items, each sharing `overlap` with the next."""
    if size < 1:
        raise ValueError(f"window size must be >= 1, got {size}")
    step = max(1, size - max(0, overlap))
    spans, start = [], 0
    while True:
        end = min(n, start + size)
        spans.append((start, end))
        if end >= n:
            return spans
        start += step


def check_chunks(n: int, max_chunks: int | None, what: str) -> None:
    if max_chunks is not None and n > max_chunks:
        raise InputTooLong(
            f"{what} needs {n} chunks, over max_chunks={max_chunks}; raise max_chunks, or treat the input as "
            "unsafe (catch InputTooLong to fail closed)"
        )
