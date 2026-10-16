"""Token windows for text longer than a model's window, shared by the guards that chunk."""

from __future__ import annotations

from bisect import bisect_right
from itertools import accumulate


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


def byte_chunks(text: str, max_bytes: int, overlap: int) -> list[str]:
    """Pieces of `text` of at most `max_bytes` UTF-8 bytes, cut at whitespace where possible.

    Each piece starts about `overlap` characters before the previous one ended, on a word
    boundary, so a phrase across a cut appears whole in one piece. For APIs that bound the
    request size but expose no tokenizer.
    """
    if max_bytes < 4:
        raise ValueError(f"max_bytes must be >= 4, got {max_bytes}")
    ends = list(accumulate(len(ch.encode()) for ch in text))  # ends[i]: bytes in text[:i + 1]
    pieces, start = [], 0
    while True:
        base = ends[start - 1] if start else 0
        end = bisect_right(ends, base + max_bytes, lo=start)  # longest text[start:end] that fits
        if end >= len(text):
            pieces.append(text[start:])
            return pieces
        space = text.rfind(" ", start + 1, end + 1)
        if space > start:  # cut after a word rather than inside one
            end = space
        pieces.append(text[start:end])
        nxt = max(start + 1, end - overlap)
        boundary = text.find(" ", nxt, end)
        start = boundary + 1 if boundary != -1 and boundary + 1 < end else nxt
