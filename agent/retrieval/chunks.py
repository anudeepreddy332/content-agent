"""Rankable CSWP chunk view with normalized source intervals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalChunk:
    """One rankable item with exact normalized source intervals."""

    ordinal: int
    chunk_id: str
    source: str
    source_path: str
    retrieval_text: str
    source_intervals: tuple[tuple[int, int], ...]
    embedding_content_token_count: int


def source_offsets(raw_text: str) -> tuple[int, int]:
    stripped = raw_text.strip()
    start = raw_text.index(stripped) if stripped else 0
    return start, start + len(stripped)


def normalize_interval(
    start: int, end: int, raw_start: int, raw_end: int
) -> tuple[int, int] | None:
    clipped_start = max(start, raw_start)
    clipped_end = min(end, raw_end)
    if clipped_start >= clipped_end:
        return None
    return clipped_start - raw_start, clipped_end - raw_start
