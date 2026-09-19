"""Evidence span coverage metrics for retrieval parity evaluation."""

from __future__ import annotations

from typing import Any

from agent.retrieval.chunks import EvalChunk


def span_covered(
    span: Any,
    rows: list[dict[str, Any]],
    chunks_by_id: dict[str, EvalChunk],
    k: int,
) -> bool:
    intervals: list[tuple[int, int]] = []
    for row in rows[:k]:
        chunk = chunks_by_id[row["chunk_id"]]
        if chunk.source != span.source:
            continue
        for chunk_start, chunk_end in chunk.source_intervals:
            start = max(span.char_start, chunk_start)
            end = min(span.char_end, chunk_end)
            if start < end:
                intervals.append((start, end))
    covered_until = span.char_start
    for start, end in sorted(intervals):
        if start > covered_until:
            break
        covered_until = max(covered_until, end)
        if covered_until >= span.char_end:
            return True
    return False


def evidence_recall(
    spans: list[Any],
    rows: list[dict[str, Any]],
    chunks_by_id: dict[str, EvalChunk],
    k: int,
) -> float:
    if not spans:
        return 0.0
    return sum(span_covered(span, rows, chunks_by_id, k) for span in spans) / len(spans)
