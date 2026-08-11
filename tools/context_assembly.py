"""Bounded post-retrieval evidence-window construction for MiniLM child hits.

Children stay small for retrieval.  This module only combines already-indexed
children after ranking; it never embeds expanded text or changes retrieval scores.
"""

from math import ceil


DEFAULT_NEIGHBOR_RADIUS = 1
MAX_EVIDENCE_WINDOW_CHARS = 2400
CHARS_PER_TOKEN_ESTIMATE = 4
_MIN_OVERLAP_CHARS = 20


def _common_boundary_overlap(left: str, right: str) -> int:
    """Return a meaningful exact suffix/prefix overlap between adjacent children."""
    maximum = min(len(left), len(right))
    for length in range(maximum, _MIN_OVERLAP_CHARS - 1, -1):
        if left.endswith(right[:length]):
            return length
    return 0


def _join_children(children: list[dict], max_chars: int) -> tuple[str, int, int]:
    """Join ordered children once, removing exact overlap and enforcing a hard cap."""
    assembled = ""
    duplicate_chars_removed = 0
    for child in children:
        text = child.get("text", "")
        if not text:
            continue
        if text == assembled or text in assembled:
            duplicate_chars_removed += len(text)
            continue
        overlap = _common_boundary_overlap(assembled, text) if assembled else 0
        duplicate_chars_removed += overlap
        assembled += text[overlap:]

    truncated_chars = max(0, len(assembled) - max_chars)
    return assembled[:max_chars], duplicate_chars_removed, truncated_chars


def _merge_seed_intervals(seed_intervals: list[tuple[int, int, dict]]) -> list[tuple[int, int, list[dict]]]:
    """Merge overlapping or adjacent source-local evidence intervals."""
    merged: list[tuple[int, int, list[dict]]] = []
    for start, end, seed in sorted(seed_intervals, key=lambda value: (value[0], value[1])):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end, [seed]))
            continue
        current_start, current_end, current_seeds = merged[-1]
        merged[-1] = (current_start, max(current_end, end), current_seeds + [seed])
    return merged


def assemble_evidence_windows(
    ranked_children: list[dict],
    source_children: dict[str, list[dict]],
    *,
    neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
    max_windows: int = 5,
    max_window_chars: int = MAX_EVIDENCE_WINDOW_CHARS,
) -> list[dict]:
    """Consolidate ranked child hits into bounded, source-local evidence windows.

    ``source_children`` is the deterministic complete child sequence for each source.
    Every returned window is ordered by the best raw child rank it represents;
    repeated sibling hits only add seeds to the same local window.
    """
    if neighbor_radius < 0 or max_windows < 0 or max_window_chars <= 0:
        raise ValueError("Neighbor radius, window count, and character cap must be valid")

    by_source: dict[str, list[dict]] = {}
    for rank, child in enumerate(ranked_children, start=1):
        source = child.get("source")
        chunk_index = child.get("chunk_index")
        if source is None or not isinstance(chunk_index, int):
            continue
        enriched = {**child, "_retrieval_rank": rank}
        by_source.setdefault(source, []).append(enriched)

    windows: list[dict] = []
    for source, seeds in by_source.items():
        children = sorted(source_children.get(source, []), key=lambda child: child.get("chunk_index", 0))
        children_by_index = {child.get("chunk_index"): child for child in children}
        if not children_by_index:
            continue
        last_index = max(children_by_index)
        intervals = [
            (
                max(0, seed["chunk_index"] - neighbor_radius),
                min(last_index, seed["chunk_index"] + neighbor_radius),
                seed,
            )
            for seed in seeds
            if seed["chunk_index"] in children_by_index
        ]
        for start, end, interval_seeds in _merge_seed_intervals(intervals):
            window_children = [
                children_by_index[index]
                for index in range(start, end + 1)
                if index in children_by_index
            ]
            text, duplicate_chars_removed, truncated_chars = _join_children(
                window_children,
                max_window_chars,
            )
            if not text:
                continue
            best_seed = min(
                interval_seeds,
                key=lambda seed: (seed["_retrieval_rank"], -seed.get("rrf_score", 0.0)),
            )
            windows.append({
                "text": text,
                "source": source,
                "chunk_index": start,
                "chunk_indices": [child["chunk_index"] for child in window_children],
                "seed_chunk_indices": sorted({seed["chunk_index"] for seed in interval_seeds}),
                "seed_ranks": sorted(seed["_retrieval_rank"] for seed in interval_seeds),
                "best_child_rank": best_seed["_retrieval_rank"],
                "distance": best_seed.get("distance"),
                "rrf_score": best_seed.get("rrf_score"),
                "duplicate_chars_removed": duplicate_chars_removed,
                "truncated_chars": truncated_chars,
                "context_chars": len(text),
                "max_window_chars": max_window_chars,
            })

    windows.sort(key=lambda window: (
        window["best_child_rank"],
        -(window.get("rrf_score") or 0.0),
        window["source"],
        window["chunk_index"],
    ))
    return windows[:max_windows]


def context_budget_stats(windows: list[dict], n_windows: int) -> dict:
    """Measure the bounded context passed to a consumer without mutating it."""
    selected = windows[:n_windows]
    total_chars = sum(window.get("context_chars", len(window.get("text", ""))) for window in selected)
    return {
        "evidence_windows": len(selected),
        "total_context_chars": total_chars,
        "estimated_context_tokens": ceil(total_chars / CHARS_PER_TOKEN_ESTIMATE),
        "unique_sources": len({window.get("source") for window in selected}),
        "duplicate_chars_removed": sum(window.get("duplicate_chars_removed", 0) for window in selected),
        "truncated_chars": sum(window.get("truncated_chars", 0) for window in selected),
        "max_window_chars": max(
            (window.get("max_window_chars", MAX_EVIDENCE_WINDOW_CHARS) for window in selected),
            default=MAX_EVIDENCE_WINDOW_CHARS,
        ),
    }
