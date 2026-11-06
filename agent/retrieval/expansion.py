"""Bounded ±1 neighbor expansion and seed-first packing for CSWP retrieval."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import tiktoken

from agent.retrieval.chunks import EvalChunk, normalize_interval, source_offsets

PACK_BUDGET_CL100K = 2000
ROOT = Path(__file__).resolve().parent.parent.parent


class ExpansionError(RuntimeError):
    """CSWP expansion or packing invariant failed."""


def cl100k():
    return tiktoken.get_encoding("cl100k_base")


def cl100k_count(text: str, encoding=None) -> int:
    return len((encoding or cl100k()).encode(text))


def same_document(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a["document_id"] == b["document_id"]
        and a["document_version"] == b["document_version"]
        and a["source_path"] == b["source_path"]
        and a["source_sha256"] == b["source_sha256"]
    )


def interval_key(unit: dict[str, Any]) -> tuple[Any, ...]:
    return (
        unit["source_path"],
        tuple(
            (s["source_char_start"], s["source_char_end"], s.get("role"))
            for s in unit["source_spans"]
        ),
    )


def neighbor_of(
    seed: dict[str, Any], by_source: dict[str, list[dict[str, Any]]], delta: int
) -> dict[str, Any] | None:
    rows = by_source[seed["source_path"]]
    index = next(
        i for i, unit in enumerate(rows) if unit["chunk_id"] == seed["chunk_id"]
    )
    target = index + delta
    if target < 0 or target >= len(rows):
        return None
    unit = rows[target]
    if not same_document(seed, unit):
        raise ExpansionError("adjacent CSWP unit crossed document identity")
    if abs(target - index) != 1:
        raise ExpansionError("neighbor is not immediately adjacent")
    return unit


def expand_seeds(
    seeds: list[dict[str, Any]],
    units: dict[str, dict[str, Any]],
    by_source: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    groups = []
    for seed_row in seeds:
        seed = units[seed_row["chunk_id"]]
        members = []
        for delta, relation in ((-1, "PREVIOUS"), (0, "SEED"), (1, "NEXT")):
            unit = seed if relation == "SEED" else neighbor_of(seed, by_source, delta)
            if unit is None:
                continue
            members.append(
                {
                    "chunk_id": unit["chunk_id"],
                    "relation": relation,
                    "seed_rank": seed_row["rank"],
                    "seed_chunk_id": seed["chunk_id"],
                    "source_path": unit["source_path"],
                    "document_id": unit["document_id"],
                    "document_version": unit["document_version"],
                    "source_sha256": unit["source_sha256"],
                    "retrieval_char_start": unit["retrieval_char_start"],
                    "retrieval_char_end": unit["retrieval_char_end"],
                    "source_intervals": [
                        (s["source_char_start"], s["source_char_end"])
                        for s in unit["source_spans"]
                    ],
                    "interval_key": interval_key(unit),
                }
            )
        members.sort(
            key=lambda row: (
                row["retrieval_char_start"],
                row["retrieval_char_end"],
                row["chunk_id"],
            )
        )
        groups.append(
            {
                "seed_rank": seed_row["rank"],
                "seed_chunk_id": seed["chunk_id"],
                "members": members,
            }
        )
    return groups


def dedupe_groups(groups: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, dict]:
    seen_ids: set[str] = set()
    seen_intervals: dict[tuple[Any, ...], str] = {}
    dropped_duplicate_ids = 0
    kept = []
    additional: dict[str, list] = defaultdict(list)
    for group in groups:
        members = []
        for row in group["members"]:
            cid = row["chunk_id"]
            key = row["interval_key"]
            if cid in seen_ids:
                dropped_duplicate_ids += 1
                additional[cid].append(
                    {
                        "seed_rank": row["seed_rank"],
                        "seed_chunk_id": row["seed_chunk_id"],
                        "relation": row["relation"],
                    }
                )
                continue
            if key in seen_intervals and seen_intervals[key] != cid:
                raise ExpansionError(
                    "identical source interval under a different chunk_id"
                )
            seen_ids.add(cid)
            seen_intervals[key] = cid
            members.append(row)
        kept.append({**group, "members": members})
    return kept, dropped_duplicate_ids, dict(additional)


def flatten(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        rows.extend(group["members"])
    return rows


def pack_units_seed_first(
    rows: list[dict[str, Any]],
    units: dict[str, dict[str, Any]],
    budget: int,
    encoding,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, bool, dict[str, Any]]:
    seed_rows = []
    seen_seed_chunk: set[str] = set()
    for row in rows:
        if row["relation"] != "SEED":
            continue
        cid = row["chunk_id"]
        if cid in seen_seed_chunk:
            continue
        seen_seed_chunk.add(cid)
        seed_rows.append(row)
    seed_rows.sort(key=lambda row: (row["seed_rank"], row["chunk_id"]))

    seed_costs = [
        (row, cl100k_count(units[row["chunk_id"]]["retrieval_text"], encoding))
        for row in seed_rows
    ]
    total_seed_cost = sum(cost for _, cost in seed_costs)
    if total_seed_cost > budget:
        raise ExpansionError(
            f"seed-only cl100k tokens {total_seed_cost} exceed budget {budget}"
        )

    packed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    packed_ids: set[str] = set()
    used = 0
    for row, cost in seed_costs:
        packed.append({**row, "cl100k_tokens": cost})
        packed_ids.add(row["chunk_id"])
        used += cost

    neighbor_rows = [row for row in rows if row["chunk_id"] not in packed_ids]
    neighbor_rows.sort(
        key=lambda row: (
            row["seed_rank"],
            row.get(
                "retrieval_char_start",
                units[row["chunk_id"]].get("retrieval_char_start", 0),
            ),
            row.get(
                "retrieval_char_end",
                units[row["chunk_id"]].get("retrieval_char_end", 0),
            ),
            row["chunk_id"],
        )
    )
    exhausted = False
    neighbors_retained = 0
    for row in neighbor_rows:
        cost = cl100k_count(units[row["chunk_id"]]["retrieval_text"], encoding)
        if used + cost > budget:
            exhausted = True
            skipped.append({**row, "cl100k_tokens": cost, "skip_reason": "PACK_BUDGET"})
            continue
        packed.append({**row, "cl100k_tokens": cost})
        packed_ids.add(row["chunk_id"])
        used += cost
        neighbors_retained += 1

    seed_ids = {row["chunk_id"] for row in seed_rows}
    if not seed_ids.issubset(packed_ids):
        raise ExpansionError("seed-preservation invariant violated")

    stats = {
        "policy": "SEED_FIRST",
        "seeds_retained": len(seed_rows),
        "neighbors_retained": neighbors_retained,
        "seed_only_cl100k": total_seed_cost,
    }
    return packed, skipped, used, exhausted, stats


def assert_seed_preservation_invariant(
    rows: list[dict[str, Any]],
    units: dict[str, dict[str, Any]],
    budget: int,
    encoding,
) -> None:
    seed_rows = []
    seen: set[str] = set()
    for row in rows:
        if row["relation"] != "SEED" or row["chunk_id"] in seen:
            continue
        seen.add(row["chunk_id"])
        seed_rows.append(row)
    total = sum(
        cl100k_count(units[row["chunk_id"]]["retrieval_text"], encoding)
        for row in seed_rows
    )
    if total > budget:
        return
    packed, _, _, _, stats = pack_units_seed_first(rows, units, budget, encoding)
    packed_seeds = {row["chunk_id"] for row in packed if row["relation"] == "SEED"}
    expected = {row["chunk_id"] for row in seed_rows}
    if packed_seeds != expected:
        raise ExpansionError("seed-preservation invariant failed")
    if stats["seeds_retained"] != len(seed_rows):
        raise ExpansionError("seed count mismatch after seed-first pack")


def eval_chunks_from_manifest(
    manifest: dict[str, Any], root: Path = ROOT
) -> list[EvalChunk]:
    offsets = {
        unit["source_path"]: source_offsets(
            (root / unit["source_path"]).read_text(encoding="utf-8")
        )
        for unit in manifest["children"]
    }
    chunks: list[EvalChunk] = []
    for ordinal, unit in enumerate(manifest["children"]):
        intervals = tuple(
            interval
            for span in unit["source_spans"]
            if (
                interval := normalize_interval(
                    span["source_char_start"],
                    span["source_char_end"],
                    *offsets[unit["source_path"]],
                )
            )
            is not None
        )
        chunks.append(
            EvalChunk(
                ordinal=ordinal,
                chunk_id=unit["chunk_id"],
                source=Path(unit["source_path"]).stem,
                source_path=unit["source_path"],
                retrieval_text=unit["retrieval_text"],
                source_intervals=intervals,
                embedding_content_token_count=unit["embedding_content_token_count"],
            )
        )
    return chunks
