"""Drafter KB exposure contracts — DRAFTER_PACKED_EVIDENCE_V1 live in production."""

from __future__ import annotations

import hashlib
from typing import Any

DRAFTER_PACKED_EVIDENCE_V1 = "DRAFTER_PACKED_EVIDENCE_V1"
LEGACY_DRAFTER_KB_V1 = "LEGACY_DRAFTER_KB_V1"

LEGACY_DRAFTER_K = 3
LEGACY_DRAFTER_CHAR_LIMIT = 2000


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_packed_seed_groups(packed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve frozen Candidate-C pack order grouped by originating seed rank."""

    groups: list[dict[str, Any]] = []
    current_rank: int | None = None
    current_members: list[dict[str, Any]] = []
    for row in packed_rows:
        rank = row["seed_rank"]
        if current_rank is None:
            current_rank = rank
        if rank != current_rank:
            groups.append(
                {
                    "seed_rank": current_rank,
                    "seed_chunk_id": current_members[0]["seed_chunk_id"],
                    "members": current_members,
                }
            )
            current_rank = rank
            current_members = []
        current_members.append(row)
    if current_members:
        groups.append(
            {
                "seed_rank": current_rank,
                "seed_chunk_id": current_members[0]["seed_chunk_id"],
                "members": current_members,
            }
        )
    return groups


def _member_identity(row: dict[str, Any], units: dict[str, dict[str, Any]]) -> dict[str, Any]:
    unit = units[row["chunk_id"]]
    text = unit["retrieval_text"]
    return {
        "seed_rank": row["seed_rank"],
        "seed_chunk_id": row["seed_chunk_id"],
        "chunk_id": row["chunk_id"],
        "constituent_chunk_ids": [row["chunk_id"]],
        "relation": row["relation"],
        "source_intervals": row["source_intervals"],
        "source_path": row["source_path"],
        "document_id": row["document_id"],
        "document_version": row["document_version"],
        "source_sha256": row["source_sha256"],
        "serialized_text": text,
        "serialized_text_sha256": sha256_text(text),
        "cl100k_tokens": row.get("cl100k_tokens"),
    }


def serialize_drafter_packed_evidence_v1(
    packed_rows: list[dict[str, Any]],
    units: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """All surviving packed groups, full text, frozen pack order, no secondary clip."""

    groups = build_packed_seed_groups(packed_rows)
    serialized: list[dict[str, Any]] = []
    for group in groups:
        members = [_member_identity(row, units) for row in group["members"]]
        serialized.append(
            {
                "contract": DRAFTER_PACKED_EVIDENCE_V1,
                "seed_rank": group["seed_rank"],
                "seed_chunk_id": group["seed_chunk_id"],
                "members": members,
                "group_text_sha256": sha256_text(
                    "\n\n".join(member["serialized_text"] for member in members)
                ),
            }
        )
    return serialized


def serialize_legacy_drafter_kb(
    packed_rows: list[dict[str, Any]],
    units: dict[str, dict[str, Any]],
    *,
    k: int = LEGACY_DRAFTER_K,
    char_limit: int = LEGACY_DRAFTER_CHAR_LIMIT,
) -> list[dict[str, Any]]:
    """Production-equivalent top-K gate with per-seed cumulative character clipping."""

    groups = build_packed_seed_groups(packed_rows)
    serialized: list[dict[str, Any]] = []
    for group in groups:
        if group["seed_rank"] > k:
            continue
        remaining = char_limit
        members: list[dict[str, Any]] = []
        for row in group["members"]:
            unit = units[row["chunk_id"]]
            text = unit["retrieval_text"]
            if remaining <= 0:
                break
            text = text[:remaining]
            remaining -= len(text)
            member = _member_identity(row, units)
            member["serialized_text"] = text
            member["serialized_text_sha256"] = sha256_text(text)
            member["legacy_char_clip_applied"] = len(text) < len(unit["retrieval_text"])
            members.append(member)
        if members:
            serialized.append(
                {
                    "contract": LEGACY_DRAFTER_KB_V1,
                    "seed_rank": group["seed_rank"],
                    "seed_chunk_id": group["seed_chunk_id"],
                    "members": members,
                    "group_text_sha256": sha256_text(
                        "\n\n".join(member["serialized_text"] for member in members)
                    ),
                }
            )
    return serialized


def packed_identity_fingerprint(serialized_groups: list[dict[str, Any]]) -> str:
    payload = [
        {
            "seed_rank": group["seed_rank"],
            "seed_chunk_id": group["seed_chunk_id"],
            "member_chunk_ids": [member["chunk_id"] for member in group["members"]],
            "member_text_sha256": [
                member["serialized_text_sha256"] for member in group["members"]
            ],
        }
        for group in serialized_groups
    ]
    return sha256_text(repr(payload))
