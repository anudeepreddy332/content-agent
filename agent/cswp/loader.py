"""Load production-owned CSWP index artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent.cswp.constants import (
    CURRENT_CORPUS_REPRESENTATION_FINGERPRINT,
    MANIFEST_FILE,
    PRODUCTION_INDEX_DIR,
    UNITS_FILE,
)
from agent.cswp.errors import CSWPError


def load_units_jsonl(path: Path) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                units.append(json.loads(line))
    return units


def load_production_index(
    index_dir: Path = PRODUCTION_INDEX_DIR,
    *,
    require_current_corpus_fingerprint: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest_path = index_dir / MANIFEST_FILE
    units_path = index_dir / UNITS_FILE
    if not manifest_path.is_file() or not units_path.is_file():
        raise CSWPError("production CSWP index missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = manifest.get("representation_fingerprint")
    if require_current_corpus_fingerprint and (
        fingerprint != CURRENT_CORPUS_REPRESENTATION_FINGERPRINT
    ):
        raise CSWPError("production CSWP representation fingerprint mismatch")
    units_list = load_units_jsonl(units_path)
    expected = manifest.get("unit_count")
    if expected is not None and expected != len(units_list):
        raise CSWPError("production CSWP unit count mismatch")
    units = {unit["chunk_id"]: unit for unit in units_list}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units_list:
        by_source[unit["source_path"]].append(unit)
    for rows in by_source.values():
        starts = [row["retrieval_char_start"] for row in rows]
        if starts != sorted(starts):
            raise CSWPError("production CSWP source order is not deterministic")
    representation = {
        "candidate": "CSWP",
        "schema_version": "phase5a2e_representation_v1",
        "packing_version": manifest["packing_version"],
        "contract_sha256": manifest["contract_sha256"],
        "frozen_B_fingerprint": manifest["frozen_B_fingerprint"],
        "children": [
            {
                key: value
                for key, value in unit.items()
                if key
                not in ("reading_order_ordinal", "previous_chunk_id", "next_chunk_id")
            }
            for unit in units_list
        ],
    }
    return representation, units, dict(by_source)


def production_index_available(index_dir: Path = PRODUCTION_INDEX_DIR) -> bool:
    return (index_dir / MANIFEST_FILE).is_file() and (index_dir / UNITS_FILE).is_file()
