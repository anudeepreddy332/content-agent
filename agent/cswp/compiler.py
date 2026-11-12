"""Production CSWP ingest compiler: source bytes to qualified retrieval units."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.cswp.constants import (
    BASELINE_MANIFEST,
    COMPILER_VERSION,
    CSWP_CONTRACT,
    MANIFEST_FILE,
    PRODUCTION_INDEX_DIR,
    ROOT,
    UNITS_FILE,
)
from agent.cswp.identity import canonical_json, sha, sha256_json
from agent.cswp.packer import pack
from agent.cswp.structural import build_corpus
from agent.cswp.tokenizer import MiniLMTokenizer

NEIGHBOR_FIELDS = ("reading_order_ordinal", "previous_chunk_id", "next_chunk_id")


def strip_neighbor_fields(unit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in unit.items() if key not in NEIGHBOR_FIELDS}


def attach_neighbors(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive same-document neighbor links without reordering the packed unit list."""
    by_document: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        key = (unit["document_version"], unit["source_path"])
        by_document[key].append(unit)
    neighbor_links: dict[str, tuple[str | None, str | None]] = {}
    for rows in by_document.values():
        ordered = sorted(
            rows,
            key=lambda row: (row["retrieval_char_start"], row["chunk_id"]),
        )
        for index, unit in enumerate(ordered):
            neighbor_links[unit["chunk_id"]] = (
                ordered[index - 1]["chunk_id"] if index else None,
                ordered[index + 1]["chunk_id"] if index + 1 < len(ordered) else None,
            )
    enriched: list[dict[str, Any]] = []
    for ordinal, unit in enumerate(units):
        previous_chunk_id, next_chunk_id = neighbor_links[unit["chunk_id"]]
        row = dict(unit)
        row["reading_order_ordinal"] = ordinal
        row["previous_chunk_id"] = previous_chunk_id
        row["next_chunk_id"] = next_chunk_id
        enriched.append(row)
    if len(enriched) != len(units):
        raise RuntimeError("neighbor attachment changed unit count")
    return enriched


def validate_neighbors(units: list[dict[str, Any]]) -> None:
    by_id = {unit["chunk_id"]: unit for unit in units}
    for unit in units:
        for edge, delta in (("previous_chunk_id", -1), ("next_chunk_id", 1)):
            neighbor_id = unit[edge]
            if neighbor_id is None:
                continue
            neighbor = by_id[neighbor_id]
            if (
                neighbor["document_id"] != unit["document_id"]
                or neighbor["document_version"] != unit["document_version"]
                or neighbor["source_path"] != unit["source_path"]
            ):
                raise RuntimeError("neighbor crossed document identity boundary")
            reverse = "next_chunk_id" if edge == "previous_chunk_id" else "previous_chunk_id"
            if neighbor[reverse] != unit["chunk_id"]:
                raise RuntimeError("neighbor back-link mismatch")


def build_representation_manifest(
    units: list[dict[str, Any]],
    manifest_b: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate": contract["candidate"],
        "schema_version": "phase5a2e_representation_v1",
        "packing_version": contract["packing_version"],
        "contract_sha256": sha256_json(contract),
        "frozen_B_fingerprint": sha(canonical_json(manifest_b)),
        "children": [strip_neighbor_fields(unit) for unit in units],
    }


def compile_corpus(
    *,
    root: Path = ROOT,
    source_root: Path | None = None,
    contract_path: Path = CSWP_CONTRACT,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Compile the frozen seed corpus into CSWP retrieval units."""
    if source_root is not None and source_root.resolve() != (root / "kb/seed_docs").resolve():
        raise RuntimeError("compiler reads kb/seed_docs via frozen baseline manifest only")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tokenizer = MiniLMTokenizer()
    manifest_b, structural_report = build_corpus(tokenizer, root=root)
    representation, pack_report = pack(manifest_b, tokenizer, contract, root=root)
    units = attach_neighbors(representation["children"])
    validate_neighbors(units)
    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    production_manifest = {
        "schema_version": "cswp_v1_production_manifest",
        "compiler_version": COMPILER_VERSION,
        "packing_version": contract["packing_version"],
        "contract_sha256": sha256_json(contract),
        "parser_name": manifest_b["parser_name"],
        "parser_version": manifest_b["parser_version"],
        "chunker_version": manifest_b["chunker_version"],
        "serialization_version": manifest_b["serialization_version"],
        "tokenizer": manifest_b["tokenizer"],
        "baseline_corpus_fingerprint": baseline["corpus"]["aggregate_fingerprint"],
        "frozen_B_fingerprint": representation["frozen_B_fingerprint"],
        "unit_count": len(units),
        "representation_fingerprint": sha256_json(representation),
        "units_file": UNITS_FILE,
        "provider_calls": 0,
        "external_network_calls": 0,
        "qdrant_writes": 0,
    }
    return production_manifest, pack_report, units, structural_report


def write_index(
    output_dir: Path,
    production_manifest: dict[str, Any],
    units: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    units_path = output_dir / UNITS_FILE
    with units_path.open("w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps(unit, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    manifest = dict(production_manifest)
    manifest["compiled_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output_dir / MANIFEST_FILE).write_bytes(manifest_bytes)


def compile_and_write(
    *,
    output_dir: Path = PRODUCTION_INDEX_DIR,
    root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    production_manifest, pack_report, units, _structural_report = compile_corpus(root=root)
    write_index(output_dir, production_manifest, units)
    return production_manifest, pack_report, units
