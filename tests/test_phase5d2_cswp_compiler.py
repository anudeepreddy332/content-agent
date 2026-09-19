"""Phase 5D2 production CSWP compiler parity and qualification gates."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from agent.cswp.compiler import (
    attach_neighbors,
    compile_and_write,
    compile_corpus,
    strip_neighbor_fields,
    validate_neighbors,
    write_index,
)
from agent.cswp.constants import (
    CURRENT_CORPUS_REPRESENTATION_FINGERPRINT,
    HISTORICAL_CSWP_MANIFEST,
    PRODUCTION_INDEX_DIR,
    ROOT,
)
from agent.cswp.identity import sha256_json
from agent.cswp.loader import load_production_index
from agent.cswp.offline import install_offline_guard
from scripts import phase5a2_shadow_ab as ab


@pytest.fixture(scope="module")
def historical_manifest():
    return json.loads(HISTORICAL_CSWP_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compiled_once():
    install_offline_guard()
    return compile_corpus(root=ROOT)


def test_current_corpus_produces_159_units_and_historical_fingerprint(compiled_once):
    production_manifest, pack_report, units, _structural = compiled_once
    assert pack_report["retrieval_chunks"] == 159
    assert production_manifest["unit_count"] == 159
    assert (
        production_manifest["representation_fingerprint"]
        == CURRENT_CORPUS_REPRESENTATION_FINGERPRINT
    )
    assert pack_report["children_above_254"] == 0
    assert pack_report["deterministic_fingerprint"] == CURRENT_CORPUS_REPRESENTATION_FINGERPRINT


def test_chunk_id_and_retrieval_text_parity(compiled_once, historical_manifest):
    _, _, units, _ = compiled_once
    historical = historical_manifest["children"]
    assert len(units) == len(historical)
    for compiled, reference in zip(
        [strip_neighbor_fields(unit) for unit in units],
        historical,
        strict=True,
    ):
        assert compiled["chunk_id"] == reference["chunk_id"]
        assert compiled["retrieval_text"] == reference["retrieval_text"]
        assert compiled["source_spans"] == reference["source_spans"]
        assert compiled["embedding_content_token_count"] == reference[
            "embedding_content_token_count"
        ]


def test_source_span_reconstruction(compiled_once):
    from agent.cswp.source import Source

    _, _, units, _ = compiled_once
    for unit in units:
        source = Source((ROOT / unit["source_path"]).read_bytes())
        core = source.text[unit["core_char_start"] : unit["core_char_end"]]
        assert core == unit["canonical_content"]
        for span in unit["source_spans"]:
            extracted = source.text[span["source_char_start"] : span["source_char_end"]]
            assert extracted == source.extract(span)


def test_neighbor_metadata_is_valid(compiled_once):
    _, _, units, _ = compiled_once
    validate_neighbors(units)
    for unit in units:
        assert "reading_order_ordinal" in unit
        assert "previous_chunk_id" in unit
        assert "next_chunk_id" in unit


def test_two_clean_compilations_are_byte_identical(tmp_path):
    install_offline_guard()
    out_a = tmp_path / "run-a"
    out_b = tmp_path / "run-b"
    manifest_a, _, units_a = compile_and_write(output_dir=out_a, root=ROOT)
    manifest_b, _, units_b = compile_and_write(output_dir=out_b, root=ROOT)
    assert manifest_a == manifest_b
    assert (out_a / "units.jsonl").read_bytes() == (out_b / "units.jsonl").read_bytes()
    assert units_a == units_b


def test_production_loader_matches_historical_representation(historical_manifest, tmp_path):
    install_offline_guard()
    compile_and_write(output_dir=tmp_path, root=ROOT)
    representation, units, by_source = load_production_index(
        tmp_path, require_current_corpus_fingerprint=True
    )
    assert sha256_json(representation) == CURRENT_CORPUS_REPRESENTATION_FINGERPRINT
    assert len(representation["children"]) == 159
    assert len(units) == 159
    assert representation["children"] == historical_manifest["children"]
    for rows in by_source.values():
        assert [row["retrieval_char_start"] for row in rows] == sorted(
            row["retrieval_char_start"] for row in rows
        )


def test_compiler_cli_rebuilds_index(tmp_path):
    install_offline_guard()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent.cswp",
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["unit_count"] == 159
    assert (
        payload["representation_fingerprint"]
        == CURRENT_CORPUS_REPRESENTATION_FINGERPRINT
    )
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "units.jsonl").is_file()


def test_no_silent_truncation(compiled_once):
    _, pack_report, units, _ = compiled_once
    assert pack_report["children_above_254"] == 0
    assert max(unit["embedding_content_token_count"] for unit in units) <= 254
    assert all(unit["embedding_total_token_count"] <= 256 for unit in units)


def test_attach_neighbors_preserves_core_fields(compiled_once):
    _, _, units, _ = compiled_once
    core = [strip_neighbor_fields(unit) for unit in units]
    enriched = attach_neighbors(core)
    assert [strip_neighbor_fields(unit) for unit in enriched] == core
