"""Frozen Phase-5A0 Baseline A and A/B/C contract checks."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.phase5a0_baseline import (  # noqa: E402
    EvidenceSpan,
    LegacyChunk,
    evidence_span_recall,
    graded_ndcg_at_k,
    hit_at_k,
    precision_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
    source_recall_at_k,
)


REPO_ROOT = Path(__file__).parent.parent
REPORT_DIR = REPO_ROOT / "reports" / "phase5" / "phase5a0"


def _load(name: str) -> dict:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def _chunk(chunk_id: str, start: int, end: int) -> LegacyChunk:
    return LegacyChunk(
        ordinal=0,
        chunk_id=chunk_id,
        source="doc",
        source_path="doc.md",
        source_sha256="s" * 64,
        chunk_index=0,
        text="x" * (end - start),
        text_sha256="t" * 64,
        source_char_start=start,
        source_char_end=end,
        cl100k_tokens=1,
        wordpiece_content_tokens=1,
        tokens_beyond_limit=0,
        truncated=False,
        truncated_fraction=0.0,
        truncated_tail_char_start=None,
        boundary_decode_exact=True,
    )


def test_baseline_manifest_freezes_exact_source_chunk_and_model_identity():
    manifest = _load("baseline_a_manifest.json")
    assert manifest["starting_head"] == "f269ab760fc78f0b3a65618ae0c744649d1a0a2e"
    assert manifest["corpus"]["source_count"] == 20
    assert manifest["corpus"]["aggregate_fingerprint"] == (
        "724a8c3486cb482eadbd592a93918a6e2954bebdbdfa12a39753556b1c3d7c02"
    )
    assert manifest["chunking"]["chunk_count"] == 73
    assert manifest["chunking"]["ordered_chunk_fingerprint"] == (
        "c3976d83cc478d5f4352e680b59aac98ed314c5bbeb9f9ffee56e0ea51fbecbb"
    )
    assert manifest["embedding"]["local_resolved_revision"] == (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    assert manifest["embedding"]["historical_production_revision"] == "NOT_VERIFIABLE"
    assert manifest["embedding"]["effective_content_token_limit"] == 254
    assert manifest["embedding"]["dimension"] == 384
    assert manifest["dense"]["reference_search"].startswith("exhaustive")
    assert manifest["index_reference"]["serving_collection_mutated"] is False


def test_truncation_report_freezes_measured_exposure_and_tail_mapping():
    report = _load("legacy_chunk_truncation.json")
    assert report["chunk_count"] == 73
    assert report["truncation_exposed_chunks"] == 64
    assert report["truncation_exposed_percentage"] == pytest.approx(87.6712)
    assert report["severity_distribution"] == {
        "low_1_25_percent": 8,
        "medium_25_50_percent": 56,
    }
    assert report["evidence_tail_affected_query_count"] == 24
    assert len(report["chunks"]) == 73
    assert all(row["wordpiece_content_tokens"] >= 0 for row in report["chunks"])


def test_baseline_report_separates_gating_diagnostic_and_absence_semantics():
    report = _load("baseline_a_report.json")
    assert len(report["per_query"]) == 35
    assert report["absence_metrics"] is None
    for channel in ("dense", "bm25", "hybrid"):
        assert report["aggregate_metrics"][channel]["all_35_diagnostic"]["query_count"] == 35
        assert (
            report["aggregate_metrics"][channel]["gating_33_excludes_q25_q26"][
                "query_count"
            ]
            == 33
        )
    by_id = {row["query_id"]: row for row in report["per_query"]}
    assert by_id["Q25"]["gating_eligible"] is False
    assert by_id["Q26"]["gating_eligible"] is False
    assert report["evidence_exposure"]["lost_after_retrieval_for_drafter"] == 2
    assert report["evidence_exposure"]["lost_after_retrieval_for_verifier"] == 0


def test_hand_checkable_source_metrics_and_graded_ndcg():
    rows = [
        {"source": "grade1"},
        {"source": "irrelevant"},
        {"source": "grade2"},
        {"source": "grade2"},
    ]
    relevance = {"grade2": 2, "grade1": 1}
    assert hit_at_k(rows, relevance, 1) == 1.0
    assert source_recall_at_k(rows, relevance, 1) == 0.5
    assert source_recall_at_k(rows, relevance, 3) == 1.0
    assert precision_at_k(rows, relevance, 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(rows, relevance, 3) == 1.0
    expected_dcg = 1 / math.log2(2) + 3 / math.log2(4)
    expected_idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert graded_ndcg_at_k(rows, relevance, 3) == pytest.approx(
        expected_dcg / expected_idcg
    )


def test_rrf_fuses_by_chunk_id_and_has_deterministic_tie_break():
    dense = [
        {"chunk_id": "b", "source": "s", "chunk_index": 1},
        {"chunk_id": "a", "source": "s", "chunk_index": 0},
    ]
    bm25 = [
        {"chunk_id": "a", "source": "s", "chunk_index": 0},
        {"chunk_id": "c", "source": "t", "chunk_index": 0},
    ]
    fused = reciprocal_rank_fusion(dense, bm25)
    assert [row["chunk_id"] for row in fused] == ["a", "b", "c"]
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["bm25_rank"] == 1
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 60)


def test_evidence_span_recall_uses_union_of_exact_source_intervals():
    first = _chunk("first", 0, 60)
    second = _chunk("second", 50, 100)
    by_id = {first.chunk_id: first, second.chunk_id: second}
    rows = [{"chunk_id": "first"}, {"chunk_id": "second"}]
    span = EvidenceSpan(
        query_id="Q",
        source="doc",
        grade=2,
        span_id="doc:50-70",
        char_start=50,
        char_end=70,
        line_start=1,
        line_end=1,
    )
    assert evidence_span_recall([span], rows, by_id, 1) == 0.0
    assert evidence_span_recall([span], rows, by_id, 2) == 1.0


def test_abc_contract_freezes_safe_children_expansion_and_promotion_rules():
    contract = json.loads(
        (REPO_ROOT / "evals" / "fixtures" / "phase5a0_abc_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["candidate_b"]["target_token_policy"]["hard_max_content_tokens"] == 254
    assert contract["candidate_b"]["target_token_policy"]["silent_truncation_allowed"] is False
    assert contract["candidate_b"]["size_sweep"] is None
    assert contract["candidate_c"]["retrieval_children_and_ranks"].startswith("bit-identical")
    assert contract["ranker_contract"]["raw_text_identity"] is False
    assert contract["ranker_contract"]["native_score_averaging"] is False
    assert "zero silent embedding truncation" in contract["promotion_contract"]["required"]
    assert contract["scope"]["production_qdrant_mutation"] is False
