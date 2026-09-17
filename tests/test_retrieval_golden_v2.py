"""Deterministic validation for retrieval golden set v2."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.retrieval_golden_v2 import (
    BASELINE_IDENTITY_PATH,
    ORACLE_PATH,
    RetrievalGoldenV2Error,
    load_baseline_identity,
    load_oracle,
    load_v1_golden_set,
    sha256_bytes,
    validate_oracle,
)


def test_v2_oracle_validates_clean():
    summary = validate_oracle()
    assert summary["query_count"] == 35
    assert summary["answerability"]["ABSENT"] == 0
    assert summary["holdout"] == 0
    assert summary["gating_eligible"] == 33
    assert summary["non_gating"] == 2
    assert summary["invalid_source_refs"] == 0
    assert summary["quote_span_mismatches"] == 0
    assert summary["evidence_spans_validated"] > 0
    assert summary["repo_head"] == "86712d200fd46382a876f2083f824f03a83341ab"


def test_v2_preserves_exact_v1_query_text():
    payload = load_oracle()
    v1 = load_v1_golden_set()
    for query in payload["queries"]:
        assert query["query"] == v1[query["v1_index"]]["query"]


def test_v2_adjudication_summary_matches_queries():
    payload = load_oracle()
    summary = payload["adjudication_summary"]
    assert summary["v1_label_verdict"] == {
        "CORRECT": 7,
        "INCOMPLETE": 20,
        "WRONG": 6,
        "AMBIGUOUS": 2,
    }
    assert summary["answerability"]["ABSENT"] == 0
    assert summary["split"]["holdout"] == 0
    assert summary["non_gating"] == 2


def test_q25_q26_non_gating_ambiguous():
    payload = load_oracle()
    by_id = {q["query_id"]: q for q in payload["queries"]}
    for qid in ("Q25", "Q26"):
        q = by_id[qid]
        assert q["gating_eligible"] is False
        assert q["v1_label_verdict"] == "AMBIGUOUS"


def test_q31_q35_partial_not_absent():
    payload = load_oracle()
    for i in range(31, 36):
        q = next(q for q in payload["queries"] if q["query_id"] == f"Q{i:02d}")
        assert q["answerability"] == "PARTIAL"
        assert q["v1_label_verdict"] == "WRONG"
        assert q["query_bucket"] == "former-oos"


def test_baseline_a_identity_sidecar():
    identity = load_baseline_identity()
    assert identity["repo_head"] == "86712d200fd46382a876f2083f824f03a83341ab"
    assert identity["corpus"]["clean_reconstructed_chunk_count"] == 73
    assert identity["embedding"]["local_resolved_revision"] == (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    assert identity["embedding"]["historical_production_revision"] == "NOT_VERIFIABLE"
    assert identity["qdrant_control_identity"]["live_server_identity"] == "NOT_VERIFIABLE"


def test_oracle_sha256_frozen():
    observed = sha256_bytes(ORACLE_PATH.read_bytes())
    assert observed == "57396e7c58698ae140104f6cb51541d3b7815bf795b431a3d80bf035c8e79cf9"


def test_v1_evaluator_file_unchanged():
    v1_path = Path(__file__).parent.parent / "scripts/retrieval_eval.py"
    text = v1_path.read_text(encoding="utf-8")
    assert "GOLDEN_SET = [" in text
    assert "retrieval_golden_v2" not in text
