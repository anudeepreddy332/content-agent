"""Deterministic coverage for source-level retrieval evaluation semantics."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import retrieval_eval
from scripts.retrieval_eval import (
    concept_coverage_at_k,
    hit_at_k,
    ndcg_at_k,
    reciprocal_rank,
    run_retrieval_eval,
    source_diversity_at_k,
    source_recall_at_k,
)
import tools.query_kb as query_kb_module


def test_hit_and_source_recall_distinguish_partial_multi_source_retrieval():
    expected = {"a", "b"}
    retrieved = ["a", "x", "y"]
    assert hit_at_k(retrieved, expected, 3) == 1.0
    assert source_recall_at_k(retrieved, expected, 3) == 0.5


def test_source_recall_is_one_when_all_expected_sources_are_retrieved():
    assert source_recall_at_k(["a", "b", "x"], {"a", "b"}, 3) == 1.0


def test_duplicate_relevant_chunks_receive_credit_once_and_ndcg_is_bounded():
    score = ndcg_at_k(["a", "a", "a"], {"a"}, 3)
    assert score == 1.0
    assert 0.0 <= score <= 1.0


def test_duplicate_occupancy_is_penalized_at_source_level():
    expected = {"a", "b"}
    duplicate_score = ndcg_at_k(["a", "a", "b"], expected, 3)
    diverse_score = ndcg_at_k(["a", "b", "x"], expected, 3)
    assert duplicate_score < diverse_score
    assert 0.0 <= duplicate_score <= 1.0
    assert 0.0 <= diverse_score <= 1.0


def test_source_diversity_counts_duplicate_sibling_slots_without_compressing_rank():
    assert source_diversity_at_k(["a", "a", "a", "b", "c"], 5) == (3, 2)


@pytest.mark.parametrize(
    "sources,expected,rr_oracle,ndcg_oracle",
    [
        (["a", "b", "c"], {"a"}, 1.0, 1.0),
        (["x", "a", "c"], {"a"}, 0.5, 0.631),
        (["x", "y", "z"], {"a"}, 0.0, 0.0),
        (["a", "x", "b"], {"a", "b"}, 1.0, 0.920),
    ],
)
def test_rank_metrics_against_hand_computed_oracles(sources, expected, rr_oracle, ndcg_oracle):
    assert reciprocal_rank(sources, expected, max_k=3) == pytest.approx(rr_oracle, abs=1e-3)
    score = ndcg_at_k(sources, expected, k=3)
    assert score == pytest.approx(ndcg_oracle, abs=1e-3)
    assert 0.0 <= score <= 1.0


def test_concept_coverage_reports_evidence_at_multiple_depths():
    results = [
        {"text": "alpha only"},
        {"text": "beta appears later"},
        {"text": "gamma completes the evidence"},
    ]
    concepts = ["alpha", "beta", "gamma"]
    assert concept_coverage_at_k(results, concepts, 1) == (["alpha"], pytest.approx(1 / 3))
    assert concept_coverage_at_k(results, concepts, 3) == (concepts, 1.0)


def test_harness_exposes_corrected_metrics_and_deprecates_fused_oos_gate(monkeypatch):
    tiny_golden = [
        {
            "query": "q-indomain",
            "expected_sources": ["doc-a", "doc-b"],
            "required_concepts": ["alpha", "beta"],
            "difficulty": "easy",
        },
        {
            "query": "q-oos",
            "expected_sources": [],
            "required_concepts": [],
            "difficulty": "out-of-scope",
            "min_distance_threshold": 0.5,
        },
    ]

    def fake_query_kb(query, n_results):
        if query == "q-indomain":
            return [
                {"source": "doc-a", "text": "alpha", "distance": 0.2, "chunk_index": 4},
                {"source": "other", "text": "beta", "distance": 0.3, "chunk_index": 2},
                {"source": "doc-a", "text": "alpha", "distance": 0.4, "chunk_index": 5},
            ]
        # A BM25-only hybrid result keeps the historical synthetic distance of 0.0.
        return [{"source": "bm25-only", "text": "t", "distance": 0.0, "chunk_index": 0, "rrf_score": 0.02}]

    def fake_diagnostics(query, n_results):
        assert query == "q-oos"
        return {
            "dense_top1_distance": None,
            "dense_top1_similarity": None,
            "bm25_top_score": 5.0,
            "hybrid_top1_rrf_score": 0.02,
            "hybrid_results": [{"rank": 1, "source": "bm25-only", "chunk_index": 0, "rrf_score": 0.02}],
        }

    monkeypatch.setattr(retrieval_eval, "GOLDEN_SET", tiny_golden)
    monkeypatch.setattr(retrieval_eval, "query_kb", fake_query_kb)
    monkeypatch.setattr(retrieval_eval, "query_kb_diagnostics", fake_diagnostics)

    results = run_retrieval_eval(k_values=[1, 3, 5])
    assert results["hit@3"] == 1.0
    assert results["source_recall@3"] == 0.5
    assert results["recall@3"] == 1.0  # legacy alias kept for historical reports
    assert results["ndcg@3"] < 1.0
    assert results["concept_coverage@1"] == 0.5
    assert results["concept_coverage@3"] == 1.0
    assert results["concept_pass@3"] == 1.0
    assert results["out_of_scope_rejection_rate"] is None
    assert results["per_query"][1]["oos_diagnostics"]["dense_top1_distance"] is None
    assert results["per_query"][1]["oos_diagnostics"]["bm25_top_score"] == 5.0
    assert results["per_query"][0]["retrieved_results"][0] == {"rank": 1, "source": "doc-a", "chunk_index": 4}


def test_all_normalized_harness_metrics_are_bounded(monkeypatch):
    monkeypatch.setattr(retrieval_eval, "GOLDEN_SET", [{
        "query": "q", "expected_sources": ["a"], "required_concepts": ["a"], "difficulty": "easy",
    }])
    monkeypatch.setattr(retrieval_eval, "query_kb", lambda query, n_results: [
        {"source": "a", "text": "a", "distance": 0.1, "chunk_index": 0},
        {"source": "a", "text": "a", "distance": 0.1, "chunk_index": 1},
    ])
    results = run_retrieval_eval(k_values=[1, 3])
    for key, value in results.items():
        if key.startswith(("hit@", "source_recall@", "ndcg@", "concept_coverage@", "concept_pass@")):
            assert 0.0 <= value <= 1.0


def test_raw_oos_diagnostics_keep_bm25_score_separate_from_synthetic_distance(monkeypatch):
    hybrid = [{"source": "bm25-only", "chunk_index": 0, "distance": 0.0, "rrf_score": 0.02}]
    bm25 = [{"source": "bm25-only", "chunk_index": 0, "distance": 0.0, "bm25_score": 5.0}]
    monkeypatch.setattr(
        query_kb_module,
        "_retrieve_components",
        lambda query, n_results, include_bm25_scores: (hybrid, [], bm25),
    )
    diagnostic = query_kb_module.query_kb_diagnostics("q", n_results=5)
    assert diagnostic["dense_top1_distance"] is None
    assert diagnostic["bm25_top_score"] == 5.0
    assert diagnostic["hybrid_top1_rrf_score"] == 0.02


def test_public_query_kb_keeps_the_hybrid_response_shape(monkeypatch):
    hybrid = [{"source": "doc", "chunk_index": 1, "text": "evidence", "distance": 0.2, "rrf_score": 0.03}]
    dense = [{**hybrid[0], "dense_similarity": 0.8}]
    monkeypatch.setattr(
        query_kb_module,
        "_retrieve_components",
        lambda query, n_results, include_bm25_scores=False: (hybrid, dense, []),
    )
    assert query_kb_module.query_kb("q", n_results=5) == hybrid
