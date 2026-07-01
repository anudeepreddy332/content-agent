"""Unit tests for the rank-sensitive retrieval metrics (MRR / nDCG@k) added to
scripts/retrieval_eval.py. The pure helpers are tested against hand-computed
oracles; the harness-level test monkeypatches query_kb over a tiny GOLDEN_SET so
it stays offline ($0, no network/LLM call) and asserts the new aggregate keys
appear without disturbing the existing recall@k numbers."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import retrieval_eval
from scripts.retrieval_eval import reciprocal_rank, ndcg_at_k, run_retrieval_eval


@pytest.mark.parametrize(
    "sources,expected,rr_oracle,ndcg_oracle",
    [
        (["a", "b", "c"], {"a"}, 1.0, 1.0),
        (["x", "a", "c"], {"a"}, 0.5, 0.631),
        (["x", "y", "z"], {"a"}, 0.0, 0.0),
        (["a", "x", "b"], {"a", "b"}, 1.0, 0.920),
    ],
)
def test_helpers_against_oracles(sources, expected, rr_oracle, ndcg_oracle):
    assert reciprocal_rank(sources, expected, max_k=3) == pytest.approx(rr_oracle, abs=1e-3)
    assert ndcg_at_k(sources, expected, k=3) == pytest.approx(ndcg_oracle, abs=1e-3)


def test_ndcg_fewer_results_than_k():
    # Only two results, but k=3 — DCG sum runs to len(sources), IDCG to min(1,3)=1.
    # rel=[0,1]; DCG = 1/log2(3) = 0.631; IDCG = 1/log2(2) = 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, k=3) == pytest.approx(0.631, abs=1e-3)


def test_reciprocal_rank_no_match_is_zero():
    assert reciprocal_rank(["x", "y"], {"a"}, max_k=5) == 0.0


def test_harness_populates_new_metrics_and_preserves_recall(monkeypatch):
    # Tiny offline golden set: one easy in-domain, one out-of-scope.
    tiny_golden = [
        {
            "query": "q-indomain",
            "expected_sources": ["doc-a"],
            "required_concepts": [],
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

    calls = {"n": 0}

    def fake_query_kb(query, n_results):
        calls["n"] += 1
        if query == "q-indomain":
            # Relevant doc at rank 2 → recall@1 miss, recall@3 hit, RR=0.5.
            return [
                {"source": "other", "text": "t", "distance": 0.1},
                {"source": "doc-a", "text": "t", "distance": 0.2},
                {"source": "other2", "text": "t", "distance": 0.3},
            ]
        # Out-of-scope: nearest distance above threshold → rejected.
        return [
            {"source": "z", "text": "t", "distance": 0.9},
            {"source": "z2", "text": "t", "distance": 0.95},
            {"source": "z3", "text": "t", "distance": 0.99},
        ]

    monkeypatch.setattr(retrieval_eval, "GOLDEN_SET", tiny_golden)
    monkeypatch.setattr(retrieval_eval, "query_kb", fake_query_kb)

    results = run_retrieval_eval(k_values=[1, 3])

    # Exactly one query_kb call per item — no extra retrieval added.
    assert calls["n"] == len(tiny_golden)

    # New aggregate metrics present and numeric.
    # In-domain query has relevant doc at rank 2: RR=0.5 → MRR=0.5.
    assert results["mrr"] == pytest.approx(0.5, abs=1e-3)
    # nDCG@1: rel=[0] → 0.0 ; nDCG@3: rel=[0,1,0] → 0.631.
    assert results["ndcg@1"] == pytest.approx(0.0, abs=1e-3)
    assert results["ndcg@3"] == pytest.approx(0.631, abs=1e-3)

    # Existing recall numbers preserved: miss@1, hit@3.
    assert results["recall@1"] == pytest.approx(0.0, abs=1e-3)
    assert results["recall@3"] == pytest.approx(1.0, abs=1e-3)

    # Per-query records gain the new fields with correct out-of-scope handling.
    in_domain = results["per_query"][0]
    oos = results["per_query"][1]
    assert in_domain["reciprocal_rank"] == pytest.approx(0.5, abs=1e-3)
    assert in_domain["ndcg_at_k"] == {
        "ndcg@1": pytest.approx(0.0, abs=1e-3),
        "ndcg@3": pytest.approx(0.631, abs=1e-3),
    }
    assert oos["reciprocal_rank"] is None
    assert oos["ndcg_at_k"] == {}
