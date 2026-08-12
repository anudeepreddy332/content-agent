"""Deterministic tests for exact-73 retrieval-channel ablation."""

from pathlib import Path

import pytest

from experiments.exact73_channel_ablation import (
    BM25_DEPTH,
    DENSE_DEPTH,
    EXPECTED_FIXTURE_FINGERPRINT,
    FUSED_DEPTH,
    arm_ranking_fingerprint,
    channel_alignment_at_k,
    complementarity_at_k,
    ranking_fingerprint,
    unique_relevant_dense_beyond_bm25,
    _metrics_from_retrieved,
)
from experiments.exact73_jina_compat import (
    FrozenChunk,
    load_exact73_fixture,
    semantic_fingerprint,
    validate_exact73_fixture,
    _rrf,
)


def test_fixture_is_exact_73_chunk_baseline():
    root = Path(__file__).resolve().parents[1]
    chunks = load_exact73_fixture(root / "kb" / "seed_docs")
    assert len(chunks) == 73
    assert semantic_fingerprint(chunks) == EXPECTED_FIXTURE_FINGERPRINT


def test_channel_depth_constants_match_experiment_brief():
    assert DENSE_DEPTH == 10
    assert BM25_DEPTH == 10
    assert FUSED_DEPTH == 5


def test_bm25_expected_displaced_without_dense_contribution():
    bm25 = [FrozenChunk("expected", 0, "text")]
    dense = [{"source": "other_expected", "chunk_index": 0, "text": "x"}]
    fused = [{"source": "noise_from_elsewhere", "chunk_index": 0, "text": "noise"}]
    row = complementarity_at_k(bm25, dense, fused, ["expected"], k=1)
    assert row["bm25_expected_displaced"] is True
    assert row["dense_induced_destructive_fusion"] is False


def test_dense_induced_destructive_fusion_requires_non_expected_dense_in_fused():
    bm25 = [FrozenChunk("expected", 0, "text")]
    dense = [{"source": "noise", "chunk_index": 0, "text": "noise"}]
    fused = [{"source": "noise", "chunk_index": 0, "text": "noise"}]
    row = complementarity_at_k(bm25, dense, fused, ["expected"], k=1)
    assert row["bm25_expected_displaced"] is True
    assert row["dense_induced_destructive_fusion"] is True
    assert row["expected_displaced_from_bm25"] == ["expected"]


def test_channel_alignment_detects_same_source_different_chunk():
    bm25 = [FrozenChunk("expected", 0, "a")]
    dense = [{"source": "expected", "chunk_index": 1, "text": "b"}]
    row = channel_alignment_at_k(bm25, dense, ["expected"], k=1)
    assert row["same_relevant_source_different_chunk"] == ["expected"]
    assert row["same_relevant_chunk_both_channels"] == []
    assert row["expected_source_chunk_reinforcement"] == 0


def test_channel_alignment_detects_chunk_reinforcement():
    bm25 = [FrozenChunk("expected", 2, "a")]
    dense = [{"source": "expected", "chunk_index": 2, "text": "a"}]
    row = channel_alignment_at_k(bm25, dense, ["expected"], k=1)
    assert row["same_relevant_chunk_both_channels"] == [["expected", 2]]
    assert row["expected_source_chunk_reinforcement"] == 1


def test_unique_dense_beyond_bm25_counts_only_expected_sources():
    bm25 = [FrozenChunk("other", 0, "x")]
    dense = [{"source": "expected", "chunk_index": 0, "text": "y"}]
    assert unique_relevant_dense_beyond_bm25(bm25, dense, ["expected"], k=1) == ["expected"]


def test_ranking_fingerprint_is_stable():
    rankings = {"q": [("a", 0), ("b", 1)]}
    assert ranking_fingerprint(rankings) == ranking_fingerprint(rankings)


def test_arm_ranking_fingerprint_covers_retrieved_order():
    arm = {
        "per_query": [
            {
                "query": "q",
                "retrieved": [
                    {"source": "a", "chunk_index": 0},
                    {"source": "b", "chunk_index": 1},
                ],
            }
        ]
    }
    assert arm_ranking_fingerprint(arm) == ranking_fingerprint({"q": [("a", 0), ("b", 1)]})


def test_rrf_fused_depth_is_deterministic():
    dense = [
        {"source": "z", "chunk_index": 0, "text": "z", "distance": 0.1},
        {"source": "a", "chunk_index": 0, "text": "a", "distance": 0.2},
    ]
    bm25 = [FrozenChunk("a", 0, "a"), FrozenChunk("z", 0, "z")]
    fused = _rrf(dense, bm25, depth=2)
    assert [row["source"] for row in fused] == ["a", "z"]


def test_metrics_from_retrieved_hit_at_one():
    retrieved = [{"source": "s", "chunk_index": 0, "text": "concept term"}]
    metrics = _metrics_from_retrieved(retrieved, ["s"], ["concept"])
    assert metrics["@1"]["hit"] == 1.0
    assert metrics["@1"]["concept_pass"] == 1.0


def test_fixture_validation_rejects_drift():
    chunks = [FrozenChunk("only", 0, "x")] * 73
    with pytest.raises(ValueError):
        validate_exact73_fixture(chunks)
