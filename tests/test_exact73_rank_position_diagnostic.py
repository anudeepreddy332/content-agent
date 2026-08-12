"""Deterministic tests for the exact-73 rank-position complementarity diagnostic."""

from pathlib import Path

import pytest

from experiments.exact73_rank_position_diagnostic import (
    ATTRIBUTION_LABELS,
    AGGREGATE_LABELS,
    REQUIRED_ARTIFACT_SHA256,
    attribution_supports_aggregate_gains,
    classify_aggregate,
    classify_query_difference_at_3,
    filter_preserving_original_ranks,
    load_verified_artifact,
    metric_winner,
    original_rank_map,
    parse_ordered_identities,
    reconstruct_historical_rrf,
    rrf_from_original_ranks,
)


def test_attribution_and_aggregate_labels_are_predeclared():
    assert ATTRIBUTION_LABELS == (
        "PRESERVED_BY_POSITION",
        "ELIMINATED",
        "REVERSED",
        "MEMBERSHIP_REQUIRED",
    )
    assert AGGREGATE_LABELS == (
        "RANK_POSITION_COMPLEMENTARITY_SUPPORTED",
        "RANK_POSITION_COMPLEMENTARITY_PARTIAL",
        "RANK_POSITION_COMPLEMENTARITY_NOT_SUFFICIENT",
    )


def test_parse_ordered_identities_uses_rank_field():
    retrieved = [
        {"rank": 2, "source": "b", "chunk_index": 1},
        {"rank": 1, "source": "a", "chunk_index": 0},
    ]
    assert parse_ordered_identities(retrieved) == [("a", 0), ("b", 1)]


def test_filter_preserving_original_ranks_does_not_compress():
    ranks = {("a", 0): 0, ("b", 1): 3, ("c", 2): 7}
    kept = filter_preserving_original_ranks(ranks, {("b", 1), ("c", 2)})
    assert kept == {("b", 1): 3, ("c", 2): 7}


def test_rrf_uses_original_ranks_and_deterministic_ties():
    texts = {("a", 0): "a", ("z", 0): "z"}
    # Equal dense+bm25 contributions after ranks; tie breaks by source then chunk.
    fused = rrf_from_original_ranks(
        {("z", 0): 0, ("a", 0): 0},
        {("a", 0): 1, ("z", 0): 1},
        depth=2,
        texts=texts,
    )
    assert [(row["source"], row["chunk_index"]) for row in fused] == [("a", 0), ("z", 0)]
    assert fused[0]["rrf_score"] == pytest.approx(1 / 60 + 1 / 61)


def test_metric_winner_uses_recall_then_ndcg():
    assert metric_winner({"source_recall": 1.0, "ndcg": 0.5}, {"source_recall": 0.5, "ndcg": 1.0}) == "left"
    assert metric_winner({"source_recall": 1.0, "ndcg": 0.5}, {"source_recall": 1.0, "ndcg": 0.9}) == "right"
    assert metric_winner({"source_recall": 1.0, "ndcg": 0.5}, {"source_recall": 1.0, "ndcg": 0.5}) == "tie"


def test_classify_query_membership_required_when_unique_dense_expected_in_fused():
    label = classify_query_difference_at_3(
        hist_minilm_at3={"source_recall": 1.0, "ndcg": 1.0},
        hist_gte_at3={"source_recall": 0.0, "ndcg": 0.0},
        replay_minilm_at3={"source_recall": 1.0, "ndcg": 1.0},
        replay_gte_at3={"source_recall": 0.0, "ndcg": 0.0},
        hist_minilm_fused_top3=[("expected", 9)],
        minilm_dense_top10=[("expected", 9), ("shared", 0)],
        shared={("shared", 0)},
        expected_sources={"expected"},
    )
    assert label == "MEMBERSHIP_REQUIRED"


def test_classify_query_preserved_eliminated_reversed():
    shared = {("expected", 0)}
    common = dict(
        hist_minilm_fused_top3=[("expected", 0)],
        minilm_dense_top10=[("expected", 0)],
        shared=shared,
        expected_sources={"expected"},
        hist_minilm_at3={"source_recall": 1.0, "ndcg": 1.0},
        hist_gte_at3={"source_recall": 0.0, "ndcg": 0.0},
    )
    assert (
        classify_query_difference_at_3(
            **common,
            replay_minilm_at3={"source_recall": 1.0, "ndcg": 1.0},
            replay_gte_at3={"source_recall": 0.0, "ndcg": 0.0},
        )
        == "PRESERVED_BY_POSITION"
    )
    assert (
        classify_query_difference_at_3(
            **common,
            replay_minilm_at3={"source_recall": 1.0, "ndcg": 1.0},
            replay_gte_at3={"source_recall": 1.0, "ndcg": 1.0},
        )
        == "ELIMINATED"
    )
    assert (
        classify_query_difference_at_3(
            **common,
            replay_minilm_at3={"source_recall": 0.0, "ndcg": 0.0},
            replay_gte_at3={"source_recall": 1.0, "ndcg": 1.0},
        )
        == "REVERSED"
    )


def test_classify_aggregate_predeclared_branches():
    better = {
        "@3": {"source_recall": 1.0, "ndcg": 1.0},
    }
    worse = {
        "@3": {"source_recall": 0.5, "ndcg": 0.5},
    }
    ndcg_only = {
        "@3": {"source_recall": 0.5, "ndcg": 1.0},
    }
    attribution = [
        {
            "expected_shared_identities": [
                {
                    "minilm_replay_fused_position": 1,
                    "gte_replay_fused_position": 3,
                    "minilm_replay_rrf_score": 0.1,
                    "gte_replay_rrf_score": 0.05,
                }
            ]
        }
    ]
    assert (
        classify_aggregate(better, worse, attribution)
        == "RANK_POSITION_COMPLEMENTARITY_SUPPORTED"
    )
    assert (
        classify_aggregate(ndcg_only, worse, attribution)
        == "RANK_POSITION_COMPLEMENTARITY_PARTIAL"
    )
    assert (
        classify_aggregate(worse, better, attribution)
        == "RANK_POSITION_COMPLEMENTARITY_NOT_SUFFICIENT"
    )


def test_attribution_supports_aggregate_gains_requires_better_placement():
    assert attribution_supports_aggregate_gains(
        [
            {
                "expected_shared_identities": [
                    {
                        "minilm_replay_fused_position": 1,
                        "gte_replay_fused_position": 2,
                        "minilm_replay_rrf_score": 0.2,
                        "gte_replay_rrf_score": 0.1,
                    }
                ]
            }
        ]
    )
    assert not attribution_supports_aggregate_gains(
        [
            {
                "expected_shared_identities": [
                    {
                        "minilm_replay_fused_position": 2,
                        "gte_replay_fused_position": 1,
                        "minilm_replay_rrf_score": 0.1,
                        "gte_replay_rrf_score": 0.2,
                    }
                ]
            }
        ]
    )


def test_load_verified_artifact_matches_hardened_sha():
    root = Path(__file__).resolve().parents[1]
    path = root / "outputs" / "exact73_channel_ablation" / "result.json"
    if not path.exists():
        pytest.skip("stored ablation artifact not present locally")
    artifact = load_verified_artifact(path)
    assert artifact["provenance"]["fingerprint"].startswith("4a1d5d1d")
    assert REQUIRED_ARTIFACT_SHA256


def test_historical_rrf_reconstruction_matches_formula():
    texts = {("a", 0): "a", ("b", 0): "b", ("c", 0): "c"}
    dense = [("a", 0), ("b", 0), ("c", 0)]
    bm25 = [("b", 0), ("a", 0), ("c", 0)]
    fused = reconstruct_historical_rrf(dense, bm25, texts)
    # a: dense0 + bm251 = 1/60 + 1/61
    # b: dense1 + bm250 = 1/61 + 1/60
    # tie on score -> source order a before b
    assert fused[0] in {("a", 0), ("b", 0)}
    assert set(fused[:2]) == {("a", 0), ("b", 0)}
