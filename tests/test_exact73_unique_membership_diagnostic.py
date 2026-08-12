"""Deterministic tests for the unique-membership crossover diagnostic."""

from pathlib import Path

import pytest

from experiments.exact73_unique_membership_diagnostic import (
    AGGREGATE_LABELS,
    QUERY_MEMBERSHIP_LABELS,
    classify_aggregate,
    classify_membership_winner,
    dense_ranks_from_anchor_and_uniques,
    enumerate_unique_assignments,
    unique_slots_for_anchor,
)
from experiments.exact73_rank_position_diagnostic import (
    REQUIRED_ARTIFACT_SHA256,
    load_verified_artifact,
    verify_historical_rrf_reproduction,
    build_fixture_text_map,
)


def test_predeclared_labels_frozen():
    assert QUERY_MEMBERSHIP_LABELS == (
        "MINILM_MEMBERSHIP_WIN",
        "GTE_MEMBERSHIP_WIN",
        "TIE",
    )
    assert AGGREGATE_LABELS == (
        "UNIQUE_MEMBERSHIP_SUPPORTED",
        "UNIQUE_MEMBERSHIP_PARTIAL",
        "UNIQUE_MEMBERSHIP_NOT_SUFFICIENT",
    )


def test_unique_slots_preserve_shared_ranks():
    shared = {("a", 0): 0, ("b", 1): 4, ("c", 2): 7}
    assert unique_slots_for_anchor(shared) == [1, 2, 3, 5, 6, 8, 9]


def test_zero_unique_enumerates_exactly_one_empty_permutation():
    assignments = enumerate_unique_assignments([], [])
    assert assignments == [{}]


def test_permutation_enumeration_is_complete_and_deterministic():
    uniques = [("u", 0), ("v", 1)]
    slots = [2, 5]
    first = enumerate_unique_assignments(uniques, slots)
    second = enumerate_unique_assignments(uniques, slots)
    assert first == second
    assert len(first) == 2
    assert {tuple(sorted(item.items())) for item in first} == {
        ((("u", 0), 2), (("v", 1), 5)),
        ((("u", 0), 5), (("v", 1), 2)),
    }


def test_dense_ranks_remain_zero_based_and_uncompressed():
    shared = {("s", 0): 0, ("s", 1): 3}
    assignment = {("u", 0): 1, ("u", 1): 2, ("u", 2): 4, ("u", 3): 5, ("u", 4): 6, ("u", 5): 7, ("u", 6): 8, ("u", 7): 9}
    ranks = dense_ranks_from_anchor_and_uniques(shared, assignment)
    assert ranks[("s", 0)] == 0
    assert ranks[("s", 1)] == 3
    assert sorted(ranks.values()) == list(range(10))


def test_both_memberships_would_use_identical_slots():
    shared = {("a", 0): 1, ("b", 0): 8}
    slots = unique_slots_for_anchor(shared)
    minilm_unique = [("m", i) for i in range(8)]
    gte_unique = [("g", i) for i in range(8)]
    m_assigns = enumerate_unique_assignments(minilm_unique, slots)
    g_assigns = enumerate_unique_assignments(gte_unique, slots)
    assert {tuple(sorted(a.values())) for a in m_assigns} == {
        tuple(sorted(a.values())) for a in g_assigns
    }
    assert set(slots) == set(m_assigns[0].values())


def test_classify_membership_winner_recall_then_ndcg():
    assert (
        classify_membership_winner(
            {"source_recall": 1.0, "ndcg": 0.1},
            {"source_recall": 0.5, "ndcg": 1.0},
        )
        == "MINILM_MEMBERSHIP_WIN"
    )
    assert (
        classify_membership_winner(
            {"source_recall": 0.5, "ndcg": 1.0},
            {"source_recall": 0.5, "ndcg": 0.2},
        )
        == "MINILM_MEMBERSHIP_WIN"
    )
    assert (
        classify_membership_winner(
            {"source_recall": 0.5, "ndcg": 0.5},
            {"source_recall": 0.5, "ndcg": 0.5},
        )
        == "TIE"
    )


def test_classify_aggregate_gates_predeclared():
    better = {"@3": {"source_recall": 1.0, "ndcg": 1.0}}
    worse = {"@3": {"source_recall": 0.5, "ndcg": 0.5}}
    assert (
        classify_aggregate(
            minilm_anchor_minilm=better,
            minilm_anchor_gte=worse,
            gte_anchor_minilm=better,
            gte_anchor_gte=worse,
            dual_anchor_minilm_wins=2,
            attribution_coherent=True,
        )
        == "UNIQUE_MEMBERSHIP_SUPPORTED"
    )
    assert (
        classify_aggregate(
            minilm_anchor_minilm={"@3": {"source_recall": 0.6, "ndcg": 0.4}},
            minilm_anchor_gte={"@3": {"source_recall": 0.5, "ndcg": 0.5}},
            gte_anchor_minilm={"@3": {"source_recall": 0.6, "ndcg": 0.4}},
            gte_anchor_gte={"@3": {"source_recall": 0.5, "ndcg": 0.5}},
            dual_anchor_minilm_wins=1,
            attribution_coherent=False,
        )
        == "UNIQUE_MEMBERSHIP_PARTIAL"
    )
    assert (
        classify_aggregate(
            minilm_anchor_minilm=worse,
            minilm_anchor_gte=better,
            gte_anchor_minilm=worse,
            gte_anchor_gte=better,
            dual_anchor_minilm_wins=0,
            attribution_coherent=False,
        )
        == "UNIQUE_MEMBERSHIP_NOT_SUFFICIENT"
    )


def test_provenance_and_rrf_reproduction_on_stored_artifact():
    root = Path(__file__).resolve().parents[1]
    path = root / "outputs" / "exact73_channel_ablation" / "result.json"
    if not path.exists():
        pytest.skip("stored ablation artifact absent")
    artifact = load_verified_artifact(path)
    texts = build_fixture_text_map(root)
    reproduction = verify_historical_rrf_reproduction(artifact, texts)
    assert reproduction["ok"] is True
    assert REQUIRED_ARTIFACT_SHA256


def test_zero_unique_query_averaging_has_no_divide_by_zero():
    from experiments.exact73_unique_membership_diagnostic import (
        average_metric_dicts,
        evaluate_membership_permutations,
    )

    texts = {("s", 0): "alpha beta", ("s", 1): "gamma"}
    # Fill ranks 0..9 with shared-only identities via fake shared map of 10.
    shared = {(f"s{i}", 0): i for i in range(10)}
    # Override texts for those identities.
    texts = {identity: f"concept {identity[0]}" for identity in shared}
    shared_only_sources = {"s0"}
    result = evaluate_membership_permutations(
        shared_ranks=shared,
        unique_identities=[],
        slots=[],
        bm25_ranks={("s0", 0): 0},
        texts=texts,
        expected_sources=["s0"],
        required_concepts=["concept"],
        shared_only_fused_top3_sources=shared_only_sources,
    )
    assert result["permutation_count"] == 1
    assert "source_recall" in result["metrics"]["@3"]
    # Equal-weight average of a single query metric row works.
    averaged = average_metric_dicts([result["metrics"], result["metrics"]])
    assert averaged["@3"]["source_recall"] == result["metrics"]["@3"]["source_recall"]
