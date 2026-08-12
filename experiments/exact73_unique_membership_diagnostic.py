"""Unique dense-identity membership crossover diagnostic from stored rankings.

DIAGNOSTIC ONLY — NOT CANDIDATE ARCHITECTURE.

Isolates WHICH non-shared dense identities MiniLM vs GTE retrieve, while
marginalizing unique-identity rank assignment via full permutation averaging.
"""

from __future__ import annotations

from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from experiments.exact73_channel_ablation import (
    FUSED_DEPTH,
    RRF_K,
    _metrics_from_retrieved,
)
from experiments.exact73_rank_position_diagnostic import (
    Identity,
    REQUIRED_ARTIFACT_SHA256,
    build_fixture_text_map,
    filter_preserving_original_ranks,
    load_verified_artifact,
    metric_winner,
    original_rank_map,
    parse_ordered_identities,
    rrf_from_original_ranks,
    verify_historical_rrf_reproduction,
)
from scripts.retrieval_eval import GOLDEN_SET

SHARED_ONLY_RESULT_SHA256 = (
    "6d6a73dac1c01a0101d4f85dcce67ff2b9884d3ee4b44f0b12981d59d71f76d4"
)

# Predeclared labels / classifications (frozen before result interpretation).
QUERY_MEMBERSHIP_LABELS = (
    "MINILM_MEMBERSHIP_WIN",
    "GTE_MEMBERSHIP_WIN",
    "TIE",
)
AGGREGATE_LABELS = (
    "UNIQUE_MEMBERSHIP_SUPPORTED",
    "UNIQUE_MEMBERSHIP_PARTIAL",
    "UNIQUE_MEMBERSHIP_NOT_SUFFICIENT",
)

METRIC_KEYS = (
    "hit",
    "source_recall",
    "ndcg",
    "concept_coverage",
    "concept_pass",
    "unique_sources",
    "duplicate_slots",
)


def unique_slots_for_anchor(
    shared_ranks: dict[Identity, int], depth: int = 10
) -> list[int]:
    """Unoccupied zero-based ranks 0..depth-1 after placing shared identities."""
    occupied = set(shared_ranks.values())
    if len(occupied) != len(shared_ranks):
        raise ValueError("shared ranks must be unique within an anchor")
    return [rank for rank in range(depth) if rank not in occupied]


def enumerate_unique_assignments(
    unique_identities: Sequence[Identity], slots: Sequence[int]
) -> list[dict[Identity, int]]:
    """Enumerate every permutation of unique identities into fixed slots.

    Zero-unique edge case: exactly one empty assignment (0! = 1).
    """
    uniques = list(unique_identities)
    slot_list = list(slots)
    if len(uniques) != len(slot_list):
        raise ValueError(
            f"unique count {len(uniques)} != slot count {len(slot_list)}"
        )
    if not uniques:
        return [{}]
    assignments: list[dict[Identity, int]] = []
    for ordering in permutations(uniques):
        assignments.append(
            {identity: slot for identity, slot in zip(ordering, slot_list, strict=True)}
        )
    return assignments


def dense_ranks_from_anchor_and_uniques(
    shared_ranks: dict[Identity, int], unique_assignment: dict[Identity, int]
) -> dict[Identity, int]:
    """Combine immutable shared ranks with one unique-slot assignment."""
    ranks = dict(shared_ranks)
    for identity, rank in unique_assignment.items():
        if identity in ranks:
            raise ValueError("unique identity collides with shared identity")
        if rank in ranks.values():
            raise ValueError("unique slot collides with shared rank")
        ranks[identity] = rank
    if sorted(ranks.values()) != list(range(10)):
        raise ValueError("dense ranks must occupy every slot 0..9 exactly once")
    return ranks


def average_metric_dicts(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Mean nested @k metrics and MRR across rows (equal weight)."""
    rows = list(rows)
    if not rows:
        raise ValueError("cannot average empty metric set")
    out: dict[str, Any] = {}
    for depth in ("@1", "@3", "@5"):
        out[depth] = {
            key: float(sum(row[depth][key] for row in rows) / len(rows))
            for key in METRIC_KEYS
        }
    out["mrr"] = float(sum(row["mrr"] for row in rows) / len(rows))
    return out


def classify_membership_winner(minilm_at3: dict[str, float], gte_at3: dict[str, float]) -> str:
    """recall@3 first, nDCG@3 second."""
    winner = metric_winner(minilm_at3, gte_at3)
    if winner == "left":
        return "MINILM_MEMBERSHIP_WIN"
    if winner == "right":
        return "GTE_MEMBERSHIP_WIN"
    return "TIE"


def classify_aggregate(
    *,
    minilm_anchor_minilm: dict[str, Any],
    minilm_anchor_gte: dict[str, Any],
    gte_anchor_minilm: dict[str, Any],
    gte_anchor_gte: dict[str, Any],
    dual_anchor_minilm_wins: int,
    attribution_coherent: bool,
) -> str:
    """Predeclared UNIQUE_MEMBERSHIP_* gates (frozen before results)."""
    ma_m = minilm_anchor_minilm["@3"]
    ma_g = minilm_anchor_gte["@3"]
    ga_m = gte_anchor_minilm["@3"]
    ga_g = gte_anchor_gte["@3"]

    cross_recall = 0.5 * (
        (ma_m["source_recall"] - ma_g["source_recall"])
        + (ga_m["source_recall"] - ga_g["source_recall"])
    )
    cross_ndcg = 0.5 * (
        (ma_m["ndcg"] - ma_g["ndcg"]) + (ga_m["ndcg"] - ga_g["ndcg"])
    )

    supported = (
        ma_m["source_recall"] > ma_g["source_recall"]
        and ma_m["ndcg"] > ma_g["ndcg"]
        and ga_m["source_recall"] > ga_g["source_recall"]
        and ga_m["ndcg"] > ga_g["ndcg"]
        and dual_anchor_minilm_wins >= 2
        and attribution_coherent
    )
    if supported:
        return "UNIQUE_MEMBERSHIP_SUPPORTED"

    not_sufficient = cross_recall <= 0 and cross_ndcg <= 0
    # Material reversal: one anchor MiniLM-positive, the other GTE-positive on both metrics.
    minilm_anchor_minilm_better = (
        ma_m["source_recall"] > ma_g["source_recall"] and ma_m["ndcg"] > ma_g["ndcg"]
    )
    gte_anchor_gte_better = (
        ga_g["source_recall"] > ga_m["source_recall"] and ga_g["ndcg"] > ga_m["ndcg"]
    )
    gte_anchor_minilm_better = (
        ga_m["source_recall"] > ga_g["source_recall"] and ga_m["ndcg"] > ga_g["ndcg"]
    )
    minilm_anchor_gte_better = (
        ma_g["source_recall"] > ma_m["source_recall"] and ma_g["ndcg"] > ma_m["ndcg"]
    )
    material_reversal = (minilm_anchor_minilm_better and gte_anchor_gte_better) or (
        gte_anchor_minilm_better and minilm_anchor_gte_better
    )
    if not_sufficient or material_reversal:
        return "UNIQUE_MEMBERSHIP_NOT_SUFFICIENT"

    positive_evidence = (
        cross_recall > 0
        or cross_ndcg > 0
        or ma_m["source_recall"] > ma_g["source_recall"]
        or ma_m["ndcg"] > ma_g["ndcg"]
        or ga_m["source_recall"] > ga_g["source_recall"]
        or ga_m["ndcg"] > ga_g["ndcg"]
        or dual_anchor_minilm_wins >= 1
    )
    if positive_evidence:
        return "UNIQUE_MEMBERSHIP_PARTIAL"
    return "UNIQUE_MEMBERSHIP_NOT_SUFFICIENT"


def load_shared_only_reference(
    project_root: Path, texts: dict[Identity, str], artifact: dict[str, Any]
) -> dict[str, Any]:
    """Reuse tracked shared-only result when SHA matches; else recompute from rankings."""
    path = project_root / "outputs" / "exact73_rank_position" / "result.json"
    sha_path = project_root / "outputs" / "exact73_rank_position" / "result.sha256"
    if path.exists() and sha_path.exists():
        digest = sha_path.read_text(encoding="utf-8").strip()
        if digest == SHARED_ONLY_RESULT_SHA256 and sha256(path.read_bytes()).hexdigest() == digest:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {
                "source": "stored_rank_position_diagnostic",
                "sha256": digest,
                "minilm_shared_only": payload["replay_arms"]["minilm_rank_replay"],
                "gte_shared_only": payload["replay_arms"]["gte_rank_replay"],
            }
    # Recompute shared-only from verified stored rankings (no embeddings).
    from experiments.exact73_rank_position_diagnostic import run_rank_position_diagnostic

    recomputed = run_rank_position_diagnostic(project_root)
    return {
        "source": "recomputed_from_stored_rankings",
        "sha256": None,
        "minilm_shared_only": recomputed["replay_arms"]["minilm_rank_replay"],
        "gte_shared_only": recomputed["replay_arms"]["gte_rank_replay"],
    }


def _expected_sources_in_fused(fused: list[dict[str, Any]], expected: set[str], k: int) -> set[str]:
    return {row["source"] for row in fused[:k] if row["source"] in expected}


def evaluate_membership_permutations(
    *,
    shared_ranks: dict[Identity, int],
    unique_identities: list[Identity],
    slots: list[int],
    bm25_ranks: dict[Identity, int],
    texts: dict[Identity, str],
    expected_sources: list[str],
    required_concepts: list[str],
    shared_only_fused_top3_sources: set[str],
) -> dict[str, Any]:
    """Average metrics across all unique-slot permutations for one query/membership."""
    assignments = enumerate_unique_assignments(unique_identities, slots)
    expected = set(expected_sources)
    unique_expected_sources = {identity[0] for identity in unique_identities if identity[0] in expected}
    metric_rows: list[dict[str, Any]] = []
    incremental_hits = 0
    for assignment in assignments:
        dense_ranks = dense_ranks_from_anchor_and_uniques(shared_ranks, assignment)
        fused = rrf_from_original_ranks(
            dense_ranks, bm25_ranks, depth=FUSED_DEPTH, texts=texts
        )
        metrics = _metrics_from_retrieved(fused, expected_sources, required_concepts)
        metric_rows.append(metrics)
        fused_expected = _expected_sources_in_fused(fused, expected, 3)
        if fused_expected - shared_only_fused_top3_sources:
            incremental_hits += 1
    averaged = average_metric_dicts(metric_rows)
    return {
        "permutation_count": len(assignments),
        "metrics": averaged,
        "incremental_expected_source_prob_at_3": float(incremental_hits / len(assignments)),
        "unique_expected_sources": sorted(unique_expected_sources),
        "unique_expected_identities": [
            [source, index]
            for source, index in sorted(unique_identities)
            if source in expected
        ],
    }


def run_unique_membership_diagnostic(
    project_root: Path,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the unique-membership crossover diagnostic from stored rankings only."""
    path = artifact_path or (
        project_root / "outputs" / "exact73_channel_ablation" / "result.json"
    )
    artifact = load_verified_artifact(path)
    texts = build_fixture_text_map(project_root)
    reproduction = verify_historical_rrf_reproduction(artifact, texts)
    if not reproduction["ok"]:
        return {
            "classification": "UNIQUE-MEMBERSHIP-DIAGNOSTIC-INVALID — RRF REPRODUCTION FAILED",
            "reproduction": reproduction,
            "artifact_sha256": REQUIRED_ARTIFACT_SHA256,
            "diagnostic_scope": "DIAGNOSTIC ONLY — NOT CANDIDATE ARCHITECTURE",
        }

    shared_ref = load_shared_only_reference(project_root, texts, artifact)
    shared_minilm_by_q = {
        row["query"]: row for row in shared_ref["minilm_shared_only"]["per_query"]
    }
    shared_gte_by_q = {
        row["query"]: row for row in shared_ref["gte_shared_only"]["per_query"]
    }

    hist_minilm = artifact["arms"]["minilm_rrf"]["metrics"]
    hist_gte = artifact["arms"]["gte_rrf"]["metrics"]
    historical_delta_recall = (
        hist_minilm["@3"]["source_recall"] - hist_gte["@3"]["source_recall"]
    )
    historical_delta_ndcg = hist_minilm["@3"]["ndcg"] - hist_gte["@3"]["ndcg"]

    golden_by_query = {
        item["query"]: item for item in GOLDEN_SET if item["expected_sources"]
    }
    arms = {
        name: {row["query"]: row for row in artifact["arms"][name]["per_query"]}
        for name in (
            "bm25_only",
            "minilm_dense_only",
            "gte_dense_only",
            "minilm_rrf",
            "gte_rrf",
        )
    }
    queries = [row["query"] for row in artifact["arms"]["bm25_only"]["per_query"]]

    per_query: list[dict[str, Any]] = []
    combo_query_metrics: dict[str, list[dict[str, Any]]] = {
        "minilm_anchor_minilm_membership": [],
        "minilm_anchor_gte_membership": [],
        "gte_anchor_minilm_membership": [],
        "gte_anchor_gte_membership": [],
    }

    unique_counts: list[int] = []
    zero_unique_queries = 0

    for query in queries:
        golden = golden_by_query[query]
        expected = set(golden["expected_sources"])
        bm25_ids = parse_ordered_identities(arms["bm25_only"][query]["retrieved"])
        minilm_ids = parse_ordered_identities(arms["minilm_dense_only"][query]["retrieved"])
        gte_ids = parse_ordered_identities(arms["gte_dense_only"][query]["retrieved"])
        bm25_ranks = original_rank_map(bm25_ids)
        minilm_ranks = original_rank_map(minilm_ids)
        gte_ranks = original_rank_map(gte_ids)

        shared = set(minilm_ids) & set(gte_ids)
        minilm_unique = [identity for identity in minilm_ids if identity not in shared]
        gte_unique = [identity for identity in gte_ids if identity not in shared]
        if len(minilm_unique) != len(gte_unique):
            raise ValueError(
                f"{query!r}: unique counts unequal "
                f"{len(minilm_unique)} vs {len(gte_unique)}"
            )
        unique_counts.append(len(minilm_unique))
        if len(minilm_unique) == 0:
            zero_unique_queries += 1

        minilm_shared_ranks = filter_preserving_original_ranks(minilm_ranks, shared)
        gte_shared_ranks = filter_preserving_original_ranks(gte_ranks, shared)
        minilm_slots = unique_slots_for_anchor(minilm_shared_ranks)
        gte_slots = unique_slots_for_anchor(gte_shared_ranks)
        if len(minilm_slots) != len(minilm_unique) or len(gte_slots) != len(gte_unique):
            raise ValueError(f"{query!r}: unique slot count mismatch")

        shared_only_minilm_top3 = {
            row["source"]
            for row in shared_minilm_by_q[query]["retrieved"][:3]
            if row["source"] in expected
        }
        shared_only_gte_top3 = {
            row["source"]
            for row in shared_gte_by_q[query]["retrieved"][:3]
            if row["source"] in expected
        }

        results = {
            "minilm_anchor_minilm_membership": evaluate_membership_permutations(
                shared_ranks=minilm_shared_ranks,
                unique_identities=minilm_unique,
                slots=minilm_slots,
                bm25_ranks=bm25_ranks,
                texts=texts,
                expected_sources=golden["expected_sources"],
                required_concepts=golden["required_concepts"],
                shared_only_fused_top3_sources=shared_only_minilm_top3,
            ),
            "minilm_anchor_gte_membership": evaluate_membership_permutations(
                shared_ranks=minilm_shared_ranks,
                unique_identities=gte_unique,
                slots=minilm_slots,
                bm25_ranks=bm25_ranks,
                texts=texts,
                expected_sources=golden["expected_sources"],
                required_concepts=golden["required_concepts"],
                shared_only_fused_top3_sources=shared_only_minilm_top3,
            ),
            "gte_anchor_minilm_membership": evaluate_membership_permutations(
                shared_ranks=gte_shared_ranks,
                unique_identities=minilm_unique,
                slots=gte_slots,
                bm25_ranks=bm25_ranks,
                texts=texts,
                expected_sources=golden["expected_sources"],
                required_concepts=golden["required_concepts"],
                shared_only_fused_top3_sources=shared_only_gte_top3,
            ),
            "gte_anchor_gte_membership": evaluate_membership_permutations(
                shared_ranks=gte_shared_ranks,
                unique_identities=gte_unique,
                slots=gte_slots,
                bm25_ranks=bm25_ranks,
                texts=texts,
                expected_sources=golden["expected_sources"],
                required_concepts=golden["required_concepts"],
                shared_only_fused_top3_sources=shared_only_gte_top3,
            ),
        }
        for key, value in results.items():
            combo_query_metrics[key].append(value["metrics"])

        minilm_unique_expected_sources = sorted(
            {identity[0] for identity in minilm_unique if identity[0] in expected}
        )
        gte_unique_expected_sources = sorted(
            {identity[0] for identity in gte_unique if identity[0] in expected}
        )
        shared_expected_sources = sorted(
            {identity[0] for identity in shared if identity[0] in expected}
        )
        minilm_only_expected_sources = sorted(
            set(minilm_unique_expected_sources) - set(shared_expected_sources)
        )
        gte_only_expected_sources = sorted(
            set(gte_unique_expected_sources) - set(shared_expected_sources)
        )

        winner_minilm_anchor = classify_membership_winner(
            results["minilm_anchor_minilm_membership"]["metrics"]["@3"],
            results["minilm_anchor_gte_membership"]["metrics"]["@3"],
        )
        winner_gte_anchor = classify_membership_winner(
            results["gte_anchor_minilm_membership"]["metrics"]["@3"],
            results["gte_anchor_gte_membership"]["metrics"]["@3"],
        )

        shared_only_minilm_at3 = shared_minilm_by_q[query]["at"]["3"]
        shared_only_gte_at3 = shared_gte_by_q[query]["at"]["3"]

        per_query.append(
            {
                "query": query,
                "shared_identity_count": len(shared),
                "minilm_unique_count": len(minilm_unique),
                "gte_unique_count": len(gte_unique),
                "permutation_count": results["minilm_anchor_minilm_membership"][
                    "permutation_count"
                ],
                "minilm_unique_expected_identities": [
                    [s, i] for s, i in sorted(minilm_unique) if s in expected
                ],
                "gte_unique_expected_identities": [
                    [s, i] for s, i in sorted(gte_unique) if s in expected
                ],
                "expected_sources_uniquely_introduced_by_minilm_uniques": minilm_only_expected_sources,
                "expected_sources_uniquely_introduced_by_gte_uniques": gte_only_expected_sources,
                "minilm_anchor": {
                    "minilm_membership": {
                        "recall@3": results["minilm_anchor_minilm_membership"]["metrics"]["@3"][
                            "source_recall"
                        ],
                        "ndcg@3": results["minilm_anchor_minilm_membership"]["metrics"]["@3"][
                            "ndcg"
                        ],
                        "concept_coverage@3": results["minilm_anchor_minilm_membership"][
                            "metrics"
                        ]["@3"]["concept_coverage"],
                        "unique_sources@3": results["minilm_anchor_minilm_membership"][
                            "metrics"
                        ]["@3"]["unique_sources"],
                        "duplicate_slots@3": results["minilm_anchor_minilm_membership"][
                            "metrics"
                        ]["@3"]["duplicate_slots"],
                        "incremental_expected_source_prob_at_3": results[
                            "minilm_anchor_minilm_membership"
                        ]["incremental_expected_source_prob_at_3"],
                        "delta_recall_vs_shared_only": results[
                            "minilm_anchor_minilm_membership"
                        ]["metrics"]["@3"]["source_recall"]
                        - shared_only_minilm_at3["source_recall"],
                        "delta_ndcg_vs_shared_only": results[
                            "minilm_anchor_minilm_membership"
                        ]["metrics"]["@3"]["ndcg"]
                        - shared_only_minilm_at3["ndcg"],
                    },
                    "gte_membership": {
                        "recall@3": results["minilm_anchor_gte_membership"]["metrics"]["@3"][
                            "source_recall"
                        ],
                        "ndcg@3": results["minilm_anchor_gte_membership"]["metrics"]["@3"][
                            "ndcg"
                        ],
                        "concept_coverage@3": results["minilm_anchor_gte_membership"][
                            "metrics"
                        ]["@3"]["concept_coverage"],
                        "unique_sources@3": results["minilm_anchor_gte_membership"][
                            "metrics"
                        ]["@3"]["unique_sources"],
                        "duplicate_slots@3": results["minilm_anchor_gte_membership"][
                            "metrics"
                        ]["@3"]["duplicate_slots"],
                        "incremental_expected_source_prob_at_3": results[
                            "minilm_anchor_gte_membership"
                        ]["incremental_expected_source_prob_at_3"],
                        "delta_recall_vs_shared_only": results[
                            "minilm_anchor_gte_membership"
                        ]["metrics"]["@3"]["source_recall"]
                        - shared_only_minilm_at3["source_recall"],
                        "delta_ndcg_vs_shared_only": results[
                            "minilm_anchor_gte_membership"
                        ]["metrics"]["@3"]["ndcg"]
                        - shared_only_minilm_at3["ndcg"],
                    },
                    "winner": winner_minilm_anchor,
                },
                "gte_anchor": {
                    "minilm_membership": {
                        "recall@3": results["gte_anchor_minilm_membership"]["metrics"]["@3"][
                            "source_recall"
                        ],
                        "ndcg@3": results["gte_anchor_minilm_membership"]["metrics"]["@3"][
                            "ndcg"
                        ],
                        "concept_coverage@3": results["gte_anchor_minilm_membership"][
                            "metrics"
                        ]["@3"]["concept_coverage"],
                        "unique_sources@3": results["gte_anchor_minilm_membership"][
                            "metrics"
                        ]["@3"]["unique_sources"],
                        "duplicate_slots@3": results["gte_anchor_minilm_membership"][
                            "metrics"
                        ]["@3"]["duplicate_slots"],
                        "incremental_expected_source_prob_at_3": results[
                            "gte_anchor_minilm_membership"
                        ]["incremental_expected_source_prob_at_3"],
                        "delta_recall_vs_shared_only": results[
                            "gte_anchor_minilm_membership"
                        ]["metrics"]["@3"]["source_recall"]
                        - shared_only_gte_at3["source_recall"],
                        "delta_ndcg_vs_shared_only": results[
                            "gte_anchor_minilm_membership"
                        ]["metrics"]["@3"]["ndcg"]
                        - shared_only_gte_at3["ndcg"],
                    },
                    "gte_membership": {
                        "recall@3": results["gte_anchor_gte_membership"]["metrics"]["@3"][
                            "source_recall"
                        ],
                        "ndcg@3": results["gte_anchor_gte_membership"]["metrics"]["@3"][
                            "ndcg"
                        ],
                        "concept_coverage@3": results["gte_anchor_gte_membership"][
                            "metrics"
                        ]["@3"]["concept_coverage"],
                        "unique_sources@3": results["gte_anchor_gte_membership"][
                            "metrics"
                        ]["@3"]["unique_sources"],
                        "duplicate_slots@3": results["gte_anchor_gte_membership"][
                            "metrics"
                        ]["@3"]["duplicate_slots"],
                        "incremental_expected_source_prob_at_3": results[
                            "gte_anchor_gte_membership"
                        ]["incremental_expected_source_prob_at_3"],
                        "delta_recall_vs_shared_only": results[
                            "gte_anchor_gte_membership"
                        ]["metrics"]["@3"]["source_recall"]
                        - shared_only_gte_at3["source_recall"],
                        "delta_ndcg_vs_shared_only": results[
                            "gte_anchor_gte_membership"
                        ]["metrics"]["@3"]["ndcg"]
                        - shared_only_gte_at3["ndcg"],
                    },
                    "winner": winner_gte_anchor,
                },
                "winner_stable_across_anchors": winner_minilm_anchor == winner_gte_anchor,
            }
        )

    aggregates = {
        key: average_metric_dicts(rows) for key, rows in combo_query_metrics.items()
    }
    dual_anchor_minilm_wins = sum(
        1
        for row in per_query
        if row["minilm_anchor"]["winner"] == "MINILM_MEMBERSHIP_WIN"
        and row["gte_anchor"]["winner"] == "MINILM_MEMBERSHIP_WIN"
    )
    attribution_coherent = dual_anchor_minilm_wins >= 1 and any(
        row["expected_sources_uniquely_introduced_by_minilm_uniques"]
        for row in per_query
        if row["minilm_anchor"]["winner"] == "MINILM_MEMBERSHIP_WIN"
        and row["gte_anchor"]["winner"] == "MINILM_MEMBERSHIP_WIN"
    )

    classification = classify_aggregate(
        minilm_anchor_minilm=aggregates["minilm_anchor_minilm_membership"],
        minilm_anchor_gte=aggregates["minilm_anchor_gte_membership"],
        gte_anchor_minilm=aggregates["gte_anchor_minilm_membership"],
        gte_anchor_gte=aggregates["gte_anchor_gte_membership"],
        dual_anchor_minilm_wins=dual_anchor_minilm_wins,
        attribution_coherent=attribution_coherent,
    )

    ma_delta_recall = (
        aggregates["minilm_anchor_minilm_membership"]["@3"]["source_recall"]
        - aggregates["minilm_anchor_gte_membership"]["@3"]["source_recall"]
    )
    ma_delta_ndcg = (
        aggregates["minilm_anchor_minilm_membership"]["@3"]["ndcg"]
        - aggregates["minilm_anchor_gte_membership"]["@3"]["ndcg"]
    )
    ga_delta_recall = (
        aggregates["gte_anchor_minilm_membership"]["@3"]["source_recall"]
        - aggregates["gte_anchor_gte_membership"]["@3"]["source_recall"]
    )
    ga_delta_ndcg = (
        aggregates["gte_anchor_minilm_membership"]["@3"]["ndcg"]
        - aggregates["gte_anchor_gte_membership"]["@3"]["ndcg"]
    )
    cross_recall = 0.5 * (ma_delta_recall + ga_delta_recall)
    cross_ndcg = 0.5 * (ma_delta_ndcg + ga_delta_ndcg)

    def _explained(delta: float, historical: float) -> float | None:
        if historical == 0:
            return None
        return float(delta / historical)

    # Shared-only comparisons (aggregate).
    shared_minilm_metrics = shared_ref["minilm_shared_only"]["metrics"]
    shared_gte_metrics = shared_ref["gte_shared_only"]["metrics"]
    harm_less_useful = {
        "minilm_anchor": {
            "minilm_membership_delta_recall_vs_shared_only": aggregates[
                "minilm_anchor_minilm_membership"
            ]["@3"]["source_recall"]
            - shared_minilm_metrics["@3"]["source_recall"],
            "minilm_membership_delta_ndcg_vs_shared_only": aggregates[
                "minilm_anchor_minilm_membership"
            ]["@3"]["ndcg"]
            - shared_minilm_metrics["@3"]["ndcg"],
            "gte_membership_delta_recall_vs_shared_only": aggregates[
                "minilm_anchor_gte_membership"
            ]["@3"]["source_recall"]
            - shared_minilm_metrics["@3"]["source_recall"],
            "gte_membership_delta_ndcg_vs_shared_only": aggregates[
                "minilm_anchor_gte_membership"
            ]["@3"]["ndcg"]
            - shared_minilm_metrics["@3"]["ndcg"],
        },
        "gte_anchor": {
            "minilm_membership_delta_recall_vs_shared_only": aggregates[
                "gte_anchor_minilm_membership"
            ]["@3"]["source_recall"]
            - shared_gte_metrics["@3"]["source_recall"],
            "minilm_membership_delta_ndcg_vs_shared_only": aggregates[
                "gte_anchor_minilm_membership"
            ]["@3"]["ndcg"]
            - shared_gte_metrics["@3"]["ndcg"],
            "gte_membership_delta_recall_vs_shared_only": aggregates[
                "gte_anchor_gte_membership"
            ]["@3"]["source_recall"]
            - shared_gte_metrics["@3"]["source_recall"],
            "gte_membership_delta_ndcg_vs_shared_only": aggregates[
                "gte_anchor_gte_membership"
            ]["@3"]["ndcg"]
            - shared_gte_metrics["@3"]["ndcg"],
        },
    }

    label_counts = {
        "minilm_anchor": {
            label: sum(1 for row in per_query if row["minilm_anchor"]["winner"] == label)
            for label in QUERY_MEMBERSHIP_LABELS
        },
        "gte_anchor": {
            label: sum(1 for row in per_query if row["gte_anchor"]["winner"] == label)
            for label in QUERY_MEMBERSHIP_LABELS
        },
        "dual_anchor_minilm_wins": dual_anchor_minilm_wins,
        "stable_across_anchors": sum(
            1 for row in per_query if row["winner_stable_across_anchors"]
        ),
    }

    summary_fingerprint = sha256(
        json.dumps(
            {
                "aggregates": aggregates,
                "classification": classification,
                "per_query_winners": [
                    [
                        row["query"],
                        row["minilm_anchor"]["winner"],
                        row["gte_anchor"]["winner"],
                    ]
                    for row in per_query
                ],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    return {
        "artifact_sha256": REQUIRED_ARTIFACT_SHA256,
        "provenance": artifact["provenance"],
        "reproduction": reproduction,
        "shared_only_reference": {
            "source": shared_ref["source"],
            "sha256": shared_ref["sha256"],
            "minilm_shared_only_metrics": shared_minilm_metrics,
            "gte_shared_only_metrics": shared_gte_metrics,
        },
        "unique_set_statistics": {
            "queries": len(queries),
            "mean_unique_count": float(sum(unique_counts) / len(unique_counts)),
            "min_unique_count": min(unique_counts),
            "max_unique_count": max(unique_counts),
            "zero_unique_queries": zero_unique_queries,
        },
        "crossover_aggregates": aggregates,
        "membership_deltas": {
            "minilm_anchor": {
                "recall@3": ma_delta_recall,
                "ndcg@3": ma_delta_ndcg,
            },
            "gte_anchor": {
                "recall@3": ga_delta_recall,
                "ndcg@3": ga_delta_ndcg,
            },
            "cross_anchor_average": {
                "recall@3": cross_recall,
                "ndcg@3": cross_ndcg,
            },
        },
        "effect_size_vs_historical": {
            "historical_delta_recall@3": historical_delta_recall,
            "historical_delta_ndcg@3": historical_delta_ndcg,
            "membership_explained_fraction_recall": _explained(
                cross_recall, historical_delta_recall
            ),
            "membership_explained_fraction_ndcg": _explained(
                cross_ndcg, historical_delta_ndcg
            ),
        },
        "shared_only_comparisons": harm_less_useful,
        "per_query_attribution": per_query,
        "attribution_summary": label_counts,
        "summary_fingerprint": summary_fingerprint,
        "classification": classification,
        "diagnostic_scope": "DIAGNOSTIC ONLY — NOT CANDIDATE ARCHITECTURE",
        "cutover": "DIAGNOSTIC ONLY — NO ARCHITECTURE CUTOVER AUTHORIZED",
    }
