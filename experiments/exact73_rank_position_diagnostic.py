"""Rank-position complementarity diagnostic from stored exact-73 rankings.

DIAGNOSTIC ONLY — NOT CANDIDATE ARCHITECTURE.

Isolates dense rank position among MiniLM∩GTE shared identities while holding
BM25, RRF-60, depths, membership, and evaluator fixed. No embedding reloads.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.exact73_channel_ablation import (
    BM25_DEPTH,
    DENSE_DEPTH,
    EXPECTED_FIXTURE_FINGERPRINT,
    FUSED_DEPTH,
    RRF_K,
    _metrics_from_retrieved,
)
from experiments.exact73_jina_compat import (
    FrozenChunk,
    load_exact73_fixture,
)
from scripts.retrieval_eval import GOLDEN_SET

REQUIRED_ARTIFACT_SHA256 = (
    "5c5a9cf1bf6ccafef3f028b01f216e434079a89751eb2cdffa4ef7ece78e4207"
)

REQUIRED_ARMS = (
    "bm25_only",
    "minilm_dense_only",
    "gte_dense_only",
    "minilm_rrf",
    "gte_rrf",
)

# Predeclared per-query attribution labels (defined before result interpretation).
ATTRIBUTION_LABELS = (
    "PRESERVED_BY_POSITION",
    "ELIMINATED",
    "REVERSED",
    "MEMBERSHIP_REQUIRED",
)

# Predeclared aggregate classifications (defined before result interpretation).
AGGREGATE_LABELS = (
    "RANK_POSITION_COMPLEMENTARITY_SUPPORTED",
    "RANK_POSITION_COMPLEMENTARITY_PARTIAL",
    "RANK_POSITION_COMPLEMENTARITY_NOT_SUFFICIENT",
)

Identity = tuple[str, int]


def artifact_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_verified_artifact(path: Path) -> dict[str, Any]:
    """Fail closed unless SHA and provenance match the hardened ablation artifact."""
    digest = artifact_sha256(path)
    if digest != REQUIRED_ARTIFACT_SHA256:
        raise ValueError(
            f"artifact SHA mismatch: expected {REQUIRED_ARTIFACT_SHA256}, got {digest}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance") or {}
    checks = {
        "fingerprint": EXPECTED_FIXTURE_FINGERPRINT,
        "chunk_count": 73,
        "source_count": 20,
        "in_domain_queries": 30,
        "dense_depth": DENSE_DEPTH,
        "bm25_depth": BM25_DEPTH,
        "fused_depth": FUSED_DEPTH,
        "rrf_k": RRF_K,
    }
    for key, expected in checks.items():
        actual = provenance.get(key) if key != "fingerprint" else provenance.get("fingerprint")
        if key == "fingerprint":
            actual = provenance.get("fingerprint")
        if actual != expected:
            raise ValueError(f"provenance {key} mismatch: expected {expected}, got {actual}")
    arms = payload.get("arms") or {}
    missing = [arm for arm in REQUIRED_ARMS if arm not in arms]
    if missing:
        raise ValueError(f"artifact missing required arms: {missing}")
    for arm, depth in (
        ("bm25_only", BM25_DEPTH),
        ("minilm_dense_only", DENSE_DEPTH),
        ("gte_dense_only", DENSE_DEPTH),
        ("minilm_rrf", FUSED_DEPTH),
        ("gte_rrf", FUSED_DEPTH),
    ):
        for row in arms[arm]["per_query"]:
            if len(row["retrieved"]) != depth:
                raise ValueError(
                    f"{arm} query {row['query']!r}: expected {depth} retrieved, "
                    f"got {len(row['retrieved'])}"
                )
    if arms["bm25_only"]["query_count"] != 30:
        raise ValueError("expected 30 in-domain queries in artifact")
    return payload


def parse_ordered_identities(retrieved: list[dict[str, Any]]) -> list[Identity]:
    """Parse stored retrieved rows into ordered (source, chunk_index) identities."""
    ordered = sorted(retrieved, key=lambda row: int(row["rank"]))
    identities = [(row["source"], int(row["chunk_index"])) for row in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError("stored ranking contains duplicate identities")
    return identities


def original_rank_map(identities: list[Identity]) -> dict[Identity, int]:
    """Map identity → original zero-based rank from the ordered list."""
    return {identity: index for index, identity in enumerate(identities)}


def filter_preserving_original_ranks(
    rank_map: dict[Identity, int], keep: set[Identity]
) -> dict[Identity, int]:
    """Keep only shared identities; DO NOT compress/renumber original ranks."""
    return {identity: rank for identity, rank in rank_map.items() if identity in keep}


def rrf_from_original_ranks(
    dense_ranks: dict[Identity, int],
    bm25_ranks: dict[Identity, int],
    *,
    depth: int,
    texts: dict[Identity, str],
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """RRF using original channel ranks; absent channel contributes zero."""
    scores: dict[Identity, float] = {}
    for identity in dense_ranks.keys() | bm25_ranks.keys():
        score = 0.0
        if identity in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[identity])
        if identity in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[identity])
        scores[identity] = score
    fused: list[dict[str, Any]] = []
    for identity, score in sorted(
        scores.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    )[:depth]:
        fused.append(
            {
                "source": identity[0],
                "chunk_index": identity[1],
                "text": texts[identity],
                "rrf_score": score,
            }
        )
    return fused


def reconstruct_historical_rrf(
    dense_identities: list[Identity],
    bm25_identities: list[Identity],
    texts: dict[Identity, str],
) -> list[Identity]:
    """Reconstruct fused top-5 identities from stored dense/BM25 top-10 rankings."""
    fused = rrf_from_original_ranks(
        original_rank_map(dense_identities),
        original_rank_map(bm25_identities),
        depth=FUSED_DEPTH,
        texts=texts,
    )
    return [(row["source"], row["chunk_index"]) for row in fused]


def verify_historical_rrf_reproduction(
    artifact: dict[str, Any], texts: dict[Identity, str]
) -> dict[str, Any]:
    """Require reconstructed MiniLM/GTE RRF top-5 to match stored historical fused rankings."""
    by_query = {
        arm: {row["query"]: row for row in artifact["arms"][arm]["per_query"]}
        for arm in REQUIRED_ARMS
    }
    failures: list[str] = []
    for query, bm25_row in by_query["bm25_only"].items():
        bm25_ids = parse_ordered_identities(bm25_row["retrieved"])
        for arm_dense, arm_rrf in (
            ("minilm_dense_only", "minilm_rrf"),
            ("gte_dense_only", "gte_rrf"),
        ):
            dense_ids = parse_ordered_identities(by_query[arm_dense][query]["retrieved"])
            expected = parse_ordered_identities(by_query[arm_rrf][query]["retrieved"])
            actual = reconstruct_historical_rrf(dense_ids, bm25_ids, texts)
            if actual != expected:
                failures.append(
                    f"{arm_rrf} {query!r}: reconstructed {actual} != stored {expected}"
                )
    return {"ok": not failures, "failures": failures}


def metric_winner(left: dict[str, float], right: dict[str, float]) -> str:
    """Deterministic winner using source_recall then nDCG. Returns left|right|tie."""
    if left["source_recall"] > right["source_recall"]:
        return "left"
    if left["source_recall"] < right["source_recall"]:
        return "right"
    if left["ndcg"] > right["ndcg"]:
        return "left"
    if left["ndcg"] < right["ndcg"]:
        return "right"
    return "tie"


def classify_query_difference_at_3(
    *,
    hist_minilm_at3: dict[str, float],
    hist_gte_at3: dict[str, float],
    replay_minilm_at3: dict[str, float],
    replay_gte_at3: dict[str, float],
    hist_minilm_fused_top3: list[Identity],
    minilm_dense_top10: list[Identity],
    shared: set[Identity],
    expected_sources: set[str],
) -> str:
    """Predeclared per-query label for the historical MiniLM-vs-GTE @3 difference.

    MEMBERSHIP_REQUIRED:
        Historical MiniLM @3 win depends on at least one expected identity that
        appears in MiniLM fused top-3 and MiniLM dense top-10 but not in the
        MiniLM∩GTE shared dense set.
    PRESERVED_BY_POSITION:
        Historical MiniLM @3 win, no unique-membership dependence, and MiniLM-rank
        replay still wins at @3.
    ELIMINATED:
        Historical MiniLM @3 win becomes a tie under replay, OR there was no
        MiniLM historical @3 win to preserve.
    REVERSED:
        Historical MiniLM @3 win becomes a GTE-rank replay win.
    """
    hist = metric_winner(hist_minilm_at3, hist_gte_at3)
    replay = metric_winner(replay_minilm_at3, replay_gte_at3)
    minilm_unique = set(minilm_dense_top10) - shared
    fused_expected = {
        identity
        for identity in hist_minilm_fused_top3[:3]
        if identity[0] in expected_sources
    }
    used_unique_membership = bool(fused_expected & minilm_unique)

    if hist == "left" and used_unique_membership:
        return "MEMBERSHIP_REQUIRED"
    if hist == "left" and replay == "left":
        return "PRESERVED_BY_POSITION"
    if hist == "left" and replay == "tie":
        return "ELIMINATED"
    if hist == "left" and replay == "right":
        return "REVERSED"
    return "ELIMINATED"


def attribution_supports_aggregate_gains(
    per_query: list[dict[str, Any]],
) -> bool:
    """Shared expected identities receive MiniLM-rank-derived better fused placement."""
    improved = 0
    for row in per_query:
        for item in row["expected_shared_identities"]:
            m_pos = item["minilm_replay_fused_position"]
            g_pos = item["gte_replay_fused_position"]
            if m_pos is not None and (g_pos is None or m_pos < g_pos):
                improved += 1
            elif (
                m_pos is not None
                and g_pos is not None
                and m_pos == g_pos
                and item["minilm_replay_rrf_score"] > item["gte_replay_rrf_score"]
            ):
                improved += 1
    return improved > 0


def classify_aggregate(
    minilm_replay_metrics: dict[str, Any],
    gte_replay_metrics: dict[str, Any],
    per_query: list[dict[str, Any]],
) -> str:
    """Predeclared aggregate diagnostic classification."""
    recall_m = minilm_replay_metrics["@3"]["source_recall"]
    recall_g = gte_replay_metrics["@3"]["source_recall"]
    ndcg_m = minilm_replay_metrics["@3"]["ndcg"]
    ndcg_g = gte_replay_metrics["@3"]["ndcg"]
    recall_better = recall_m > recall_g
    ndcg_better = ndcg_m > ndcg_g
    if recall_better and ndcg_better:
        if attribution_supports_aggregate_gains(per_query):
            return "RANK_POSITION_COMPLEMENTARITY_SUPPORTED"
        return "RANK_POSITION_COMPLEMENTARITY_NOT_SUFFICIENT"
    if (not recall_better) and ndcg_better:
        return "RANK_POSITION_COMPLEMENTARITY_PARTIAL"
    return "RANK_POSITION_COMPLEMENTARITY_NOT_SUFFICIENT"


def _fused_position_map(fused: list[dict[str, Any]]) -> dict[Identity, int]:
    return {
        (row["source"], row["chunk_index"]): index
        for index, row in enumerate(fused, start=1)
    }


def _score_map(
    dense_ranks: dict[Identity, int], bm25_ranks: dict[Identity, int]
) -> dict[Identity, float]:
    scores: dict[Identity, float] = {}
    for identity in dense_ranks.keys() | bm25_ranks.keys():
        score = 0.0
        if identity in dense_ranks:
            score += 1.0 / (RRF_K + dense_ranks[identity])
        if identity in bm25_ranks:
            score += 1.0 / (RRF_K + bm25_ranks[identity])
        scores[identity] = score
    return scores


def build_fixture_text_map(project_root: Path) -> dict[Identity, str]:
    chunks = load_exact73_fixture(project_root / "kb" / "seed_docs")
    return {chunk.identity: chunk.text for chunk in chunks}


def evaluate_retrieved_rows(
    retrieved: list[dict[str, Any]],
    expected_sources: list[str],
    required_concepts: list[str],
) -> dict[str, Any]:
    return _metrics_from_retrieved(retrieved, expected_sources, required_concepts)


def run_rank_position_diagnostic(
    project_root: Path,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the stored-ranking rank-position complementarity diagnostic."""
    path = artifact_path or (
        project_root / "outputs" / "exact73_channel_ablation" / "result.json"
    )
    artifact = load_verified_artifact(path)
    texts = build_fixture_text_map(project_root)
    reproduction = verify_historical_rrf_reproduction(artifact, texts)
    if not reproduction["ok"]:
        return {
            "classification": "RANK-POSITION-DIAGNOSTIC-INVALID — RRF REPRODUCTION FAILED",
            "reproduction": reproduction,
            "artifact_sha256": REQUIRED_ARTIFACT_SHA256,
        }

    golden_by_query = {
        item["query"]: item for item in GOLDEN_SET if item["expected_sources"]
    }
    arms = {
        name: {row["query"]: row for row in artifact["arms"][name]["per_query"]}
        for name in REQUIRED_ARMS
    }
    queries = [row["query"] for row in artifact["arms"]["bm25_only"]["per_query"]]

    minilm_replay_rows: list[dict[str, Any]] = []
    gte_replay_rows: list[dict[str, Any]] = []
    attributions: list[dict[str, Any]] = []
    shared_counts: list[int] = []

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
        shared_counts.append(len(shared))

        minilm_shared_ranks = filter_preserving_original_ranks(minilm_ranks, shared)
        gte_shared_ranks = filter_preserving_original_ranks(gte_ranks, shared)
        if set(minilm_shared_ranks) != set(gte_shared_ranks):
            raise ValueError(f"{query!r}: shared membership mismatch after filter")

        minilm_fused = rrf_from_original_ranks(
            minilm_shared_ranks, bm25_ranks, depth=FUSED_DEPTH, texts=texts
        )
        gte_fused = rrf_from_original_ranks(
            gte_shared_ranks, bm25_ranks, depth=FUSED_DEPTH, texts=texts
        )
        minilm_metrics = evaluate_retrieved_rows(
            minilm_fused, golden["expected_sources"], golden["required_concepts"]
        )
        gte_metrics = evaluate_retrieved_rows(
            gte_fused, golden["expected_sources"], golden["required_concepts"]
        )
        minilm_replay_rows.append(
            {
                "query": query,
                "difficulty": golden["difficulty"],
                "expected_sources": golden["expected_sources"],
                "retrieved": [
                    {
                        "rank": index,
                        "source": row["source"],
                        "chunk_index": row["chunk_index"],
                    }
                    for index, row in enumerate(minilm_fused, start=1)
                ],
                "at": {str(k): minilm_metrics[f"@{k}"] for k in (1, 3, 5)},
                "mrr_component": minilm_metrics["mrr"],
            }
        )
        gte_replay_rows.append(
            {
                "query": query,
                "difficulty": golden["difficulty"],
                "expected_sources": golden["expected_sources"],
                "retrieved": [
                    {
                        "rank": index,
                        "source": row["source"],
                        "chunk_index": row["chunk_index"],
                    }
                    for index, row in enumerate(gte_fused, start=1)
                ],
                "at": {str(k): gte_metrics[f"@{k}"] for k in (1, 3, 5)},
                "mrr_component": gte_metrics["mrr"],
            }
        )

        minilm_pos = _fused_position_map(minilm_fused)
        gte_pos = _fused_position_map(gte_fused)
        minilm_scores = _score_map(minilm_shared_ranks, bm25_ranks)
        gte_scores = _score_map(gte_shared_ranks, bm25_ranks)
        expected_shared = sorted(
            identity for identity in shared if identity[0] in expected
        )
        expected_shared_details = []
        for identity in expected_shared:
            expected_shared_details.append(
                {
                    "source": identity[0],
                    "chunk_index": identity[1],
                    "bm25_original_rank": bm25_ranks.get(identity),
                    "minilm_original_dense_rank": minilm_ranks[identity],
                    "gte_original_dense_rank": gte_ranks[identity],
                    "minilm_replay_rrf_score": minilm_scores.get(identity),
                    "gte_replay_rrf_score": gte_scores.get(identity),
                    "minilm_replay_fused_position": minilm_pos.get(identity),
                    "gte_replay_fused_position": gte_pos.get(identity),
                }
            )

        hist_minilm_fused = parse_ordered_identities(arms["minilm_rrf"][query]["retrieved"])
        label = classify_query_difference_at_3(
            hist_minilm_at3=arms["minilm_rrf"][query]["at"]["3"],
            hist_gte_at3=arms["gte_rrf"][query]["at"]["3"],
            replay_minilm_at3=minilm_metrics["@3"],
            replay_gte_at3=gte_metrics["@3"],
            hist_minilm_fused_top3=hist_minilm_fused,
            minilm_dense_top10=minilm_ids,
            shared=shared,
            expected_sources=expected,
        )
        attributions.append(
            {
                "query": query,
                "shared_dense_identity_count": len(shared),
                "expected_shared_identities": expected_shared_details,
                "classification": label,
            }
        )

    def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        rollups = {
            k: {
                "hit": [],
                "source_recall": [],
                "ndcg": [],
                "concept_coverage": [],
                "concept_pass": [],
                "unique_sources": [],
                "duplicate_slots": [],
            }
            for k in (1, 3, 5)
        }
        mrrs: list[float] = []
        for row in rows:
            mrrs.append(row["mrr_component"])
            for k in (1, 3, 5):
                for name, value in row["at"][str(k)].items():
                    rollups[k][name].append(value)
        metrics = {
            f"@{k}": {
                name: float(sum(vals) / len(vals)) for name, vals in rollups[k].items()
            }
            for k in (1, 3, 5)
        }
        metrics["mrr"] = float(sum(mrrs) / len(mrrs))
        return {"metrics": metrics, "per_query": rows, "query_count": len(rows)}

    minilm_arm = _aggregate(minilm_replay_rows)
    gte_arm = _aggregate(gte_replay_rows)
    classification = classify_aggregate(
        minilm_arm["metrics"], gte_arm["metrics"], attributions
    )
    label_counts = {
        label: sum(1 for row in attributions if row["classification"] == label)
        for label in ATTRIBUTION_LABELS
    }
    ranking_fingerprint = sha256(
        json.dumps(
            {
                "minilm": [row["retrieved"] for row in minilm_replay_rows],
                "gte": [row["retrieved"] for row in gte_replay_rows],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    return {
        "artifact_sha256": REQUIRED_ARTIFACT_SHA256,
        "provenance": artifact["provenance"],
        "reproduction": reproduction,
        "common_identity_statistics": {
            "queries": len(shared_counts),
            "mean_shared_dense_identities_top10": float(
                sum(shared_counts) / len(shared_counts)
            ),
            "min_shared": min(shared_counts),
            "max_shared": max(shared_counts),
            "queries_with_zero_shared": sum(1 for count in shared_counts if count == 0),
        },
        "replay_arms": {
            "minilm_rank_replay": minilm_arm,
            "gte_rank_replay": gte_arm,
        },
        "per_query_attribution": attributions,
        "attribution_label_counts": label_counts,
        "ranking_fingerprint": ranking_fingerprint,
        "classification": classification,
        "diagnostic_scope": "DIAGNOSTIC ONLY — NOT CANDIDATE ARCHITECTURE",
    }
