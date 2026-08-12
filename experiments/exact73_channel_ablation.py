"""Exact-73 retrieval-channel ablation: BM25 vs dense vs RRF for MiniLM/GTE/Jina."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from experiments.exact73_jina_compat import (
    BASELINE_SPEC,
    CANDIDATE_SPEC as JINA_SPEC,
    EncoderSpec,
    EVALUATION_DEPTHS,
    EXPECTED_FIXTURE_FINGERPRINT,
    FrozenChunk,
    _dense_rankings,
    _rrf,
    build_bm25_rankings,
    ingest_arm,
    load_encoder as load_jina_or_minilm_encoder,
    load_exact73_fixture,
    prepare_jina_snapshot,
    validate_exact73_fixture,
)
from scripts.retrieval_eval import (
    CONCEPT_PASS_THRESHOLD,
    GOLDEN_SET,
    concept_coverage_at_k,
    hit_at_k,
    ndcg_at_k,
    reciprocal_rank,
    source_diversity_at_k,
    source_recall_at_k,
)

DENSE_DEPTH = 10
BM25_DEPTH = 10
FUSED_DEPTH = 5
RRF_K = 60

MINILM_SPEC = BASELINE_SPEC
GTE_SPEC = EncoderSpec(
    name="gte-modernbert",
    model_id="Alibaba-NLP/gte-modernbert-base",
    revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
    dimension=768,
    max_sequence_length=8192,
)

RECORDED_FUSION_METRICS: dict[str, dict[str, Any]] = {
    "minilm_rrf": {
        "@1": {"hit": 0.9666666666666667},
        "@3": {
            "hit": 1.0,
            "source_recall": 0.95,
            "ndcg": 0.9484196257020611,
            "concept_coverage": 0.8222222222222223,
            "concept_pass": 0.8333333333333334,
        },
        "@5": {
            "hit": 1.0,
            "source_recall": 0.9833333333333333,
            "ndcg": 0.9642328065533475,
            "concept_coverage": 0.8833333333333333,
            "concept_pass": 0.9,
        },
        "mrr": 0.9833333333333333,
    },
    "gte_rrf": {
        "@1": {"hit": 0.9333333333333333},
        "@3": {
            "hit": 0.9666666666666667,
            "source_recall": 0.9333333333333333,
            "ndcg": 0.9258981642750187,
            "concept_coverage": 0.7972222222222223,
            "concept_pass": 0.8,
        },
        "@5": {
            "hit": 1.0,
            "source_recall": 0.9666666666666667,
            "ndcg": 0.9435027057798582,
            "concept_coverage": 0.8333333333333334,
            "concept_pass": 0.8333333333333334,
        },
        "mrr": 0.9583333333333334,
    },
    "jina_rrf": {
        "@1": {"hit": 0.43333333333333335},
        "@3": {
            "hit": 0.9,
            "source_recall": 0.85,
            "ndcg": 0.6767787027411581,
            "concept_coverage": 0.5333333333333333,
            "concept_pass": 0.5,
        },
        "@5": {
            "hit": 0.9333333333333333,
            "source_recall": 0.9166666666666666,
            "ndcg": 0.7087391295151106,
            "concept_coverage": 0.7055555555555556,
            "concept_pass": 0.6333333333333333,
        },
        "mrr": 0.6472222222222223,
    },
}

FUSION_REPRODUCTION_EPSILON = 1e-6


def load_gte_encoder() -> SentenceTransformer:
    """Load pinned GTE with explicit 8192-token contract."""
    encoder = SentenceTransformer(
        GTE_SPEC.model_id,
        revision=GTE_SPEC.revision,
    )
    encoder.max_seq_length = GTE_SPEC.max_sequence_length
    encoder.tokenizer.model_max_length = GTE_SPEC.max_sequence_length
    dimension = encoder.get_embedding_dimension()
    if dimension != GTE_SPEC.dimension:
        raise ValueError(f"GTE: expected {GTE_SPEC.dimension}-D, got {dimension}-D")
    return encoder


def _chunks_to_rows(chunks: Iterable[FrozenChunk]) -> list[dict[str, Any]]:
    return [
        {"source": chunk.source, "chunk_index": chunk.chunk_index, "text": chunk.text}
        for chunk in chunks
    ]


def _metrics_from_retrieved(
    retrieved: list[dict[str, Any]],
    expected_sources: list[str],
    required_concepts: list[str],
) -> dict[str, Any]:
    sources = [row["source"] for row in retrieved]
    expected = set(expected_sources)
    aggregate: dict[int, dict[str, float]] = {}
    for k in EVALUATION_DEPTHS:
        _, coverage = concept_coverage_at_k(retrieved, required_concepts, k)
        unique, duplicates = source_diversity_at_k(sources, k)
        aggregate[k] = {
            "hit": hit_at_k(sources, expected, k),
            "source_recall": source_recall_at_k(sources, expected, k),
            "ndcg": ndcg_at_k(sources, expected, k),
            "concept_coverage": coverage,
            "concept_pass": float(coverage >= CONCEPT_PASS_THRESHOLD),
            "unique_sources": float(unique),
            "duplicate_slots": float(duplicates),
        }
    mrr = reciprocal_rank(sources, expected, max(EVALUATION_DEPTHS))
    metrics = {f"@{k}": aggregate[k] for k in EVALUATION_DEPTHS}
    metrics["mrr"] = mrr
    return metrics


def evaluate_channel(
    in_domain: list[dict[str, Any]],
    retrieve_fn: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate frozen evaluator metrics for one retrieval channel."""
    per_query: list[dict[str, Any]] = []
    rollups: dict[int, dict[str, list[float]]] = {
        k: {
            "hit": [],
            "source_recall": [],
            "ndcg": [],
            "concept_coverage": [],
            "concept_pass": [],
            "unique_sources": [],
            "duplicate_slots": [],
        }
        for k in EVALUATION_DEPTHS
    }
    reciprocal_ranks: list[float] = []
    for item in in_domain:
        retrieved = retrieve_fn(item)
        row_metrics = _metrics_from_retrieved(
            retrieved, item["expected_sources"], item["required_concepts"]
        )
        query_row = {
            "query": item["query"],
            "difficulty": item["difficulty"],
            "expected_sources": item["expected_sources"],
            "retrieved": [
                {
                    "rank": index,
                    "source": r["source"],
                    "chunk_index": r["chunk_index"],
                }
                for index, r in enumerate(retrieved, start=1)
            ],
            "at": {str(k): row_metrics[f"@{k}"] for k in EVALUATION_DEPTHS},
            "mrr_component": row_metrics["mrr"],
        }
        per_query.append(query_row)
        reciprocal_ranks.append(row_metrics["mrr"])
        for k in EVALUATION_DEPTHS:
            for name, value in row_metrics[f"@{k}"].items():
                rollups[k][name].append(value)
    metrics = {
        f"@{k}": {name: float(sum(vals) / len(vals)) for name, vals in rollups[k].items()}
        for k in EVALUATION_DEPTHS
    }
    metrics["mrr"] = float(sum(reciprocal_ranks) / len(reciprocal_ranks))
    return {"metrics": metrics, "per_query": per_query, "query_count": len(in_domain)}


def complementarity_at_k(
    bm25_chunks: list[FrozenChunk],
    dense_rows: list[dict[str, Any]],
    fused_rows: list[dict[str, Any]],
    expected_sources: list[str],
    k: int,
) -> dict[str, Any]:
    """Per-query overlap and displacement diagnostics at one depth."""
    expected = set(expected_sources)
    bm25_sources = [chunk.source for chunk in bm25_chunks[:k]]
    dense_sources = [row["source"] for row in dense_rows[:k]]
    fused_sources = [row["source"] for row in fused_rows[:k]]
    bm25_expected = expected & set(bm25_sources)
    dense_expected = expected & set(dense_sources)
    fused_expected = expected & set(fused_sources[:k])
    unique_bm25 = sorted(bm25_expected - dense_expected)
    unique_dense = sorted(dense_expected - bm25_expected)
    displaced_from_bm25 = sorted(bm25_expected - set(fused_sources[:k]))
    displaced_from_dense = sorted(dense_expected - set(fused_sources[:k]))
    bm25_identities = {chunk.identity for chunk in bm25_chunks[:k]}
    dense_identities = {(row["source"], row["chunk_index"]) for row in dense_rows[:k]}
    destructive_fusion = bool(bm25_expected) and bool(displaced_from_bm25)
    return {
        "k": k,
        "expected_in_bm25": sorted(bm25_expected),
        "expected_in_dense": sorted(dense_expected),
        "expected_unique_bm25": unique_bm25,
        "expected_unique_dense": unique_dense,
        "expected_recovered_fusion": sorted(fused_expected),
        "expected_displaced_from_bm25": displaced_from_bm25,
        "expected_displaced_from_dense": displaced_from_dense,
        "chunk_identity_overlap": len(bm25_identities & dense_identities),
        "source_overlap": len(set(bm25_sources[:k]) & set(dense_sources[:k])),
        "destructive_fusion": destructive_fusion,
    }


def unique_relevant_dense_beyond_bm25(
    bm25_chunks: list[FrozenChunk],
    dense_rows: list[dict[str, Any]],
    expected_sources: list[str],
    k: int,
) -> list[str]:
    """Expected sources dense recovers within top-k that BM25 top-k missed."""
    expected = set(expected_sources)
    bm25_hits = expected & {chunk.source for chunk in bm25_chunks[:k]}
    dense_hits = expected & {row["source"] for row in dense_rows[:k]}
    return sorted(dense_hits - bm25_hits)


def ranking_fingerprint(rankings: dict[str, list[tuple[str, int]]]) -> str:
    """Stable hash of query → (source, chunk_index) ranking lists."""
    payload = json.dumps(rankings, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _fusion_reproduction_failures(computed: dict[str, Any], recorded_key: str) -> list[str]:
    recorded = RECORDED_FUSION_METRICS[recorded_key]
    failures: list[str] = []
    for depth in ("@1", "@3", "@5", "mrr"):
        if depth not in recorded:
            continue
        if depth == "mrr":
            delta = abs(computed["metrics"]["mrr"] - recorded["mrr"])
            if delta > FUSION_REPRODUCTION_EPSILON:
                failures.append(f"{recorded_key} mrr delta {delta}")
            continue
        for metric, expected in recorded[depth].items():
            actual = computed["metrics"][depth][metric]
            if abs(actual - expected) > FUSION_REPRODUCTION_EPSILON:
                failures.append(f"{recorded_key} {metric}{depth}={actual} != {expected}")
    return failures


def minilm_visible_prefix(text: str, encoder: SentenceTransformer, max_tokens: int = 256) -> str:
    """Text MiniLM actually embeds after truncation."""
    tokens = encoder.tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_tokens)
    return encoder.tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)


def early_prefix_bias_evidence(
    minilm_fused: dict[str, Any],
    other_fused: dict[str, Any],
    encoder: SentenceTransformer,
    chunks_by_source: dict[str, list[FrozenChunk]],
) -> dict[str, Any]:
    """Check whether MiniLM-only fusion wins align with early-prefix concept presence."""
    other_by_query = {row["query"]: row for row in other_fused["per_query"]}
    supported: list[str] = []
    unsupported: list[str] = []
    for row in minilm_fused["per_query"]:
        other = other_by_query[row["query"]]
        minilm_pass = row["at"]["3"]["concept_pass"] == 1.0
        other_pass = other["at"]["3"]["concept_pass"] == 1.0
        if not (minilm_pass and not other_pass):
            continue
        concepts = next(
            item["required_concepts"]
            for item in GOLDEN_SET
            if item["query"] == row["query"]
        )
        prefix_texts = []
        for source in row["expected_sources"]:
            for chunk in chunks_by_source.get(source, []):
                prefix_texts.append(minilm_visible_prefix(chunk.text, encoder))
        combined = "\n".join(prefix_texts).lower()
        if all(concept.lower() in combined for concept in concepts):
            supported.append(row["query"])
        else:
            unsupported.append(row["query"])
    if supported and not unsupported:
        verdict = "SUPPORTED"
    elif unsupported:
        verdict = "UNSUPPORTED"
    else:
        verdict = "UNSUPPORTED"
    return {
        "verdict": verdict,
        "minilm_only_concept_pass_at_3": len(supported) + len(unsupported),
        "supported_queries": supported,
        "unsupported_queries": unsupported,
    }


def pairwise_metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Metric differences left minus right at @1/@3/@5 and MRR."""
    deltas: dict[str, Any] = {}
    for depth in ("@1", "@3", "@5"):
        deltas[depth] = {
            metric: left["metrics"][depth][metric] - right["metrics"][depth][metric]
            for metric in left["metrics"][depth]
        }
    deltas["mrr"] = left["metrics"]["mrr"] - right["metrics"]["mrr"]
    return deltas


def run_ablation(project_root: Path) -> dict[str, Any]:
    """Execute all seven arms with shared BM25 and frozen evaluator semantics."""
    chunks = load_exact73_fixture(project_root / "kb" / "seed_docs")
    provenance = validate_exact73_fixture(chunks)
    in_domain = [item for item in GOLDEN_SET if item["expected_sources"]]
    queries = [item["query"] for item in GOLDEN_SET]
    bm25_by_query = build_bm25_rankings(chunks, queries, depth=BM25_DEPTH)
    chunks_by_source: dict[str, list[FrozenChunk]] = {}
    for chunk in chunks:
        chunks_by_source.setdefault(chunk.source, []).append(chunk)

    client = QdrantClient(":memory:")
    arms: dict[str, Any] = {}

    arms["bm25_only"] = evaluate_channel(
        in_domain,
        lambda item: _chunks_to_rows(bm25_by_query[item["query"]][:BM25_DEPTH]),
    )

    minilm_encoder = load_jina_or_minilm_encoder(MINILM_SPEC)
    ingest_arm(client, minilm_encoder, MINILM_SPEC, chunks)

    def minilm_dense(item: dict[str, Any]) -> list[dict[str, Any]]:
        return _dense_rankings(client, minilm_encoder, MINILM_SPEC, item["query"], DENSE_DEPTH)

    arms["minilm_dense_only"] = evaluate_channel(in_domain, minilm_dense)

    def minilm_fused(item: dict[str, Any]) -> list[dict[str, Any]]:
        dense = _dense_rankings(client, minilm_encoder, MINILM_SPEC, item["query"], DENSE_DEPTH)
        bm25 = bm25_by_query[item["query"]]
        return _rrf(dense, bm25, FUSED_DEPTH)

    arms["minilm_rrf"] = evaluate_channel(in_domain, minilm_fused)

    gte_encoder = load_gte_encoder()
    ingest_arm(client, gte_encoder, GTE_SPEC, chunks)

    def gte_dense(item: dict[str, Any]) -> list[dict[str, Any]]:
        return _dense_rankings(client, gte_encoder, GTE_SPEC, item["query"], DENSE_DEPTH)

    arms["gte_dense_only"] = evaluate_channel(in_domain, gte_dense)

    def gte_fused(item: dict[str, Any]) -> list[dict[str, Any]]:
        dense = _dense_rankings(client, gte_encoder, GTE_SPEC, item["query"], DENSE_DEPTH)
        return _rrf(dense, bm25_by_query[item["query"]], FUSED_DEPTH)

    arms["gte_rrf"] = evaluate_channel(in_domain, gte_fused)

    jina_provenance = prepare_jina_snapshot()
    jina_encoder = load_jina_or_minilm_encoder(
        JINA_SPEC,
        candidate_snapshot=Path(jina_provenance["local_package_path"]),
    )
    ingest_arm(client, jina_encoder, JINA_SPEC, chunks)

    def jina_dense(item: dict[str, Any]) -> list[dict[str, Any]]:
        return _dense_rankings(client, jina_encoder, JINA_SPEC, item["query"], DENSE_DEPTH)

    arms["jina_dense_only"] = evaluate_channel(in_domain, jina_dense)

    def jina_fused(item: dict[str, Any]) -> list[dict[str, Any]]:
        dense = _dense_rankings(client, jina_encoder, JINA_SPEC, item["query"], DENSE_DEPTH)
        return _rrf(dense, bm25_by_query[item["query"]], FUSED_DEPTH)

    arms["jina_rrf"] = evaluate_channel(in_domain, jina_fused)

    fusion_failures = (
        _fusion_reproduction_failures(arms["minilm_rrf"], "minilm_rrf")
        + _fusion_reproduction_failures(arms["gte_rrf"], "gte_rrf")
        + _fusion_reproduction_failures(arms["jina_rrf"], "jina_rrf")
    )

    complementarity: dict[str, list[dict[str, Any]]] = {}
    unique_dense_contributions: dict[str, list[dict[str, Any]]] = {}
    for label, spec, encoder in (
        ("minilm", MINILM_SPEC, minilm_encoder),
        ("gte", GTE_SPEC, gte_encoder),
        ("jina", JINA_SPEC, jina_encoder),
    ):
        comp_rows: list[dict[str, Any]] = []
        unique_rows: list[dict[str, Any]] = []
        fused_by_query = {r["query"]: r for r in arms[f"{label}_rrf"]["per_query"]}
        for item in in_domain:
            bm25 = bm25_by_query[item["query"]]
            dense = _dense_rankings(client, encoder, spec, item["query"], DENSE_DEPTH)
            fused = [
                {"source": r["source"], "chunk_index": r["chunk_index"], "text": ""}
                for r in fused_by_query[item["query"]]["retrieved"]
            ]
            comp_entry = {"query": item["query"], "at": {}}
            for k in EVALUATION_DEPTHS:
                comp_entry["at"][str(k)] = complementarity_at_k(
                    bm25, dense, fused, item["expected_sources"], k
                )
            comp_rows.append(comp_entry)
            unique_rows.append(
                {
                    "query": item["query"],
                    "unique_relevant_beyond_bm25_at_5": unique_relevant_dense_beyond_bm25(
                        bm25, dense, item["expected_sources"], FUSED_DEPTH
                    ),
                }
            )
        complementarity[label] = comp_rows
        unique_dense_contributions[label] = unique_rows

    ranking_fps = {
        "bm25": ranking_fingerprint(
            {
                q: [(c.source, c.chunk_index) for c in bm25_by_query[q][:FUSED_DEPTH]]
                for q in queries
            }
        ),
        "minilm_dense": ranking_fingerprint(
            {
                row["query"]: [(r["source"], r["chunk_index"]) for r in row["retrieved"]]
                for row in arms["minilm_dense_only"]["per_query"]
            }
        ),
        "minilm_rrf": ranking_fingerprint(
            {
                row["query"]: [(r["source"], r["chunk_index"]) for r in row["retrieved"]]
                for row in arms["minilm_rrf"]["per_query"]
            }
        ),
    }

    pairwise = {
        "minilm_dense_vs_gte_dense": pairwise_metric_deltas(
            arms["minilm_dense_only"], arms["gte_dense_only"]
        ),
        "minilm_dense_vs_jina_dense": pairwise_metric_deltas(
            arms["minilm_dense_only"], arms["jina_dense_only"]
        ),
        "gte_dense_vs_jina_dense": pairwise_metric_deltas(
            arms["gte_dense_only"], arms["jina_dense_only"]
        ),
        "minilm_rrf_vs_minilm_dense": pairwise_metric_deltas(arms["minilm_rrf"], arms["minilm_dense_only"]),
        "minilm_rrf_vs_bm25": pairwise_metric_deltas(arms["minilm_rrf"], arms["bm25_only"]),
        "gte_rrf_vs_gte_dense": pairwise_metric_deltas(arms["gte_rrf"], arms["gte_dense_only"]),
        "gte_rrf_vs_bm25": pairwise_metric_deltas(arms["gte_rrf"], arms["bm25_only"]),
        "jina_rrf_vs_jina_dense": pairwise_metric_deltas(arms["jina_rrf"], arms["jina_dense_only"]),
        "jina_rrf_vs_bm25": pairwise_metric_deltas(arms["jina_rrf"], arms["bm25_only"]),
    }

    prefix_vs_gte = early_prefix_bias_evidence(
        arms["minilm_rrf"], arms["gte_rrf"], minilm_encoder, chunks_by_source
    )
    prefix_vs_jina = early_prefix_bias_evidence(
        arms["minilm_rrf"], arms["jina_rrf"], minilm_encoder, chunks_by_source
    )

    return {
        "provenance": {
            **provenance,
            "fingerprint": EXPECTED_FIXTURE_FINGERPRINT,
            "in_domain_queries": len(in_domain),
            "dense_depth": DENSE_DEPTH,
            "bm25_depth": BM25_DEPTH,
            "fused_depth": FUSED_DEPTH,
            "rrf_k": RRF_K,
            "jina_local_feasibility": "JINA-LOCAL-FEASIBILITY-FAIL (~6 GB peak RSS > frozen 4 GB gate)",
        },
        "fusion_reproduction_failures": fusion_failures,
        "arms": arms,
        "complementarity": complementarity,
        "unique_dense_beyond_bm25": unique_dense_contributions,
        "pairwise_deltas": pairwise,
        "early_prefix_bias": {"vs_gte": prefix_vs_gte, "vs_jina": prefix_vs_jina},
        "ranking_fingerprints": ranking_fps,
        "jina_model_provenance": jina_provenance,
    }
