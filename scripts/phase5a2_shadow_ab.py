"""Run the offline Phase 5A2 Baseline-A versus Candidate-B retrieval experiment.

The script builds both representations from the frozen Markdown corpus, embeds
them with one local MiniLM snapshot, and ranks the same golden-v2 queries with
the same exact cosine, BM25, and RRF implementations. It runs two repetitions,
writes deterministic results separately from timing observations, and never
imports production serving or Qdrant code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.retrieval.chunks import EvalChunk, normalize_interval, source_offsets  # noqa: E402
from agent.retrieval.metrics import span_covered as _span_covered_impl  # noqa: E402
STARTING_HEAD = "4a1567544005eba30c5f679aa66de3dc76b567d3"
BASELINE_MANIFEST = ROOT / "reports/phase5/phase5a0/baseline_a_manifest.json"
CANDIDATE_B_MANIFEST = ROOT / "reports/phase5/phase5a1/candidate_b_manifest.json"
ABC_CONTRACT = ROOT / "evals/fixtures/phase5a0_abc_contract.json"
DEFAULT_OUTPUT = ROOT / "reports/phase5/phase5a2"
K_VALUES = (1, 3, 5, 10)
PRODUCTION_K = 5
SMALL_CHUNK_TOKEN_MAX = 32
NEAR_REDUNDANT_COSINE = 0.95
QUALITY_KEYS = (
    "evidence_span_recall@5",
    "source_recall@5",
    "mrr@10",
    "ndcg@5",
)


class Phase5A2Error(RuntimeError):
    """Fail closed when the frozen experiment contract cannot be reproduced."""


def canonical_json(value: Any) -> str:
    """Serialize deterministic experiment evidence."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def write_json(path: Path, value: Any, *, canonical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(value) if canonical else json.dumps(value, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def install_offline_guard() -> None:
    """Disable tracing/model downloads and reject every socket/DNS operation."""
    for name in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2",
    ):
        os.environ[name] = "false"
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    class NetworkForbidden(BaseException):
        pass

    def audit(event: str, _args: tuple[Any, ...]) -> None:
        if event in {
            "socket.connect",
            "socket.getaddrinfo",
            "socket.sendto",
            "socket.sendmsg",
        }:
            raise NetworkForbidden(f"OFFLINE_NETWORK_FORBIDDEN: {event}")

    sys.addaudithook(audit)


def _source_offsets(raw_text: str) -> tuple[int, int]:
    return source_offsets(raw_text)


def _normalize_interval(
    start: int, end: int, raw_start: int, raw_end: int
) -> tuple[int, int] | None:
    return normalize_interval(start, end, raw_start, raw_end)


def load_representations() -> tuple[
    dict[str, Any],
    list[Any],
    list[EvalChunk],
    list[Any],
    dict[str, str],
    dict[str, Any],
]:
    """Rebuild and validate A and B from the same frozen corpus."""
    from scripts import phase5a0_baseline as baseline
    from scripts import phase5a1_shadow as shadow
    from scripts.retrieval_golden_v2 import load_oracle, validate_oracle
    from transformers import AutoTokenizer

    oracle = load_oracle()
    validate_oracle(oracle)
    source_entries, source_texts, evidence = baseline.load_sources_and_evidence(oracle)
    snapshot = baseline.resolve_local_model_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True
    )
    chunks_a = baseline.reconstruct_legacy_chunks(
        source_entries, source_texts, tokenizer
    )
    if len(chunks_a) != 73:
        raise Phase5A2Error(f"Baseline A chunk count drift: {len(chunks_a)}")
    baseline_manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    observed_a = [
        {
            "ordinal": chunk.ordinal,
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "source_char_start": chunk.source_char_start,
            "source_char_end": chunk.source_char_end,
            "text_sha256": chunk.text_sha256,
        }
        for chunk in chunks_a
    ]
    if observed_a != baseline_manifest["chunking"]["ordered_chunks"]:
        raise Phase5A2Error("Baseline A differs from the frozen 5A0 manifest")
    if (
        sha256_bytes((snapshot / "model.safetensors").read_bytes())
        != baseline_manifest["embedding"]["model_safetensors_sha256"]
    ):
        raise Phase5A2Error("frozen MiniLM model weights changed")

    manifest_b, report_b = shadow.build_corpus(shadow.MiniLMTokenizer())
    checked_b = json.loads(CANDIDATE_B_MANIFEST.read_text(encoding="utf-8"))
    if manifest_b != checked_b:
        raise Phase5A2Error("Candidate B differs from the frozen 5A1 manifest")
    if report_b["child_count"] != 510 or report_b["overflow_count"] != 0:
        raise Phase5A2Error("Candidate B count or overflow invariant drift")

    source_by_path = {
        entry["path"]: entry["source"] for entry in oracle["corpus_manifest"]
    }
    raw_offsets = {
        entry["path"]: _source_offsets(
            (ROOT / entry["path"]).read_text(encoding="utf-8")
        )
        for entry in oracle["corpus_manifest"]
    }
    chunks_b: list[EvalChunk] = []
    for ordinal, child in enumerate(manifest_b["children"]):
        path = child["source_path"]
        raw_start, raw_end = raw_offsets[path]
        intervals = tuple(
            interval
            for span in child["source_spans"]
            if (
                interval := _normalize_interval(
                    span["source_char_start"],
                    span["source_char_end"],
                    raw_start,
                    raw_end,
                )
            )
            is not None
        )
        chunks_b.append(
            EvalChunk(
                ordinal=ordinal,
                chunk_id=child["chunk_id"],
                source=source_by_path[path],
                source_path=path,
                retrieval_text=child["retrieval_text"],
                source_intervals=intervals,
                embedding_content_token_count=child[
                    "embedding_content_token_count"
                ],
            )
        )
    return oracle, chunks_a, chunks_b, evidence, source_texts, {
        "snapshot": snapshot,
        "tokenizer": tokenizer,
        "candidate_b_manifest": manifest_b,
        "candidate_b_report": report_b,
    }


def _a_eval_chunks(chunks: list[Any]) -> list[EvalChunk]:
    return [
        EvalChunk(
            ordinal=chunk.ordinal,
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            source_path=chunk.source_path,
            retrieval_text=chunk.text,
            source_intervals=((chunk.source_char_start, chunk.source_char_end),),
            embedding_content_token_count=chunk.wordpiece_content_tokens,
        )
        for chunk in chunks
    ]


def build_experiment_manifest(
    oracle: dict[str, Any],
    chunks_a: list[Any],
    chunks_b: list[EvalChunk],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Freeze all causal controls and both isolated index identities."""
    from scripts import phase5a0_baseline as baseline
    from scripts import phase5a1_shadow as shadow

    baseline_manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    contract = json.loads(ABC_CONTRACT.read_text(encoding="utf-8"))
    candidate_b_manifest = context["candidate_b_manifest"]
    model = baseline_manifest["embedding"]
    common = {
        "embedding_model_id": model["model_id"],
        "embedding_model_revision": model["local_resolved_revision"],
        "embedding_model_sha256": model["model_safetensors_sha256"],
        "embedding_tokenizer_fingerprint": sha256_json(
            model["tokenizer_file_sha256"]
        ),
        "embedding_dimension": model["dimension"],
        "embedding_normalization": model["normalization"],
        "dense_index_config": {
            "implementation": "exhaustive float32 cosine over normalized vectors",
            "tie_break": "native score descending, chunk_id ascending",
        },
        "bm25_config": {
            "implementation": "rank_bm25.BM25Okapi defaults",
            "tokenization": "text.lower().split()",
            "zero_score_candidates_excluded": True,
            "tie_break": "np.argsort(scores)[::-1] over frozen corpus order",
        },
        "fusion_config": {
            "algorithm": "reciprocal rank fusion over chunk_id",
            "candidate_k": baseline.CANDIDATE_K,
            "rank_origin": 0,
            "rrf_constant": baseline.RRF_CONSTANT,
            "final_k": baseline.FINAL_K,
            "tie_break": "rrf_score descending, chunk_id ascending",
        },
        "query_set_sha256": sha256_json(
            [
                {"query_id": row["query_id"], "query": row["query"]}
                for row in oracle["queries"]
            ]
        ),
        "golden_v2_sha256": sha256_bytes(
            (ROOT / "evals/fixtures/retrieval_golden_v2.json").read_bytes()
        ),
        "metric_implementation": "phase5a0_baseline metric functions reused",
    }

    def index_manifest(
        arm: str,
        representation_fingerprint: str,
        parser_version: str,
        chunker_version: str,
        serialization_version: str,
        vector_count: int,
    ) -> dict[str, Any]:
        identity_payload = {
            "ordered_corpus_manifest_digest": baseline_manifest["corpus"][
                "aggregate_fingerprint"
            ],
            "parser_name": (
                baseline_manifest["parser"]["name"]
                if arm == "A"
                else shadow.PARSER_NAME
            ),
            "parser_version": parser_version,
            "chunker_version": chunker_version,
            "retrieval_serialization_version": serialization_version,
            "embedding_model_id": common["embedding_model_id"],
            "embedding_model_revision": common["embedding_model_revision"],
            "embedding_tokenizer_fingerprint": common[
                "embedding_tokenizer_fingerprint"
            ],
            "dense_index_config": common["dense_index_config"],
            "bm25_config": common["bm25_config"],
            "fusion_config": common["fusion_config"],
        }
        return {
            "arm": arm,
            "index_id": f"ca:index:{sha256_json(identity_payload)}",
            "representation_fingerprint": representation_fingerprint,
            "parser_version": parser_version,
            "chunker_version": chunker_version,
            "retrieval_serialization_version": serialization_version,
            "vector_count": vector_count,
            "document_embedding_count_per_run": vector_count,
            "qdrant_collection": None,
            "index_type": "isolated in-memory exact reference",
            "identity_payload": identity_payload,
        }

    manifest = {
        "schema_version": "phase5a2_shadow_ab_manifest_v1",
        "required_parent": STARTING_HEAD,
        "experiment_question": (
            "Does frozen Candidate B improve retrieval over frozen Baseline A "
            "when document representation is the only intended variable?"
        ),
        "causal_variable": "document representation",
        "held_constant": common,
        "arms": {
            "A": index_manifest(
                "A",
                baseline_manifest["chunking"]["ordered_chunk_fingerprint"],
                baseline.PARSER_VERSION,
                baseline.CHUNKER_VERSION,
                "legacy-raw-chunk-v1",
                len(chunks_a),
            ),
            "B": index_manifest(
                "B",
                shadow.sha(shadow.canonical_json(candidate_b_manifest)),
                shadow.PARSER_VERSION,
                shadow.CHUNKER_VERSION,
                shadow.SERIALIZATION_VERSION,
                len(chunks_b),
            ),
        },
        "embedding_counts_per_repetition": {
            "A_document_embeddings": len(chunks_a),
            "B_document_embeddings": len(chunks_b),
            "shared_query_embeddings": len(oracle["queries"]),
            "total_embedding_outputs": (
                len(chunks_a) + len(chunks_b) + len(oracle["queries"])
            ),
        },
        "candidate_b_representation_invariants": {
            key: context["candidate_b_report"][key]
            for key in (
                "children_above_254",
                "overflow_count",
                "structural_boundary_violations",
                "provenance_failures",
                "duplicate_logical_ids",
            )
        },
        "candidate_b_retrieval_text_grammar": candidate_b_manifest[
            "retrieval_text_grammar"
        ],
        "metric_k_values": list(K_VALUES),
        "per_query_classification": {
            "channel": "hybrid",
            "dimensions": list(QUALITY_KEYS),
            "rule": (
                "REGRESSED if any dimension decreases; otherwise IMPROVED if "
                "any dimension increases; otherwise SAME. Precision is reported "
                "but excluded because redundant same-source children can inflate it."
            ),
        },
        "diagnostic_non_gating_queries": ["Q25", "Q26"],
        "planned_repetitions": 2,
        "determinism_gate": "canonical deterministic results must be byte-identical",
        "absence_metrics": None,
        "absence_metrics_reason": "retrieval golden v2 contains zero ABSENT cases",
        "shadow_only": True,
        "production_retrieval_changed": False,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
        "provider_calls": 0,
        "external_network_calls": 0,
        "contract_index_id_fields": contract["identity_contract"][
            "index_id_payload"
        ],
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def _rank_dense(
    query_embedding: Any, chunk_embeddings: Any, chunks: list[EvalChunk]
) -> list[dict[str, Any]]:
    import numpy as np

    scores = np.asarray(chunk_embeddings @ query_embedding, dtype=np.float64)
    order = sorted(
        range(len(chunks)),
        key=lambda index: (-float(scores[index]), chunks[index].chunk_id),
    )
    return [
        {
            "chunk_id": chunks[index].chunk_id,
            "source": chunks[index].source,
            "chunk_index": chunks[index].ordinal,
            "native_score": round(float(scores[index]), 8),
            "rank": rank,
        }
        for rank, index in enumerate(order, start=1)
    ]


def _rank_bm25(
    query: str, bm25: Any, chunks: list[EvalChunk]
) -> list[dict[str, Any]]:
    import numpy as np

    scores = bm25.get_scores(query.lower().split())
    order = np.argsort(scores)[::-1]
    ranked: list[dict[str, Any]] = []
    for raw_index in order:
        index = int(raw_index)
        score = float(scores[index])
        if score == 0.0:
            continue
        chunk = chunks[index]
        ranked.append(
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "chunk_index": chunk.ordinal,
                "native_score": round(score, 8),
                "rank": len(ranked) + 1,
            }
        )
    return ranked


def _span_covered(
    span: Any,
    rows: list[dict[str, Any]],
    chunks_by_id: dict[str, EvalChunk],
    k: int,
) -> bool:
    return _span_covered_impl(span, rows, chunks_by_id, k)


def _evidence_recall(
    spans: list[Any],
    rows: list[dict[str, Any]],
    chunks_by_id: dict[str, EvalChunk],
    k: int,
) -> float:
    if not spans:
        return 0.0
    return sum(_span_covered(span, rows, chunks_by_id, k) for span in spans) / len(
        spans
    )


def _rank_metrics(
    query: dict[str, Any],
    spans: list[Any],
    rows: list[dict[str, Any]],
    chunks_by_id: dict[str, EvalChunk],
) -> dict[str, float]:
    from scripts import phase5a0_baseline as baseline

    relevance = {
        row["source"]: int(row["grade"]) for row in query["relevant_sources"]
    }
    metrics: dict[str, float] = {}
    for k in K_VALUES:
        recall = baseline.source_recall_at_k(rows, relevance, k)
        metrics[f"recall@{k}"] = round(recall, 8)
        metrics[f"hit@{k}"] = round(baseline.hit_at_k(rows, relevance, k), 8)
        metrics[f"precision@{k}"] = round(
            baseline.precision_at_k(rows, relevance, k), 8
        )
        metrics[f"ndcg@{k}"] = round(
            baseline.graded_ndcg_at_k(rows, relevance, k), 8
        )
        metrics[f"source_recall@{k}"] = round(recall, 8)
        metrics[f"evidence_span_recall@{k}"] = round(
            _evidence_recall(spans, rows, chunks_by_id, k), 8
        )
    metrics["mrr@10"] = round(baseline.reciprocal_rank(rows, relevance, 10), 8)
    return metrics


def _aggregate(
    rows: list[dict[str, Any]], channel: str, *, gating_only: bool
) -> dict[str, Any]:
    eligible = [row for row in rows if row["gating_eligible"] or not gating_only]
    keys = eligible[0]["channels"][channel]["metrics"]
    return {
        "query_count": len(eligible),
        **{
            key: round(
                sum(row["channels"][channel]["metrics"][key] for row in eligible)
                / len(eligible),
                8,
            )
            for key in keys
        },
    }


def _timed(callable_: Any, *args: Any) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    value = callable_(*args)
    return value, (time.perf_counter_ns() - start) / 1_000_000


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.fmean(ordered), 6),
        "p50_ms": round(statistics.median(ordered), 6),
        "p95_ms": round(
            ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
            6,
        ),
        "total_ms": round(sum(ordered), 6),
    }


def _candidate_diagnostics(
    rows: list[dict[str, Any]],
    chunks: list[EvalChunk],
    embeddings: Any,
) -> dict[str, Any]:
    import numpy as np

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    position = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    top = rows[:10]
    sources = [row["source"] for row in top]
    small = [
        row["chunk_id"]
        for row in top
        if by_id[row["chunk_id"]].embedding_content_token_count
        <= SMALL_CHUNK_TOKEN_MAX
    ]
    exact_text_pairs = 0
    near_pairs = 0
    same_source_near_pairs = 0
    for left in range(len(top)):
        for right in range(left + 1, len(top)):
            left_chunk = by_id[top[left]["chunk_id"]]
            right_chunk = by_id[top[right]["chunk_id"]]
            if left_chunk.retrieval_text == right_chunk.retrieval_text:
                exact_text_pairs += 1
            cosine = float(
                np.dot(
                    embeddings[position[left_chunk.chunk_id]],
                    embeddings[position[right_chunk.chunk_id]],
                )
            )
            if cosine >= NEAR_REDUNDANT_COSINE:
                near_pairs += 1
                if left_chunk.source == right_chunk.source:
                    same_source_near_pairs += 1
    return {
        "top10_returned": len(top),
        "unique_sources": len(set(sources)),
        "repeated_source_slots": len(sources) - len(set(sources)),
        "small_chunk_slots": len(small),
        "small_chunk_ids": small,
        "exact_duplicate_text_pairs": exact_text_pairs,
        "near_redundant_pairs_cosine_gte_0_95": near_pairs,
        "same_source_near_redundant_pairs": same_source_near_pairs,
    }


def evaluate_arm(
    arm: str,
    chunks: list[EvalChunk],
    chunk_embeddings: Any,
    query_embeddings: Any,
    oracle: dict[str, Any],
    evidence: list[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate all three frozen retrieval channels for one arm."""
    from rank_bm25 import BM25Okapi
    from scripts.phase5a0_baseline import reciprocal_rank_fusion

    bm25, index_ms = _timed(
        BM25Okapi, [chunk.retrieval_text.lower().split() for chunk in chunks]
    )
    spans_by_query: dict[str, list[Any]] = {}
    for span in evidence:
        spans_by_query.setdefault(span.query_id, []).append(span)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    timings: dict[str, list[float]] = {"dense": [], "bm25": [], "hybrid": []}
    per_query: list[dict[str, Any]] = []
    for query_index, query in enumerate(oracle["queries"]):
        dense, elapsed = _timed(
            _rank_dense, query_embeddings[query_index], chunk_embeddings, chunks
        )
        timings["dense"].append(elapsed)
        bm25_rows, elapsed = _timed(_rank_bm25, query["query"], bm25, chunks)
        timings["bm25"].append(elapsed)
        hybrid, elapsed = _timed(reciprocal_rank_fusion, dense, bm25_rows)
        timings["hybrid"].append(elapsed)
        spans = spans_by_query.get(query["query_id"], [])
        channels = {}
        for name, ranked in (
            ("dense", dense),
            ("bm25", bm25_rows),
            ("hybrid", hybrid),
        ):
            channels[name] = {
                "metrics": _rank_metrics(
                    query, spans, ranked, chunks_by_id
                ),
                "top10": ranked[:10],
                "candidate_top20": (
                    ranked[:20] if name in {"dense", "bm25"} else None
                ),
            }
        channels["hybrid"]["candidate_diagnostics"] = _candidate_diagnostics(
            hybrid, chunks, chunk_embeddings
        )
        per_query.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "gating_eligible": query["gating_eligible"],
                "answerability": query["answerability"],
                "query_bucket": query.get("query_bucket"),
                "failure_tags": query.get("failure_tags", []),
                "channels": channels,
            }
        )
    aggregate = {
        channel: {
            "all_35_diagnostic": _aggregate(
                per_query, channel, gating_only=False
            ),
            "gating_33_excludes_q25_q26": _aggregate(
                per_query, channel, gating_only=True
            ),
        }
        for channel in ("dense", "bm25", "hybrid")
    }
    deterministic = {
        "arm": arm,
        "vector_count": len(chunks),
        "embedding_count": len(chunks),
        "aggregate_metrics": aggregate,
        "per_query": per_query,
    }
    runtime = {
        "arm": arm,
        "bm25_index_build_ms": round(index_ms, 6),
        "retrieval_latency": {
            channel: _latency_summary(values)
            for channel, values in timings.items()
        },
    }
    return deterministic, runtime


def _delta(after: float, before: float) -> float:
    return round(after - before, 8)


def _classification(
    metrics_a: dict[str, float], metrics_b: dict[str, float]
) -> str:
    deltas = [_delta(metrics_b[key], metrics_a[key]) for key in QUALITY_KEYS]
    if any(value < 0 for value in deltas):
        return "REGRESSED"
    if any(value > 0 for value in deltas):
        return "IMPROVED"
    return "SAME"


def compare_results(
    arm_a: dict[str, Any],
    arm_b: dict[str, Any],
    chunks_a: list[EvalChunk],
    chunks_b: list[EvalChunk],
    evidence: list[Any],
    source_texts: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build aggregate, per-query, and supported regression evidence."""
    query_a = {row["query_id"]: row for row in arm_a["per_query"]}
    query_b = {row["query_id"]: row for row in arm_b["per_query"]}
    per_query: list[dict[str, Any]] = []
    for query_id in query_a:
        left, right = query_a[query_id], query_b[query_id]
        channels = {}
        for channel in ("dense", "bm25", "hybrid"):
            metrics_a = left["channels"][channel]["metrics"]
            metrics_b = right["channels"][channel]["metrics"]
            channels[channel] = {
                "A": left["channels"][channel],
                "B": right["channels"][channel],
                "metric_deltas_B_minus_A": {
                    key: _delta(metrics_b[key], metrics_a[key])
                    for key in metrics_a
                },
                "classification": _classification(metrics_a, metrics_b),
            }
        crowd_a = left["channels"]["hybrid"]["candidate_diagnostics"]
        crowd_b = right["channels"]["hybrid"]["candidate_diagnostics"]
        crowding = {
            "A": crowd_a,
            "B": crowd_b,
            "unique_source_delta_B_minus_A": (
                crowd_b["unique_sources"] - crowd_a["unique_sources"]
            ),
            "repeated_source_slot_delta_B_minus_A": (
                crowd_b["repeated_source_slots"]
                - crowd_a["repeated_source_slots"]
            ),
        }
        crowding["small_child_crowding_observed"] = bool(
            crowding["unique_source_delta_B_minus_A"] < 0
            and crowd_b["small_chunk_slots"] > 0
            and crowding["repeated_source_slot_delta_B_minus_A"] > 0
        )
        per_query.append(
            {
                "query_id": query_id,
                "query": left["query"],
                "gating_eligible": left["gating_eligible"],
                "answerability": left["answerability"],
                "query_bucket": left["query_bucket"],
                "failure_tags": left["failure_tags"],
                "classification": channels["hybrid"]["classification"],
                "channels": channels,
                "hybrid_top10_corpus_size_diagnostics": crowding,
            }
        )

    aggregate = {
        "schema_version": "phase5a2_aggregate_comparison_v1",
        "arms": {
            "A": arm_a["aggregate_metrics"],
            "B": arm_b["aggregate_metrics"],
        },
        "deltas_B_minus_A": {
            channel: {
                slice_name: {
                    key: _delta(
                        arm_b["aggregate_metrics"][channel][slice_name][key],
                        arm_a["aggregate_metrics"][channel][slice_name][key],
                    )
                    for key in arm_a["aggregate_metrics"][channel][slice_name]
                    if key != "query_count"
                }
                for slice_name in (
                    "all_35_diagnostic",
                    "gating_33_excludes_q25_q26",
                )
            }
            for channel in ("dense", "bm25", "hybrid")
        },
        "gating_queries": {
            classification.lower(): [
                row["query_id"]
                for row in per_query
                if row["gating_eligible"]
                and row["classification"] == classification
            ]
            for classification in ("IMPROVED", "SAME", "REGRESSED")
        },
        "absence_metrics": None,
        "absence_metrics_reason": "retrieval golden v2 contains zero ABSENT cases",
        "single_composite_score": None,
    }
    required_ids = ("Q07", "Q15", "Q19", "Q21", "Q22", "Q23")
    aggregate["required_query_inspection"] = {
        query_id: {
            "classification": next(
                row["classification"]
                for row in per_query
                if row["query_id"] == query_id
            ),
            "A_hybrid": {
                key: query_a[query_id]["channels"]["hybrid"]["metrics"][key]
                for key in QUALITY_KEYS
            },
            "B_hybrid": {
                key: query_b[query_id]["channels"]["hybrid"]["metrics"][key]
                for key in QUALITY_KEYS
            },
        }
        for query_id in required_ids
    }
    aggregate["gating_improvement_details"] = {
        row["query_id"]: {
            "A_hybrid": {
                key: query_a[row["query_id"]]["channels"]["hybrid"]["metrics"][
                    key
                ]
                for key in QUALITY_KEYS
            },
            "B_hybrid": {
                key: query_b[row["query_id"]]["channels"]["hybrid"]["metrics"][
                    key
                ]
                for key in QUALITY_KEYS
            },
            "deltas_B_minus_A": {
                key: row["channels"]["hybrid"]["metric_deltas_B_minus_A"][key]
                for key in QUALITY_KEYS
            },
        }
        for row in per_query
        if row["gating_eligible"] and row["classification"] == "IMPROVED"
    }
    corpus_effect = _corpus_size_effect(per_query, chunks_a, chunks_b)
    aggregate["corpus_size_effect"] = corpus_effect
    aggregate["promotion_assessment"] = {
        "B_ready_to_advance_as_frozen": False,
        "reasons": [
            "gating hybrid exact evidence-span recall regressed at every reported K",
            "critical per-query evidence regressions are present",
            "510-child top-K crowding materially reduced candidate diversity",
            "index/vector growth is 6.98630137x",
        ],
        "precision_caution": (
            "B precision rises while source diversity and exact-span recall fall; "
            "repeated relevant-source children occupy more result slots."
        ),
        "next_experiment": (
            "Pre-register one Candidate-B structural-packing ablation: combine "
            "adjacent compatible blocks only within the same structural parent "
            "up to the unchanged 254-WordPiece hard limit, assign a new chunker/"
            "serialization identity, and compare A vs frozen B vs packed B with "
            "the same model, queries, K=20, RRF k=60, metrics, and per-query gates."
        ),
    }
    per_query_report = {
        "schema_version": "phase5a2_per_query_comparison_v1",
        "classification_policy": {
            "dimensions": list(QUALITY_KEYS),
            "regression_precedence": True,
        },
        "queries": per_query,
    }
    failures = _failure_analysis(
        per_query,
        chunks_a,
        chunks_b,
        evidence,
        query_a,
        query_b,
        source_texts,
    )
    return aggregate, per_query_report, failures


def _corpus_size_effect(
    per_query: list[dict[str, Any]],
    chunks_a: list[EvalChunk],
    chunks_b: list[EvalChunk],
) -> dict[str, Any]:
    a_vectors, b_vectors = len(chunks_a), len(chunks_b)
    crowding_queries = [
        row["query_id"]
        for row in per_query
        if row["hybrid_top10_corpus_size_diagnostics"][
            "small_child_crowding_observed"
        ]
    ]

    def averages(arm: str) -> dict[str, float]:
        rows = [
            row["hybrid_top10_corpus_size_diagnostics"][arm]
            for row in per_query
        ]
        keys = (
            "unique_sources",
            "repeated_source_slots",
            "small_chunk_slots",
            "exact_duplicate_text_pairs",
            "near_redundant_pairs_cosine_gte_0_95",
            "same_source_near_redundant_pairs",
        )
        return {
            key: round(statistics.fmean(float(row[key]) for row in rows), 8)
            for key in keys
        }

    return {
        "A_vector_count": a_vectors,
        "B_vector_count": b_vectors,
        "vector_count_delta": b_vectors - a_vectors,
        "vector_count_ratio": round(b_vectors / a_vectors, 8),
        "float32_vector_bytes": {
            "A": a_vectors * 384 * 4,
            "B": b_vectors * 384 * 4,
            "delta": (b_vectors - a_vectors) * 384 * 4,
        },
        "embedding_work": {
            "A_document_embeddings_per_run": a_vectors,
            "B_document_embeddings_per_run": b_vectors,
            "delta": b_vectors - a_vectors,
            "ratio": round(b_vectors / a_vectors, 8),
            "A_attempted_content_wordpieces": sum(
                chunk.embedding_content_token_count for chunk in chunks_a
            ),
            "B_attempted_content_wordpieces": sum(
                chunk.embedding_content_token_count for chunk in chunks_b
            ),
            "A_effective_content_wordpieces_after_model_limit": sum(
                min(chunk.embedding_content_token_count, 254)
                for chunk in chunks_a
            ),
            "B_effective_content_wordpieces_after_model_limit": sum(
                min(chunk.embedding_content_token_count, 254)
                for chunk in chunks_b
            ),
        },
        "hybrid_top10_average": {"A": averages("A"), "B": averages("B")},
        "small_child_crowding_query_count": len(crowding_queries),
        "small_child_crowding_queries": crowding_queries,
        "crowding_observed": bool(crowding_queries),
        "crowding_definition": (
            "B has lower top-10 source diversity, more repeated-source slots, "
            "and at least one <=32-token child than A for the same query."
        ),
    }


def _span_structure(
    span: Any, chunks: list[EvalChunk], source_text: str
) -> dict[str, Any]:
    intervals: list[tuple[int, int, str]] = []
    for chunk in chunks:
        if chunk.source != span.source:
            continue
        for start, end in chunk.source_intervals:
            clipped = (max(start, span.char_start), min(end, span.char_end))
            if clipped[0] < clipped[1]:
                intervals.append((clipped[0], clipped[1], chunk.chunk_id))
    covered_until = span.char_start
    children_used: list[str] = []
    while covered_until < span.char_end:
        choices = [
            row for row in intervals if row[0] <= covered_until < row[1]
        ]
        if not choices:
            break
        best = max(choices, key=lambda row: (row[1], row[2]))
        covered_until = best[1]
        if best[2] not in children_used:
            children_used.append(best[2])

    merged: list[list[int]] = []
    for start, end, _chunk_id in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps: list[tuple[int, int]] = []
    cursor = span.char_start
    for start, end in merged:
        if cursor < start:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < span.char_end:
        gaps.append((cursor, span.char_end))
    return {
        "minimum_children_to_exactly_cover": (
            len(children_used) if covered_until >= span.char_end else None
        ),
        "corpus_child_count_intersecting_span": len(
            {chunk_id for _start, _end, chunk_id in intervals}
        ),
        "uncovered_gaps": [
            {
                "char_start": start,
                "char_end": end,
                "whitespace_only": source_text[start:end].isspace(),
            }
            for start, end in gaps
        ],
        "only_uncovered_structural_whitespace": bool(gaps)
        and all(source_text[start:end].isspace() for start, end in gaps),
    }


def _failure_analysis(
    comparisons: list[dict[str, Any]],
    chunks_a: list[EvalChunk],
    chunks_b: list[EvalChunk],
    evidence: list[Any],
    query_a: dict[str, dict[str, Any]],
    query_b: dict[str, dict[str, Any]],
    source_texts: dict[str, str],
) -> dict[str, Any]:
    spans_by_query: dict[str, list[Any]] = {}
    for span in evidence:
        spans_by_query.setdefault(span.query_id, []).append(span)
    by_a = {chunk.chunk_id: chunk for chunk in chunks_a}
    by_b = {chunk.chunk_id: chunk for chunk in chunks_b}
    regressions: list[dict[str, Any]] = []
    for comparison in comparisons:
        if not comparison["gating_eligible"] or comparison["classification"] != "REGRESSED":
            continue
        query_id = comparison["query_id"]
        row_a = query_a[query_id]
        row_b = query_b[query_id]
        hybrid_a = row_a["channels"]["hybrid"]["top10"]
        hybrid_b = row_b["channels"]["hybrid"]["top10"]
        lost_spans = []
        supported: set[str] = set()
        support_rows: list[str] = []
        for span in spans_by_query.get(query_id, []):
            covered_a = _span_covered(span, hybrid_a, by_a, PRODUCTION_K)
            covered_b = _span_covered(span, hybrid_b, by_b, PRODUCTION_K)
            if not covered_a or covered_b:
                continue
            structure = _span_structure(
                span, chunks_b, source_texts[span.source]
            )
            lost_spans.append(
                {
                    "span_id": span.span_id,
                    "source": span.source,
                    "grade": span.grade,
                    "A_hybrid_top5_covered": True,
                    "B_hybrid_top5_covered": False,
                    "B_span_structure": structure,
                }
            )
            pieces = structure["minimum_children_to_exactly_cover"]
            if pieces is not None and pieces > 1:
                supported.add("evidence split across children")
                support_rows.append(
                    f"{span.span_id} requires {pieces} Candidate-B children"
                )
            elif structure["only_uncovered_structural_whitespace"]:
                supported.add("evidence split across children")
                support_rows.append(
                    f"{span.span_id} crosses structural whitespace omitted between B children"
                )
            dense_b = row_b["channels"]["dense"]["candidate_top20"]
            bm25_b = row_b["channels"]["bm25"]["candidate_top20"]
            if (
                _span_covered(span, dense_b, by_b, 20)
                or _span_covered(span, bm25_b, by_b, 20)
            ) and not covered_b:
                supported.add("RRF interaction")
                support_rows.append(
                    f"{span.span_id} is covered by a B candidate top-20 but not hybrid top-5"
                )
        crowding = comparison["hybrid_top10_corpus_size_diagnostics"]
        if crowding["small_child_crowding_observed"]:
            supported.add("candidate-pool crowding")
            support_rows.append(
                "B top-10 loses source diversity while adding small repeated-source children"
            )
        b_diag = crowding["B"]
        a_diag = crowding["A"]
        if (
            b_diag["same_source_near_redundant_pairs"]
            > a_diag["same_source_near_redundant_pairs"]
        ):
            supported.add("duplicate/redundant structural children")
            support_rows.append(
                "B has more same-source cosine>=0.95 pairs in hybrid top-10"
            )
        if not supported:
            supported.add("unknown")
            support_rows.append(
                "representation changed chunk boundaries and serialization together; "
                "this A/B does not isolate a narrower cause"
            )
        regressions.append(
            {
                "query_id": query_id,
                "quality_metric_deltas": {
                    key: comparison["channels"]["hybrid"][
                        "metric_deltas_B_minus_A"
                    ][key]
                    for key in QUALITY_KEYS
                },
                "lost_evidence_spans": lost_spans,
                "likely_causes_supported": sorted(supported),
                "support": sorted(set(support_rows)),
                "not_established_without_separate_ablation": [
                    "chunks became too small",
                    "heading/context serialization altered similarity",
                    "BM25 lexical dilution",
                    "evaluator issue",
                ],
            }
        )
    return {
        "schema_version": "phase5a2_failure_analysis_v1",
        "meaningful_regression_definition": (
            "gating-query hybrid REGRESSED under the conservative per-query rule"
        ),
        "regression_count": len(regressions),
        "regressions": regressions,
        "q22_note": (
            "Q22 uses the corrected golden-v2 span and is not treated as a "
            "known Baseline-A miss."
        ),
        "causal_caution": (
            "Candidate B jointly changes structural boundaries and its frozen "
            "serialization. Narrower causes are reported only when rank/span "
            "evidence supports them; otherwise the cause remains unknown."
        ),
    }


def _encode(encoder: Any, texts: list[str]) -> tuple[Any, float]:
    import numpy as np

    start = time.perf_counter_ns()
    embeddings = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return embeddings, (time.perf_counter_ns() - start) / 1_000_000


def run_once(
    repetition: int,
    encoder: Any,
    oracle: dict[str, Any],
    chunks_a: list[EvalChunk],
    chunks_b: list[EvalChunk],
    evidence: list[Any],
    source_texts: dict[str, str],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one independent embedding/index/ranking repetition."""
    queries = [row["query"] for row in oracle["queries"]]
    embeddings_a, embed_a_ms = _encode(
        encoder, [chunk.retrieval_text for chunk in chunks_a]
    )
    embeddings_b, embed_b_ms = _encode(
        encoder, [chunk.retrieval_text for chunk in chunks_b]
    )
    query_embeddings, embed_query_ms = _encode(encoder, queries)
    arm_a, runtime_a = evaluate_arm(
        "A", chunks_a, embeddings_a, query_embeddings, oracle, evidence
    )
    arm_b, runtime_b = evaluate_arm(
        "B", chunks_b, embeddings_b, query_embeddings, oracle, evidence
    )
    aggregate, per_query, failures = compare_results(
        arm_a, arm_b, chunks_a, chunks_b, evidence, source_texts
    )
    deterministic = {
        "schema_version": "phase5a2_deterministic_result_v1",
        "experiment_manifest_sha256": manifest["manifest_sha256"],
        "arms": {"A": arm_a, "B": arm_b},
        "aggregate_comparison": aggregate,
        "per_query_comparison": per_query,
        "failure_analysis": failures,
        "provider_calls": 0,
        "external_network_calls": 0,
        "production_retrieval_changed": False,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
    }
    deterministic_hash = sha256_json(deterministic)
    runtime = {
        "schema_version": "phase5a2_runtime_observation_v1",
        "repetition": repetition,
        "timing_is_excluded_from_deterministic_hash": True,
        "embedding_ms": {
            "A_documents": round(embed_a_ms, 6),
            "B_documents": round(embed_b_ms, 6),
            "shared_queries": round(embed_query_ms, 6),
        },
        "retrieval": {"A": runtime_a, "B": runtime_b},
        "deterministic_result_sha256": deterministic_hash,
    }
    return deterministic, runtime


def main() -> None:
    install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(DEFAULT_OUTPUT.resolve()):
        raise Phase5A2Error("output must remain under reports/phase5/phase5a2")

    from sentence_transformers import SentenceTransformer
    from scripts import phase5a0_baseline as baseline

    oracle, legacy, chunks_b, evidence, source_texts, context = (
        load_representations()
    )
    chunks_a = _a_eval_chunks(legacy)
    manifest = build_experiment_manifest(
        oracle, legacy, chunks_b, context
    )
    encoder = SentenceTransformer(
        str(context["snapshot"]), local_files_only=True
    )
    if (
        encoder.max_seq_length != baseline.MODEL_MAX_SEQUENCE_LENGTH
        or encoder.get_embedding_dimension() != baseline.EMBEDDING_DIMENSION
    ):
        raise Phase5A2Error("frozen MiniLM runtime identity drift")

    results = []
    runtimes = []
    for repetition in (1, 2):
        deterministic, runtime = run_once(
            repetition,
            encoder,
            oracle,
            chunks_a,
            chunks_b,
            evidence,
            source_texts,
            manifest,
        )
        results.append(deterministic)
        runtimes.append(runtime)
        run_dir = output / f"run-{repetition}"
        stale_full_result = run_dir / "deterministic_result.json"
        if stale_full_result.exists():
            stale_full_result.unlink()
        write_json(
            run_dir / "determinism_report.json",
            {
                "schema_version": "phase5a2_run_determinism_report_v1",
                "repetition": repetition,
                "deterministic_result_sha256": sha256_json(deterministic),
                "component_sha256": {
                    "aggregate_comparison": sha256_json(
                        deterministic["aggregate_comparison"]
                    ),
                    "per_query_comparison": sha256_json(
                        deterministic["per_query_comparison"]
                    ),
                    "failure_analysis": sha256_json(
                        deterministic["failure_analysis"]
                    ),
                },
            },
        )
        write_json(run_dir / "runtime_observation.json", runtime)

    hashes = [sha256_json(result) for result in results]
    if len(set(hashes)) != 1 or results[0] != results[1]:
        raise Phase5A2Error(f"deterministic rerun mismatch: {hashes}")
    write_json(output / "experiment_manifest.json", manifest)
    write_json(
        output / "aggregate_comparison.json",
        results[0]["aggregate_comparison"],
    )
    write_json(
        output / "per_query_comparison.json",
        results[0]["per_query_comparison"],
    )
    write_json(
        output / "failure_analysis.json", results[0]["failure_analysis"]
    )
    runtime_summary = {
        "schema_version": "phase5a2_runtime_summary_v1",
        "repetitions": runtimes,
        "B_over_A_cost_ratios_by_repetition": [
            {
                "repetition": runtime["repetition"],
                "document_embedding_wall_time": round(
                    runtime["embedding_ms"]["B_documents"]
                    / runtime["embedding_ms"]["A_documents"],
                    8,
                ),
                "bm25_index_build_wall_time": round(
                    runtime["retrieval"]["B"]["bm25_index_build_ms"]
                    / runtime["retrieval"]["A"]["bm25_index_build_ms"],
                    8,
                ),
                "retrieval_mean_latency": {
                    channel: round(
                        runtime["retrieval"]["B"]["retrieval_latency"][channel][
                            "mean_ms"
                        ]
                        / runtime["retrieval"]["A"]["retrieval_latency"][channel][
                            "mean_ms"
                        ],
                        8,
                    )
                    for channel in ("dense", "bm25", "hybrid")
                },
            }
            for runtime in runtimes
        ],
        "deterministic_result_sha256": hashes[0],
        "deterministic_rerun_equal": True,
        "latency_scope": (
            "single local machine, two repetitions; descriptive and not a "
            "production capacity benchmark"
        ),
    }
    write_json(output / "runtime_summary.json", runtime_summary)
    print(
        json.dumps(
            {
                "A_vectors": len(chunks_a),
                "B_vectors": len(chunks_b),
                "deterministic_result_sha256": hashes[0],
                "deterministic_rerun_equal": True,
                "gating_queries": results[0]["aggregate_comparison"][
                    "gating_queries"
                ],
                "provider_calls": 0,
                "external_network_calls": 0,
                "production_qdrant_writes": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
