"""Reproduce and measure the frozen Phase-5 Baseline A without Qdrant writes.

The evaluator deliberately reconstructs the legacy corpus from the checked-in
Markdown sources.  Dense retrieval uses the exact, local MiniLM snapshot and an
exhaustive cosine comparison; BM25 uses the production lowercase/whitespace
tokenization; fusion uses RRF over deterministic chunk IDs.

No network service, provider, or operational vector collection is read or
written by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.retrieval_golden_v2 import canonical_json_dumps, load_oracle, validate_oracle


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "phase5" / "phase5a0"
STARTING_HEAD = "f269ab760fc78f0b3a65618ae0c744649d1a0a2e"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_CACHE_DIRNAME = "models--sentence-transformers--all-MiniLM-L6-v2"
PARSER_VERSION = "legacy-direct-text-strip-v1"
CHUNKER_VERSION = "legacy-cl100k-400-50-v1"
TOKENIZER_NAME = "cl100k_base"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
CHUNK_STRIDE = CHUNK_SIZE - CHUNK_OVERLAP
MODEL_MAX_SEQUENCE_LENGTH = 256
MODEL_SPECIAL_TOKENS = 2
CONTENT_TOKEN_LIMIT = MODEL_MAX_SEQUENCE_LENGTH - MODEL_SPECIAL_TOKENS
EMBEDDING_DIMENSION = 384
RRF_CONSTANT = 60
FINAL_K = 10
CANDIDATE_K = 20
K_VALUES = (1, 3, 5, 10)
PRODUCTION_RETRIEVAL_K = 5
DRAFTER_K = 3
DRAFTER_CHAR_LIMIT = 2000
VERIFIER_K = 5


class Phase5A0Error(RuntimeError):
    """Baseline reconstruction or measurement violated a frozen invariant."""


@dataclass(frozen=True)
class EvidenceSpan:
    query_id: str
    source: str
    grade: int
    span_id: str
    char_start: int
    char_end: int
    line_start: int
    line_end: int


@dataclass(frozen=True)
class LegacyChunk:
    ordinal: int
    chunk_id: str
    source: str
    source_path: str
    source_sha256: str
    chunk_index: int
    text: str
    text_sha256: str
    source_char_start: int
    source_char_end: int
    cl100k_tokens: int
    wordpiece_content_tokens: int
    tokens_beyond_limit: int
    truncated: bool
    truncated_fraction: float
    truncated_tail_char_start: int | None
    boundary_decode_exact: bool


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def deterministic_id(kind: str, payload: dict[str, Any]) -> str:
    digest = sha256_text(canonical_json_dumps(payload))
    return f"ca:{kind}:{digest}"


def current_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def require_starting_head() -> None:
    observed = current_head()
    if observed != STARTING_HEAD:
        raise Phase5A0Error(
            f"starting HEAD mismatch: observed={observed} required={STARTING_HEAD}"
        )


def resolve_local_model_snapshot(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / MODEL_CACHE_DIRNAME
            / "snapshots"
            / MODEL_REVISION,
        ]
    )
    for candidate in candidates:
        if (candidate / "model.safetensors").is_file() and (
            candidate / "tokenizer.json"
        ).is_file():
            return candidate.resolve()
    raise Phase5A0Error(
        "frozen MiniLM snapshot is unavailable locally; refusing network resolution"
    )


def _line_char_span(text: str, line_start: int, line_end: int) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    if line_start < 1 or line_end < line_start or line_end > len(lines):
        raise Phase5A0Error(f"invalid line span {line_start}-{line_end}")
    start = sum(len(line) for line in lines[: line_start - 1])
    end = sum(len(line) for line in lines[:line_end])
    while end > start and text[end - 1] in "\r\n":
        end -= 1
    return start, end


def load_sources_and_evidence(
    oracle: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str], list[EvidenceSpan]]:
    source_entries: list[dict[str, Any]] = []
    source_texts: dict[str, str] = {}
    for entry in oracle["corpus_manifest"]:
        path = REPO_ROOT / entry["path"]
        raw = path.read_text(encoding="utf-8")
        observed_sha = sha256_text(raw)
        if observed_sha != entry["sha256"]:
            raise Phase5A0Error(
                f"source hash drift for {entry['source']}: {observed_sha} != {entry['sha256']}"
            )
        stripped = raw.strip()
        source_entries.append(
            {
                "source": entry["source"],
                "path": entry["path"],
                "source_sha256": observed_sha,
                "ingested_text_sha256": sha256_text(stripped),
                "byte_count": len(raw.encode("utf-8")),
                "character_count": len(raw),
                "line_count": entry["line_count"],
            }
        )
        source_texts[entry["source"]] = stripped

    evidence: list[EvidenceSpan] = []
    for query in oracle["queries"]:
        for relevant in query["relevant_sources"]:
            text = source_texts[relevant["source"]]
            for span in relevant["evidence"]:
                start, end = _line_char_span(text, span["line_start"], span["line_end"])
                observed_quote = text[start:end]
                if observed_quote != span["quote"]:
                    raise Phase5A0Error(
                        f"evidence quote mapping drift for {query['query_id']}/{span['span_id']}"
                    )
                evidence.append(
                    EvidenceSpan(
                        query_id=query["query_id"],
                        source=relevant["source"],
                        grade=relevant["grade"],
                        span_id=span["span_id"],
                        char_start=start,
                        char_end=end,
                        line_start=span["line_start"],
                        line_end=span["line_end"],
                    )
                )
    return source_entries, source_texts, evidence


def reconstruct_legacy_chunks(
    source_entries: list[dict[str, Any]],
    source_texts: dict[str, str],
    embedding_tokenizer: Any,
) -> list[LegacyChunk]:
    encoding = tiktoken.get_encoding(TOKENIZER_NAME)
    chunks: list[LegacyChunk] = []
    for source_entry in source_entries:
        source = source_entry["source"]
        text = source_texts[source]
        token_ids = encoding.encode(text)
        decoded, offsets = encoding.decode_with_offsets(token_ids)
        if decoded != text:
            raise Phase5A0Error(f"cl100k round-trip failed for {source}")
        for chunk_index, start in enumerate(range(0, len(token_ids), CHUNK_STRIDE)):
            end = min(start + CHUNK_SIZE, len(token_ids))
            chunk_ids = token_ids[start:end]
            chunk_text = encoding.decode(chunk_ids)
            source_char_start = offsets[start]
            source_char_end = offsets[end] if end < len(offsets) else len(text)
            boundary_decode_exact = chunk_text == text[source_char_start:source_char_end]

            encoded = embedding_tokenizer(
                chunk_text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            wordpiece_ids = encoded["input_ids"]
            wordpiece_offsets = encoded["offset_mapping"]
            wordpiece_count = len(wordpiece_ids)
            beyond = max(0, wordpiece_count - CONTENT_TOKEN_LIMIT)
            truncated = beyond > 0
            local_tail_start = (
                int(wordpiece_offsets[CONTENT_TOKEN_LIMIT][0]) if truncated else None
            )
            tail_start = (
                source_char_start + local_tail_start
                if local_tail_start is not None
                else None
            )
            text_hash = sha256_text(chunk_text)
            chunk_identity_payload = {
                "source_sha256": source_entry["source_sha256"],
                "parser_version": PARSER_VERSION,
                "chunker_version": CHUNKER_VERSION,
                "chunk_index": chunk_index,
                "source_char_start": source_char_start,
                "source_char_end": source_char_end,
                "text_sha256": text_hash,
            }
            chunks.append(
                LegacyChunk(
                    ordinal=len(chunks),
                    chunk_id=deterministic_id("chunk", chunk_identity_payload),
                    source=source,
                    source_path=source_entry["path"],
                    source_sha256=source_entry["source_sha256"],
                    chunk_index=chunk_index,
                    text=chunk_text,
                    text_sha256=text_hash,
                    source_char_start=source_char_start,
                    source_char_end=source_char_end,
                    cl100k_tokens=len(chunk_ids),
                    wordpiece_content_tokens=wordpiece_count,
                    tokens_beyond_limit=beyond,
                    truncated=truncated,
                    truncated_fraction=round(beyond / wordpiece_count, 8)
                    if wordpiece_count
                    else 0.0,
                    truncated_tail_char_start=tail_start,
                    boundary_decode_exact=boundary_decode_exact,
                )
            )
    return chunks


def _span_overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def evidence_tail_analysis(
    evidence: list[EvidenceSpan], chunks: list[LegacyChunk]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks_by_source: dict[str, list[LegacyChunk]] = {}
    for chunk in chunks:
        chunks_by_source.setdefault(chunk.source, []).append(chunk)

    span_rows: list[dict[str, Any]] = []
    query_status: dict[str, dict[str, Any]] = {}
    for span in evidence:
        partial_chunk_ids: list[str] = []
        whole_chunk_ids: list[str] = []
        for chunk in chunks_by_source[span.source]:
            tail_start = chunk.truncated_tail_char_start
            if tail_start is None:
                continue
            if _span_overlaps(
                span.char_start,
                span.char_end,
                tail_start,
                chunk.source_char_end,
            ):
                partial_chunk_ids.append(chunk.chunk_id)
                if tail_start <= span.char_start and span.char_end <= chunk.source_char_end:
                    whole_chunk_ids.append(chunk.chunk_id)
        status = "whole" if whole_chunk_ids else "partial" if partial_chunk_ids else "none"
        row = {
            "query_id": span.query_id,
            "source": span.source,
            "span_id": span.span_id,
            "line_start": span.line_start,
            "line_end": span.line_end,
            "tail_exposure": status,
            "partial_tail_chunk_ids": partial_chunk_ids,
            "whole_tail_chunk_ids": whole_chunk_ids,
        }
        span_rows.append(row)
        if status != "none":
            aggregate = query_status.setdefault(
                span.query_id,
                {"query_id": span.query_id, "tail_exposure": "partial", "span_ids": []},
            )
            aggregate["span_ids"].append(span.span_id)
            if status == "whole":
                aggregate["tail_exposure"] = "whole"
    return span_rows, [query_status[qid] for qid in sorted(query_status)]


def _rank_dense(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: list[LegacyChunk],
) -> list[dict[str, Any]]:
    scores = np.asarray(chunk_embeddings @ query_embedding, dtype=np.float64)
    order = sorted(
        range(len(chunks)),
        key=lambda index: (-float(scores[index]), chunks[index].chunk_id),
    )
    return [
        {
            "chunk_id": chunks[index].chunk_id,
            "source": chunks[index].source,
            "chunk_index": chunks[index].chunk_index,
            "native_score": round(float(scores[index]), 8),
            "rank": rank,
        }
        for rank, index in enumerate(order, start=1)
    ]


def _rank_bm25(
    query: str,
    bm25: BM25Okapi,
    chunks: list[LegacyChunk],
) -> list[dict[str, Any]]:
    scores = bm25.get_scores(query.lower().split())
    # This reproduces production np.argsort(scores)[::-1], including its stable
    # dependence on the frozen corpus order for equal BM25 scores.
    order = np.argsort(scores)[::-1]
    ranked: list[dict[str, Any]] = []
    for index in order:
        score = float(scores[index])
        if score == 0.0:
            continue
        chunk = chunks[int(index)]
        ranked.append(
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "native_score": round(score, 8),
                "rank": len(ranked) + 1,
            }
        )
    return ranked


def reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    bm25: list[dict[str, Any]],
    *,
    candidate_k: int = CANDIDATE_K,
    rrf_constant: int = RRF_CONSTANT,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for channel, rows in (("dense", dense[:candidate_k]), ("bm25", bm25[:candidate_k])):
        for zero_rank, row in enumerate(rows):
            entry = by_id.setdefault(
                row["chunk_id"],
                {
                    "chunk_id": row["chunk_id"],
                    "source": row["source"],
                    "chunk_index": row["chunk_index"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                },
            )
            entry["rrf_score"] += 1.0 / (rrf_constant + zero_rank)
            entry[f"{channel}_rank"] = zero_rank + 1
    ranked = sorted(
        by_id.values(),
        key=lambda row: (-row["rrf_score"], row["chunk_id"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["rrf_score"] = round(row["rrf_score"], 8)
    return ranked


def _relevance(query: dict[str, Any]) -> dict[str, int]:
    return {row["source"]: int(row["grade"]) for row in query["relevant_sources"]}


def hit_at_k(rows: list[dict[str, Any]], relevance: dict[str, int], k: int) -> float:
    return float(any(row["source"] in relevance for row in rows[:k]))


def source_recall_at_k(
    rows: list[dict[str, Any]], relevance: dict[str, int], k: int
) -> float:
    if not relevance:
        return 0.0
    observed = {row["source"] for row in rows[:k] if row["source"] in relevance}
    return len(observed) / len(relevance)


def precision_at_k(
    rows: list[dict[str, Any]], relevance: dict[str, int], k: int
) -> float:
    return sum(row["source"] in relevance for row in rows[:k]) / k


def reciprocal_rank(rows: list[dict[str, Any]], relevance: dict[str, int], k: int) -> float:
    for rank, row in enumerate(rows[:k], start=1):
        if row["source"] in relevance:
            return 1.0 / rank
    return 0.0


def graded_ndcg_at_k(
    rows: list[dict[str, Any]], relevance: dict[str, int], k: int
) -> float:
    credited: set[str] = set()
    dcg = 0.0
    for rank, row in enumerate(rows[:k], start=1):
        source = row["source"]
        if source in relevance and source not in credited:
            dcg += (2 ** relevance[source] - 1) / math.log2(rank + 1)
            credited.add(source)
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _span_fully_covered_by_rows(
    span: EvidenceSpan,
    rows: list[dict[str, Any]],
    chunks_by_id: dict[str, LegacyChunk],
    k: int,
    *,
    char_limit: int | None = None,
) -> bool:
    intervals: list[tuple[int, int]] = []
    for row in rows[:k]:
        chunk = chunks_by_id[row["chunk_id"]]
        if chunk.source != span.source:
            continue
        end = chunk.source_char_end
        if char_limit is not None:
            end = min(end, chunk.source_char_start + char_limit)
        start = max(span.char_start, chunk.source_char_start)
        stop = min(span.char_end, end)
        if start < stop:
            intervals.append((start, stop))
    if not intervals:
        return False
    intervals.sort()
    covered_until = span.char_start
    for start, end in intervals:
        if start > covered_until:
            break
        covered_until = max(covered_until, end)
        if covered_until >= span.char_end:
            return True
    return False


def _missed_span_rows(
    spans: list[EvidenceSpan],
    dense: list[dict[str, Any]],
    bm25: list[dict[str, Any]],
    hybrid: list[dict[str, Any]],
    chunks_by_id: dict[str, LegacyChunk],
    tail_span_ids: set[str],
    *,
    evaluator_ambiguous: bool,
) -> list[dict[str, Any]]:
    misses: list[dict[str, Any]] = []
    for span in spans:
        if _span_fully_covered_by_rows(
            span, hybrid, chunks_by_id, PRODUCTION_RETRIEVAL_K
        ):
            continue
        dense_candidate = _span_fully_covered_by_rows(
            span, dense, chunks_by_id, CANDIDATE_K
        )
        bm25_candidate = _span_fully_covered_by_rows(
            span, bm25, chunks_by_id, CANDIDATE_K
        )
        hybrid_top10 = _span_fully_covered_by_rows(
            span, hybrid, chunks_by_id, FINAL_K
        )
        layers: list[str] = []
        confidence = "demonstrated"
        if evaluator_ambiguous:
            layers.append("evaluator ambiguity")
        if dense_candidate or bm25_candidate:
            layers.append("fusion/ranking failure")
        else:
            layers.append("candidate-K failure")
            if not dense_candidate:
                layers.append("embedding semantic failure")
            if not bm25_candidate:
                layers.append("lexical/BM25 failure")
        if hybrid_top10:
            layers.append("final-K ranking failure")
        if span.span_id in tail_span_ids:
            layers.append("embedding truncation")
            confidence = "plausible-contributor-not-causally-isolated"
        misses.append(
            {
                "span_id": span.span_id,
                "source": span.source,
                "grade": span.grade,
                "dense_candidate_top20": dense_candidate,
                "bm25_candidate_top20": bm25_candidate,
                "hybrid_covered_top10": hybrid_top10,
                "causal_layers": sorted(set(layers)),
                "causal_confidence": confidence,
                "critical": span.grade == 2,
            }
        )
    return misses


def evidence_span_recall(
    spans: list[EvidenceSpan],
    rows: list[dict[str, Any]],
    chunks_by_id: dict[str, LegacyChunk],
    k: int,
    *,
    char_limit: int | None = None,
) -> float:
    if not spans:
        return 0.0
    covered = sum(
        _span_fully_covered_by_rows(
            span, rows, chunks_by_id, k, char_limit=char_limit
        )
        for span in spans
    )
    return covered / len(spans)


def _rank_summary(
    rows: list[dict[str, Any]],
    query: dict[str, Any],
    query_spans: list[EvidenceSpan],
    chunks_by_id: dict[str, LegacyChunk],
) -> dict[str, Any]:
    relevance = _relevance(query)
    summary: dict[str, Any] = {}
    for k in K_VALUES:
        summary[f"hit@{k}"] = round(hit_at_k(rows, relevance, k), 8)
        summary[f"source_recall@{k}"] = round(
            source_recall_at_k(rows, relevance, k), 8
        )
        summary[f"precision@{k}"] = round(precision_at_k(rows, relevance, k), 8)
        summary[f"ndcg@{k}"] = round(graded_ndcg_at_k(rows, relevance, k), 8)
        summary[f"evidence_span_recall@{k}"] = round(
            evidence_span_recall(query_spans, rows, chunks_by_id, k), 8
        )
    summary["mrr@10"] = round(reciprocal_rank(rows, relevance, 10), 8)
    return summary


def _aggregate_metrics(
    per_query: list[dict[str, Any]], channel: str, *, gating_only: bool
) -> dict[str, Any]:
    eligible = [row for row in per_query if not gating_only or row["gating_eligible"]]
    metrics = eligible[0]["channels"][channel]["metrics"]
    return {
        "query_count": len(eligible),
        **{
            key: round(
                sum(row["channels"][channel]["metrics"][key] for row in eligible)
                / len(eligible),
                8,
            )
            for key in metrics
        },
    }


def _source_best_rank(rows: list[dict[str, Any]], source: str) -> int | None:
    return next((row["rank"] for row in rows if row["source"] == source), None)


def _failure_rows(
    query: dict[str, Any],
    dense: list[dict[str, Any]],
    bm25: list[dict[str, Any]],
    hybrid: list[dict[str, Any]],
    tail_query_ids: set[str],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    hybrid_top_sources = {row["source"] for row in hybrid[:PRODUCTION_RETRIEVAL_K]}
    for relevant in query["relevant_sources"]:
        source = relevant["source"]
        if source in hybrid_top_sources:
            continue
        dense_rank = _source_best_rank(dense, source)
        bm25_rank = _source_best_rank(bm25, source)
        layers: list[str] = []
        confidence = "demonstrated"
        if not query["gating_eligible"]:
            layers.append("evaluator ambiguity")
        if (dense_rank is not None and dense_rank <= CANDIDATE_K) or (
            bm25_rank is not None and bm25_rank <= CANDIDATE_K
        ):
            layers.append("fusion/ranking failure")
        if dense_rank is None or dense_rank > CANDIDATE_K:
            layers.append("embedding semantic failure")
        if bm25_rank is None or bm25_rank > CANDIDATE_K:
            layers.append("lexical/BM25 failure")
        if (dense_rank is not None and dense_rank > CANDIDATE_K) or (
            bm25_rank is not None and bm25_rank > CANDIDATE_K
        ):
            layers.append("candidate-K failure")
        if query["query_id"] in tail_query_ids and (
            dense_rank is None or dense_rank > CANDIDATE_K
        ):
            layers.append("embedding truncation")
            confidence = "plausible-contributor-not-causally-isolated"
        failures.append(
            {
                "source": source,
                "grade": relevant["grade"],
                "dense_best_rank": dense_rank,
                "bm25_best_rank": bm25_rank,
                "hybrid_best_rank": _source_best_rank(hybrid, source),
                "causal_layers": sorted(set(layers)),
                "causal_confidence": confidence,
                "critical": relevant["grade"] == 2,
            }
        )
    return failures


def _bm25_diagnostics() -> list[dict[str, Any]]:
    cases = [
        "API",
        "C++",
        "foo.bar",
        "interrupt()",
        "tools/query_kb.py",
        "state-of-the-art",
        "v1.9.2",
        "RRF(k=60)",
    ]
    return [{"input": case, "tokens": case.lower().split()} for case in cases]


def build_manifest(
    source_entries: list[dict[str, Any]],
    chunks: list[LegacyChunk],
    model_snapshot: Path,
    tokenizer: Any,
    encoder: SentenceTransformer,
) -> dict[str, Any]:
    ordered_source_identity = [
        {
            "ordinal": ordinal,
            "source": entry["source"],
            "path": entry["path"],
            "source_sha256": entry["source_sha256"],
        }
        for ordinal, entry in enumerate(source_entries)
    ]
    ordered_chunk_identity = [
        {
            "ordinal": chunk.ordinal,
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "source_char_start": chunk.source_char_start,
            "source_char_end": chunk.source_char_end,
            "text_sha256": chunk.text_sha256,
        }
        for chunk in chunks
    ]
    tokenizer_files = {
        name: sha256_bytes((model_snapshot / name).read_bytes())
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt")
    }
    return {
        "schema_version": "phase5a0_baseline_a_manifest_v1",
        "starting_head": STARTING_HEAD,
        "corpus": {
            "source_root": "kb/seed_docs",
            "source_count": len(source_entries),
            "ordered_source_manifest": ordered_source_identity,
            "aggregate_fingerprint_algorithm": "sha256(canonical-json(ordered_source_manifest))",
            "aggregate_fingerprint": sha256_text(
                canonical_json_dumps(ordered_source_identity)
            ),
        },
        "parser": {"name": "direct UTF-8 read + str.strip", "version": PARSER_VERSION},
        "chunking": {
            "version": CHUNKER_VERSION,
            "tokenizer": TOKENIZER_NAME,
            "size_tokens": CHUNK_SIZE,
            "overlap_tokens": CHUNK_OVERLAP,
            "stride_tokens": CHUNK_STRIDE,
            "chunk_count": len(chunks),
            "ordered_chunk_fingerprint_algorithm": (
                "sha256(canonical-json(ordered chunk identity rows))"
            ),
            "ordered_chunk_fingerprint": sha256_text(
                canonical_json_dumps(ordered_chunk_identity)
            ),
            "ordered_chunks": ordered_chunk_identity,
        },
        "embedding": {
            "model_id": MODEL_ID,
            "local_resolved_revision": MODEL_REVISION,
            "historical_production_revision": "NOT_VERIFIABLE",
            "snapshot_revision_directory": f"snapshots/{MODEL_REVISION}",
            "model_safetensors_sha256": sha256_bytes(
                (model_snapshot / "model.safetensors").read_bytes()
            ),
            "tokenizer_class": tokenizer.__class__.__name__,
            "tokenizer_is_fast": bool(tokenizer.is_fast),
            "tokenizer_file_sha256": tokenizer_files,
            "runtime_max_sequence_length": encoder.max_seq_length,
            "special_tokens": MODEL_SPECIAL_TOKENS,
            "effective_content_token_limit": CONTENT_TOKEN_LIMIT,
            "dimension": encoder.get_embedding_dimension(),
            "normalization": "model Normalize module; evaluator also requests normalize_embeddings=True",
            "runtime_libraries": {
                package: importlib.metadata.version(package)
                for package in (
                    "sentence-transformers",
                    "transformers",
                    "tokenizers",
                    "tiktoken",
                    "rank-bm25",
                )
            },
        },
        "dense": {
            "reference_search": "exhaustive float32 cosine over normalized vectors",
            "candidate_k": CANDIDATE_K,
            "tie_break": "native score descending, chunk_id ascending",
            "qdrant_mutated": False,
        },
        "bm25": {
            "implementation": "rank_bm25.BM25Okapi defaults",
            "tokenization": "text.lower().split()",
            "zero_score_candidates_excluded": True,
            "candidate_k": CANDIDATE_K,
            "tie_break": "np.argsort(scores)[::-1] over frozen corpus order",
        },
        "fusion": {
            "algorithm": "reciprocal rank fusion over chunk_id",
            "rank_origin": 0,
            "rrf_constant": RRF_CONSTANT,
            "final_k": FINAL_K,
            "tie_break": "rrf_score descending, chunk_id ascending",
        },
        "production_exposure_observed": {
            "retrieve_k": PRODUCTION_RETRIEVAL_K,
            "drafter_k": DRAFTER_K,
            "drafter_character_prefix_per_chunk": DRAFTER_CHAR_LIMIT,
            "verifier_k": VERIFIER_K,
            "verifier_character_clip": None,
        },
        "index_reference": {
            "experimental_reference": "exact/plain search",
            "serving_collection": "machinist_evergreen",
            "serving_collection_read": False,
            "serving_collection_mutated": False,
            "historical_production_identity": "NOT_VERIFIABLE",
        },
    }


def run_measurement(model_snapshot: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require_starting_head()
    oracle = load_oracle()
    oracle_summary = validate_oracle(oracle)
    source_entries, source_texts, evidence = load_sources_and_evidence(oracle)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_snapshot), local_files_only=True, use_fast=True
    )
    encoder = SentenceTransformer(str(model_snapshot), local_files_only=True)
    if not tokenizer.is_fast:
        raise Phase5A0Error("fast tokenizer with offset mapping is required")
    if encoder.max_seq_length != MODEL_MAX_SEQUENCE_LENGTH:
        raise Phase5A0Error(
            f"MiniLM max sequence drift: {encoder.max_seq_length} != {MODEL_MAX_SEQUENCE_LENGTH}"
        )
    if encoder.get_embedding_dimension() != EMBEDDING_DIMENSION:
        raise Phase5A0Error("MiniLM embedding dimension drift")

    chunks = reconstruct_legacy_chunks(source_entries, source_texts, tokenizer)
    if len(chunks) != 73:
        raise Phase5A0Error(f"legacy chunk count drift: {len(chunks)} != 73")

    manifest = build_manifest(source_entries, chunks, model_snapshot, tokenizer, encoder)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    span_tail_rows, tail_queries = evidence_tail_analysis(evidence, chunks)
    tail_query_ids = {row["query_id"] for row in tail_queries}
    tail_span_ids = {
        row["span_id"] for row in span_tail_rows if row["tail_exposure"] != "none"
    }

    chunk_embeddings = encoder.encode(
        [chunk.text for chunk in chunks],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    query_embeddings = encoder.encode(
        [query["query"] for query in oracle["queries"]],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    bm25 = BM25Okapi([chunk.text.lower().split() for chunk in chunks])
    evidence_by_query: dict[str, list[EvidenceSpan]] = {}
    for span in evidence:
        evidence_by_query.setdefault(span.query_id, []).append(span)

    per_query: list[dict[str, Any]] = []
    dense_only_wins: list[str] = []
    bm25_only_wins: list[str] = []
    hybrid_wins: list[str] = []
    fusion_hurts: list[str] = []
    for query_index, query in enumerate(oracle["queries"]):
        dense = _rank_dense(query_embeddings[query_index], chunk_embeddings, chunks)
        bm25_rows = _rank_bm25(query["query"], bm25, chunks)
        hybrid = reciprocal_rank_fusion(dense, bm25_rows)
        spans = evidence_by_query.get(query["query_id"], [])
        channels: dict[str, Any] = {}
        for name, rows in (("dense", dense), ("bm25", bm25_rows), ("hybrid", hybrid)):
            channels[name] = {
                "metrics": _rank_summary(rows, query, spans, chunks_by_id),
                "top10": rows[:FINAL_K],
            }

        score_tuple = {
            channel: (
                channels[channel]["metrics"]["source_recall@5"],
                channels[channel]["metrics"]["evidence_span_recall@5"],
                channels[channel]["metrics"]["ndcg@5"],
            )
            for channel in ("dense", "bm25", "hybrid")
        }
        if score_tuple["dense"] > max(score_tuple["bm25"], score_tuple["hybrid"]):
            dense_only_wins.append(query["query_id"])
        if score_tuple["bm25"] > max(score_tuple["dense"], score_tuple["hybrid"]):
            bm25_only_wins.append(query["query_id"])
        if score_tuple["hybrid"] > max(score_tuple["dense"], score_tuple["bm25"]):
            hybrid_wins.append(query["query_id"])
        if score_tuple["hybrid"] < max(score_tuple["dense"], score_tuple["bm25"]):
            fusion_hurts.append(query["query_id"])

        exposure = {
            "retrieved_top5_evidence_span_recall": round(
                evidence_span_recall(spans, hybrid, chunks_by_id, PRODUCTION_RETRIEVAL_K),
                8,
            ),
            "expanded_top5_evidence_span_recall": round(
                evidence_span_recall(spans, hybrid, chunks_by_id, PRODUCTION_RETRIEVAL_K),
                8,
            ),
            "packed_top5_evidence_span_recall": round(
                evidence_span_recall(spans, hybrid, chunks_by_id, PRODUCTION_RETRIEVAL_K),
                8,
            ),
            "drafter_exposed_evidence_span_recall": round(
                evidence_span_recall(
                    spans,
                    hybrid,
                    chunks_by_id,
                    DRAFTER_K,
                    char_limit=DRAFTER_CHAR_LIMIT,
                ),
                8,
            ),
            "verifier_exposed_evidence_span_recall": round(
                evidence_span_recall(spans, hybrid, chunks_by_id, VERIFIER_K), 8
            ),
        }
        exposure["lost_after_retrieval_for_drafter"] = (
            exposure["retrieved_top5_evidence_span_recall"]
            > exposure["drafter_exposed_evidence_span_recall"]
        )
        exposure["lost_after_retrieval_for_verifier"] = (
            exposure["retrieved_top5_evidence_span_recall"]
            > exposure["verifier_exposed_evidence_span_recall"]
        )
        failures = _failure_rows(
            query, dense, bm25_rows, hybrid, tail_query_ids
        )
        span_misses = _missed_span_rows(
            spans,
            dense,
            bm25_rows,
            hybrid,
            chunks_by_id,
            tail_span_ids,
            evaluator_ambiguous=not query["gating_eligible"],
        )
        per_query.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "answerability": query["answerability"],
                "gating_eligible": query["gating_eligible"],
                "v1_label_verdict": query["v1_label_verdict"],
                "query_bucket": query.get("query_bucket"),
                "failure_tags": query.get("failure_tags", []),
                "relevant_sources": query["relevant_sources"],
                "channels": channels,
                "evidence_exposure": exposure,
                "missed_evidence_at_hybrid_k5": failures,
                "missed_evidence_spans_at_hybrid_k5": span_misses,
            }
        )

    aggregate = {
        channel: {
            "all_35_diagnostic": _aggregate_metrics(
                per_query, channel, gating_only=False
            ),
            "gating_33_excludes_q25_q26": _aggregate_metrics(
                per_query, channel, gating_only=True
            ),
        }
        for channel in ("dense", "bm25", "hybrid")
    }

    exposure_keys = list(per_query[0]["evidence_exposure"])
    exposure_aggregate: dict[str, Any] = {"query_count": len(per_query)}
    for key in exposure_keys:
        values = [row["evidence_exposure"][key] for row in per_query]
        exposure_aggregate[key] = (
            sum(bool(value) for value in values)
            if key.startswith("lost_after")
            else round(sum(float(value) for value in values) / len(values), 8)
        )

    truncation_rows = [
        {
            "ordinal": chunk.ordinal,
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "cl100k_tokens": chunk.cl100k_tokens,
            "wordpiece_content_tokens": chunk.wordpiece_content_tokens,
            "tokens_beyond_254": chunk.tokens_beyond_limit,
            "truncated": chunk.truncated,
            "truncated_fraction": chunk.truncated_fraction,
            "source_char_start": chunk.source_char_start,
            "source_char_end": chunk.source_char_end,
            "truncated_tail_char_start": chunk.truncated_tail_char_start,
            "boundary_decode_exact": chunk.boundary_decode_exact,
        }
        for chunk in chunks
    ]
    exposed = [row for row in truncation_rows if row["truncated"]]
    severity = Counter(
        "low_1_25_percent"
        if row["truncated_fraction"] <= 0.25
        else "medium_25_50_percent"
        if row["truncated_fraction"] <= 0.50
        else "high_over_50_percent"
        for row in exposed
    )
    truncation_report = {
        "schema_version": "phase5a0_legacy_truncation_v1",
        "starting_head": STARTING_HEAD,
        "effective_content_token_limit": CONTENT_TOKEN_LIMIT,
        "chunk_count": len(chunks),
        "truncation_exposed_chunks": len(exposed),
        "truncation_exposed_percentage": round(100 * len(exposed) / len(chunks), 4),
        "severity_distribution": dict(sorted(severity.items())),
        "boundary_decode_non_exact_chunks": sum(
            not row["boundary_decode_exact"] for row in truncation_rows
        ),
        "evidence_tail_affected_query_count": len(tail_queries),
        "evidence_tail_affected_queries": tail_queries,
        "evidence_span_tail_analysis": span_tail_rows,
        "chunks": truncation_rows,
    }

    report = {
        "schema_version": "phase5a0_baseline_a_report_v1",
        "starting_head": STARTING_HEAD,
        "oracle_validation": oracle_summary,
        "dataset_status": oracle["dataset_status"],
        "metric_semantics": {
            "recall": "macro unique-source recall; duplicate chunks credited once",
            "hit": "macro fraction with any relevant source",
            "precision": "relevant chunk slots divided by K; duplicate relevant chunks count as relevant slots",
            "mrr": "first relevant chunk, capped at 10",
            "ndcg": "graded source nDCG (grade 2/1), first occurrence per source only",
            "evidence_span_recall": "fraction of adjudicated exact source spans fully covered by the union of top-K chunk source intervals",
        },
        "aggregate_metrics": aggregate,
        "evidence_exposure": exposure_aggregate,
        "comparative_diagnostics_at_k5": {
            "winner_metric_order": [
                "source_recall@5",
                "evidence_span_recall@5",
                "ndcg@5",
            ],
            "dense_only_wins": dense_only_wins,
            "bm25_only_wins": bm25_only_wins,
            "hybrid_wins": hybrid_wins,
            "fusion_hurts": fusion_hurts,
        },
        "critical_evidence_misses": [
            {
                "query_id": row["query_id"],
                "source_misses": [
                    failure
                    for failure in row["missed_evidence_at_hybrid_k5"]
                    if failure["critical"]
                ],
                "span_misses": [
                    failure
                    for failure in row["missed_evidence_spans_at_hybrid_k5"]
                    if failure["critical"]
                ],
            }
            for row in per_query
            if any(failure["critical"] for failure in row["missed_evidence_at_hybrid_k5"])
            or any(
                failure["critical"]
                for failure in row["missed_evidence_spans_at_hybrid_k5"]
            )
        ],
        "bm25_audit": {
            "production_tokenization": "lower().split()",
            "diagnostic_cases": _bm25_diagnostics(),
            "conclusion": (
                "Token preservation is punctuation-sensitive and no normalization occurs. "
                "V2 channel misses are reported, but v2 does not isolate tokenizer redesign "
                "as a causal improvement; no BM25 change is authorized in 5A0."
            ),
        },
        "per_query": per_query,
        "absence_metrics": None,
        "absence_metrics_reason": "retrieval golden v2 contains zero ABSENT cases",
        "provider_calls": 0,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
    }
    return manifest, truncation_report, report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-snapshot", type=Path)
    args = parser.parse_args()
    snapshot = resolve_local_model_snapshot(args.model_snapshot)
    manifest, truncation, report = run_measurement(snapshot)
    output_dir = args.output_dir.resolve()
    write_json(output_dir / "baseline_a_manifest.json", manifest)
    write_json(output_dir / "legacy_chunk_truncation.json", truncation)
    write_json(output_dir / "baseline_a_report.json", report)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "corpus_fingerprint": manifest["corpus"]["aggregate_fingerprint"],
                "legacy_chunk_fingerprint": manifest["chunking"]["ordered_chunk_fingerprint"],
                "chunk_count": manifest["chunking"]["chunk_count"],
                "truncation_exposed_chunks": truncation["truncation_exposed_chunks"],
                "evidence_tail_affected_query_count": truncation[
                    "evidence_tail_affected_query_count"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
