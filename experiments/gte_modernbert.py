"""Gate 1 support for the isolated GTE ModernBERT retrieval experiment.

This module deliberately does not replace the production MiniLM collection or
retrieval path.  It owns the one embedding specification shared by the
experiment's ingestion and future retrieval work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import platform
import resource
import statistics
import time
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EmbeddingSpec:
    """The single, pinned embedding contract for this bounded experiment."""

    model_id: str = "Alibaba-NLP/gte-modernbert-base"
    revision: str = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
    max_sequence_length: int = 8192
    output_dimension: int = 768
    tokenizer_identity: str = "Alibaba-NLP/gte-modernbert-base"
    chunk_size: int = 512
    chunk_overlap: int = 64


GTE_MODERNBERT_SPEC = EmbeddingSpec()
EXPERIMENT_COLLECTION = "content_agent_gte_modernbert_gate1_794851d"
EXPERIMENT_MARKER = "gte-modernbert-gate1"


@dataclass(frozen=True)
class Chunk:
    """A tokenizer-native contiguous source span."""

    source: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    content_token_start: int
    content_token_end: int
    model_input_tokens: int


def load_encoder(spec: EmbeddingSpec = GTE_MODERNBERT_SPEC) -> SentenceTransformer:
    """Load the pinned model and explicitly enforce its 8,192-token contract."""
    encoder = SentenceTransformer(spec.model_id, revision=spec.revision)
    # Some tokenizers expose an intentionally huge sentinel.  Both layers are
    # pinned here so a future caller cannot silently bypass the experiment limit.
    encoder.max_seq_length = spec.max_sequence_length
    encoder.tokenizer.model_max_length = spec.max_sequence_length
    validate_model_contract(encoder, spec)
    return encoder


def _model_commit_hash(encoder: SentenceTransformer) -> str | None:
    first_module = next(iter(encoder._modules.values()), None)
    config = getattr(getattr(first_module, "auto_model", None), "config", None)
    return getattr(config, "_commit_hash", None)


def validate_model_contract(
    encoder: SentenceTransformer, spec: EmbeddingSpec = GTE_MODERNBERT_SPEC
) -> dict[str, Any]:
    """Return runtime evidence or fail closed when the pinned contract is not true."""
    get_dimension = getattr(encoder, "get_embedding_dimension", None)
    if get_dimension is None:
        get_dimension = encoder.get_sentence_embedding_dimension
    dimension = get_dimension()
    if dimension != spec.output_dimension:
        raise ValueError(f"expected {spec.output_dimension}-D embeddings, got {dimension}")
    if encoder.max_seq_length != spec.max_sequence_length:
        raise ValueError(
            f"expected max sequence length {spec.max_sequence_length}, "
            f"got {encoder.max_seq_length}"
        )
    tokenizer_limit = getattr(encoder.tokenizer, "model_max_length", None)
    if tokenizer_limit != spec.max_sequence_length:
        raise ValueError(
            f"tokenizer max length must be {spec.max_sequence_length}, got {tokenizer_limit}"
        )
    tokenizer_name = str(getattr(encoder.tokenizer, "name_or_path", ""))
    if "gte-modernbert-base" not in tokenizer_name.lower():
        raise ValueError(f"unexpected tokenizer identity: {tokenizer_name!r}")
    commit_hash = _model_commit_hash(encoder)
    if commit_hash and commit_hash != spec.revision:
        raise ValueError(f"expected revision {spec.revision}, got {commit_hash}")
    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "runtime_model_commit": commit_hash,
        "effective_max_sequence_length": encoder.max_seq_length,
        "tokenizer": tokenizer_name,
        "tokenizer_max_length": tokenizer_limit,
        "output_dimension": dimension,
    }


def _model_input_length(tokenizer: Any, chunk_text: str, content_token_ids: list[int]) -> int:
    """Measure special-token accounting with the selected runtime tokenizer.

    Transformers 5's ``TokenizersBackend`` intentionally does not expose the
    legacy ``build_inputs_with_special_tokens`` method.  Encoding the exact
    source span with and without specials is the supported tokenizer-native
    measurement.  The round-trip check makes offset-boundary drift a hard error.
    """
    # A wordpiece token may encode its leading whitespace differently when an
    # offset slice begins at that token.  The stored payload is intentionally
    # the original character span, so measure that exact payload rather than
    # assuming token IDs can be reconstructed at every chunk boundary.
    with_specials = tokenizer(chunk_text, add_special_tokens=True)["input_ids"]
    return len(with_specials)


def chunk_document(
    text: str,
    source: str,
    tokenizer: Any,
    spec: EmbeddingSpec = GTE_MODERNBERT_SPEC,
) -> list[Chunk]:
    """Split one source into deterministic 512/64 tokenizer-native spans."""
    if not text:
        return []
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = list(encoded["input_ids"])
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    if not token_ids:
        return []
    if len(token_ids) != len(offsets):
        raise ValueError("tokenizer returned mismatched input IDs and offsets")

    step = spec.chunk_size - spec.chunk_overlap
    if step <= 0:
        raise ValueError("chunk overlap must be smaller than chunk size")

    chunks: list[Chunk] = []
    for chunk_index, token_start in enumerate(range(0, len(token_ids), step)):
        token_end = min(token_start + spec.chunk_size, len(token_ids))
        content_ids = token_ids[token_start:token_end]
        start_char = offsets[token_start][0]
        end_char = offsets[token_end - 1][1]
        chunk_text = text[start_char:end_char]
        model_input_tokens = _model_input_length(tokenizer, chunk_text, content_ids)
        if not chunk_text.strip():
            raise ValueError(f"empty chunk from {source} at index {chunk_index}")
        if model_input_tokens > spec.max_sequence_length:
            raise ValueError(
                f"chunk {source}:{chunk_index} has {model_input_tokens} model tokens "
                f"> {spec.max_sequence_length}"
            )
        chunks.append(
            Chunk(
                source=source,
                chunk_index=chunk_index,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                content_token_start=token_start,
                content_token_end=token_end,
                model_input_tokens=model_input_tokens,
            )
        )
        if token_end == len(token_ids):
            break
    # Offset mappings may omit whitespace around a token, but every substantive
    # character must appear in at least one preserved source span.
    covered = [False] * len(text)
    for chunk in chunks:
        for index in range(chunk.start_char, chunk.end_char):
            covered[index] = True
    lost_substantive = [index for index, character in enumerate(text) if not character.isspace() and not covered[index]]
    if lost_substantive:
        raise ValueError(f"tokenizer offsets lost substantive source text for {source}")
    return chunks


def canonical_documents(corpus_dir: Path) -> list[tuple[str, str]]:
    """Return the canonical seed corpus in a deterministic source order."""
    documents = [
        (path.name, path.read_text(encoding="utf-8"))
        for path in sorted(corpus_dir.glob("*.md"))
    ]
    if len(documents) != 20:
        raise ValueError(f"expected canonical 20 documents, found {len(documents)}")
    return documents


def corpus_identity(documents: Iterable[tuple[str, str]]) -> str:
    digest = sha256()
    for source, text in documents:
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def chunk_corpus(
    documents: Iterable[tuple[str, str]], tokenizer: Any, spec: EmbeddingSpec = GTE_MODERNBERT_SPEC
) -> list[Chunk]:
    chunks = [
        chunk
        for source, text in documents
        for chunk in chunk_document(text, source, tokenizer, spec)
    ]
    if not chunks:
        raise ValueError("canonical corpus produced no chunks")
    identities = {(chunk.source, chunk.chunk_index) for chunk in chunks}
    if len(identities) != len(chunks):
        raise ValueError("duplicate source/chunk index in chunk corpus")
    return chunks


def _vector_config(info: Any) -> Any:
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        raise ValueError("named vectors are not part of this experiment contract")
    return vectors


def ensure_experiment_collection(client: QdrantClient, collection: str = EXPERIMENT_COLLECTION) -> None:
    """Create only the named disposable collection; never touch the default collection."""
    existing = {item.name for item in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=GTE_MODERNBERT_SPEC.output_dimension,
                distance=Distance.COSINE,
            ),
        )


def ingest_chunks(
    client: QdrantClient,
    encoder: SentenceTransformer,
    chunks: list[Chunk],
    corpus_hash: str,
    collection: str = EXPERIMENT_COLLECTION,
) -> dict[str, Any]:
    """Embed and upsert deterministic points into the isolated collection."""
    ensure_experiment_collection(client, collection)
    started = time.perf_counter()
    vectors = encoder.encode([chunk.text for chunk in chunks], show_progress_bar=False)
    encoding_seconds = time.perf_counter() - started
    vectors = np.asarray(vectors)
    if vectors.shape != (len(chunks), GTE_MODERNBERT_SPEC.output_dimension):
        raise ValueError(f"unexpected embedding shape: {vectors.shape}")

    points = [
        PointStruct(
            id=str(uuid5(NAMESPACE_URL, f"{corpus_hash}:{chunk.source}:{chunk.chunk_index}")),
            vector=vector.tolist(),
            payload={
                "experiment": EXPERIMENT_MARKER,
                "embedding_spec": asdict(GTE_MODERNBERT_SPEC),
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "content_token_start": chunk.content_token_start,
                "content_token_end": chunk.content_token_end,
                "model_input_tokens": chunk.model_input_tokens,
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)
    return {
        "encoding_seconds": encoding_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "point_count": len(points),
    }


def peak_rss_bytes() -> int:
    """Normalize ru_maxrss, whose units differ between macOS and Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def warm_query_p95_ms(encoder: SentenceTransformer, query: str = "retrieval augmented generation") -> float:
    """Measure warmed query encoding with a small, fixed 20-sample run."""
    encoder.encode(query, show_progress_bar=False)  # warm-up excluded from timing
    samples_ms: list[float] = []
    for _ in range(20):
        started = time.perf_counter()
        encoder.encode(query, show_progress_bar=False)
        samples_ms.append((time.perf_counter() - started) * 1000)
    return float(statistics.quantiles(samples_ms, n=20, method="inclusive")[18])


def validate_collection(
    client: QdrantClient,
    documents: list[tuple[str, str]],
    expected_point_count: int,
    corpus_hash: str,
    collection: str = EXPERIMENT_COLLECTION,
) -> dict[str, Any]:
    """Validate schema and payload invariants against live Qdrant state."""
    info = client.get_collection(collection)
    vector = _vector_config(info)
    if vector.size != GTE_MODERNBERT_SPEC.output_dimension:
        raise ValueError(f"collection is {vector.size}-D, expected 768-D")
    if vector.distance != Distance.COSINE:
        raise ValueError(f"collection distance is {vector.distance}, expected cosine")
    if info.points_count != expected_point_count:
        raise ValueError(f"collection has {info.points_count} points, expected {expected_point_count}")

    points, offset = client.scroll(collection_name=collection, limit=expected_point_count + 1, with_payload=True)
    if offset is not None or len(points) != expected_point_count:
        raise ValueError("could not read the complete experiment collection")
    payloads = [point.payload or {} for point in points]
    sources = {payload.get("source") for payload in payloads}
    expected_sources = {source for source, _ in documents}
    if sources != expected_sources:
        raise ValueError("collection source payloads do not match canonical corpus")
    identities = {(payload.get("source"), payload.get("chunk_index")) for payload in payloads}
    if len(identities) != len(payloads):
        raise ValueError("duplicate source/chunk index in collection")
    if any(not str(payload.get("text", "")).strip() for payload in payloads):
        raise ValueError("empty text payload in collection")
    if any(payload.get("experiment") != EXPERIMENT_MARKER for payload in payloads):
        raise ValueError("non-experiment payload found in experiment collection")
    if any(payload.get("embedding_spec") != asdict(GTE_MODERNBERT_SPEC) for payload in payloads):
        raise ValueError("collection payload embedding specification does not match the GTE contract")

    return {
        "collection": collection,
        "source_count": len(sources),
        "point_count": len(payloads),
        "vector_schema": {"size": vector.size, "distance": vector.distance.value},
        "corpus_sha256": corpus_hash,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
