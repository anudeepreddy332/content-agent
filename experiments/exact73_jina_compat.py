"""Exact-73, embedding-only retrieval experiment support.

This module deliberately does not import or mutate the production Qdrant
singletons.  The historical ``cl100k_base`` chunks, BM25 semantics, RRF-60,
and frozen evaluator are held fixed; only the dense encoder varies by arm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import resource
import shutil
import statistics
import tempfile
import time
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import tiktoken
from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

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


EXPECTED_FIXTURE_FINGERPRINT = "4a1d5d1d67b56867c71497cb58ed4964d356a122a14d47ef822c227dba5924e4"
HISTORICAL_CHUNK_SIZE = 400
HISTORICAL_CHUNK_OVERLAP = 50
RRF_K = 60
EVALUATION_DEPTHS = (1, 3, 5)
# The accepted historical report displays 29/30 as 0.967.  Preserve its exact
# underlying count so an unrounded calculation is not incorrectly rejected.
HIT_AT_1_MINIMUM = 29 / 30
# The model snapshot's auto_map names this separate repository without a
# revision.  This is the latest implementation commit preceding the pinned
# model snapshot's 2023-10-26T11:29:31Z timestamp; pinning it avoids executing
# a floating ``main`` revision while retaining the model's intended code.
JINA_REMOTE_CODE_REPOSITORY = "jinaai/jina-bert-implementation"
JINA_REMOTE_CODE_REVISION = "a9db86227f71a0bd7bc05e5dda0359f1e09abb0f"


@dataclass(frozen=True)
class EncoderSpec:
    """Pinned dense-encoder contract for one arm of the experiment."""

    name: str
    model_id: str
    revision: str | None
    dimension: int
    max_sequence_length: int


BASELINE_SPEC = EncoderSpec(
    name="minilm",
    model_id="all-MiniLM-L6-v2",
    revision=None,
    dimension=384,
    max_sequence_length=256,
)
CANDIDATE_SPEC = EncoderSpec(
    name="jina-v2-small-en",
    model_id="jinaai/jina-embeddings-v2-small-en",
    revision="1c993a952ef47cdd9e3576c1f22f935e5252f40c",
    dimension=512,
    max_sequence_length=8192,
)


@dataclass(frozen=True)
class FrozenChunk:
    """A byte-preserved historical payload with its stable identity."""

    source: str
    chunk_index: int
    text: str

    @property
    def identity(self) -> tuple[str, int]:
        return (self.source, self.chunk_index)


def _fixture_rows(chunks: Iterable[FrozenChunk]) -> list[list[object]]:
    return [[chunk.source, chunk.chunk_index, chunk.text] for chunk in chunks]


def semantic_fingerprint(chunks: Iterable[FrozenChunk]) -> str:
    """Hash the historical source/index/text tuple sequence without formatting drift."""
    encoded = json.dumps(
        _fixture_rows(chunks), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_exact73_fixture(corpus_dir: Path) -> list[FrozenChunk]:
    """Regenerate the exact historical 400/50 cl100k_base chunk payloads."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks: list[FrozenChunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        tokens = tokenizer.encode(path.read_text(encoding="utf-8"))
        for chunk_index, start in enumerate(
            range(0, len(tokens), HISTORICAL_CHUNK_SIZE - HISTORICAL_CHUNK_OVERLAP)
        ):
            chunks.append(
                FrozenChunk(
                    source=path.stem,
                    chunk_index=chunk_index,
                    text=tokenizer.decode(tokens[start : start + HISTORICAL_CHUNK_SIZE]),
                )
            )
    validate_exact73_fixture(chunks)
    return chunks


def validate_exact73_fixture(chunks: list[FrozenChunk]) -> dict[str, Any]:
    """Fail closed unless the frozen historical fixture is reproduced exactly."""
    sources = {chunk.source for chunk in chunks}
    identities = {chunk.identity for chunk in chunks}
    fingerprint = semantic_fingerprint(chunks)
    if len(chunks) != 73:
        raise ValueError(f"expected 73 historical chunks, got {len(chunks)}")
    if len(sources) != 20:
        raise ValueError(f"expected 20 sources, got {len(sources)}")
    if len(identities) != 73:
        raise ValueError("historical chunk source/index identities are not unique")
    if fingerprint != EXPECTED_FIXTURE_FINGERPRINT:
        raise ValueError(
            "historical fixture fingerprint mismatch: "
            f"expected {EXPECTED_FIXTURE_FINGERPRINT}, got {fingerprint}"
        )
    return {"chunk_count": len(chunks), "source_count": len(sources), "fingerprint": fingerprint}


def prepare_jina_snapshot() -> dict[str, Any]:
    """Prepare the model and separately pinned custom code before execution."""
    model_snapshot = Path(
        snapshot_download(
            repo_id=CANDIDATE_SPEC.model_id,
            revision=CANDIDATE_SPEC.revision,
        )
    ).resolve()
    if model_snapshot.name != CANDIDATE_SPEC.revision:
        raise ValueError(
            "Jina snapshot does not resolve to the requested revision: "
            f"expected {CANDIDATE_SPEC.revision}, got {model_snapshot.name}"
        )
    code_snapshot = Path(
        snapshot_download(
            repo_id=JINA_REMOTE_CODE_REPOSITORY,
            revision=JINA_REMOTE_CODE_REVISION,
        )
    ).resolve()
    if code_snapshot.name != JINA_REMOTE_CODE_REVISION:
        raise ValueError(
            "Jina implementation snapshot does not resolve to its pinned revision: "
            f"expected {JINA_REMOTE_CODE_REVISION}, got {code_snapshot.name}"
        )
    remote_code_hashes = hash_remote_modeling_code(code_snapshot)
    if not remote_code_hashes:
        raise ValueError("pinned Jina snapshot contains no remote Python modeling code to attest")
    config = json.loads((model_snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("emb_pooler") != "mean":
        raise ValueError("pinned Jina model does not declare mean pooling")
    package = _build_local_jina_package(model_snapshot, code_snapshot, config)
    return {
        "snapshot_path": str(model_snapshot),
        "local_package_path": str(package),
        "model_id": CANDIDATE_SPEC.model_id,
        "revision": CANDIDATE_SPEC.revision,
        "remote_code_repository": JINA_REMOTE_CODE_REPOSITORY,
        "remote_code_revision": JINA_REMOTE_CODE_REVISION,
        "trust_remote_code": True,
        "local_files_only": True,
        "pooling": config["emb_pooler"],
        "compatibility_patch": {
            "files": ["configuration_bert.py", "modeling_bert.py"],
            "reason": "Transformers 5 removes deprecated exports, implicit encoder defaults, and meta-device alibi init used by Jina's unused ONNX/pruning helpers",
            "effect_on_inference": "none; no model weights, tokenizer, pooling, or retrieval setting changed",
        },
        "remote_code_sha256": remote_code_hashes,
    }


def hash_remote_modeling_code(snapshot: Path) -> dict[str, str]:
    """Hash every shipped Python file that can be executed by remote model loading."""
    return {
        str(path.relative_to(snapshot)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(snapshot.rglob("*.py"))
    }


def _build_local_jina_package(model_snapshot: Path, code_snapshot: Path, config: dict[str, Any]) -> Path:
    """Vendor pre-attested code alongside the pinned weights without altering either snapshot."""
    package = Path(tempfile.mkdtemp(prefix="content_agent_exact73_jina_", dir="/private/tmp"))
    for source in model_snapshot.iterdir():
        if source.name != "config.json":
            (package / source.name).symlink_to(source)
    for source in code_snapshot.rglob("*.py"):
        destination = package / source.relative_to(code_snapshot)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    configuration = package / "configuration_bert.py"
    original_configuration = configuration.read_text(encoding="utf-8")
    expected_import = "from transformers.onnx import OnnxConfig"
    if expected_import not in original_configuration:
        raise ValueError("unexpected Jina configuration module; cannot apply narrow compatibility patch")
    configuration.write_text(
        original_configuration.replace(
            expected_import,
            "try:\n"
            "    from transformers.onnx import OnnxConfig\n"
            "except ModuleNotFoundError:\n"
            "    class OnnxConfig:  # Compatibility shim; Jina inference never instantiates this export helper.\n"
            "        pass",
        ),
        encoding="utf-8",
    )
    localized_configuration = configuration.read_text(encoding="utf-8")
    expected_config_init = "        super().__init__(pad_token_id=pad_token_id, **kwargs)"
    if expected_config_init not in localized_configuration:
        raise ValueError("unexpected Jina configuration initializer; cannot set encoder default")
    configuration.write_text(
        localized_configuration.replace(
            expected_config_init,
            expected_config_init
            + "\n        self.is_decoder = False  # Transformers 5 no longer sets this legacy encoder default."
            + "\n        self.add_cross_attention = False  # Transformers 5 no longer sets this legacy encoder default.",
        ),
        encoding="utf-8",
    )
    modeling = package / "modeling_bert.py"
    original_modeling = modeling.read_text(encoding="utf-8")
    expected_pytorch_import = (
        "from transformers.pytorch_utils import (\n"
        "    apply_chunking_to_forward,\n"
        "    find_pruneable_heads_and_indices,\n"
        "    prune_linear_layer,\n"
        ")"
    )
    if expected_pytorch_import not in original_modeling:
        raise ValueError("unexpected Jina modeling module; cannot apply narrow compatibility patch")
    expected_gradient_checkpoint = (
        "    def _set_gradient_checkpointing(self, module, value=False):\n"
        "        if isinstance(module, JinaBertEncoder):\n"
        "            module.gradient_checkpointing = value\n"
    )
    attention_mask_shims = (
        "\n\n    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):\n"
        "        # Transformers 5 compatibility shim; Jina inference passes head_mask=None.\n"
        "        if head_mask is not None:\n"
        "            head_mask = head_mask.to(dtype=self.dtype, device=self.device)\n"
        "            if head_mask.dim() == 1:\n"
        "                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)\n"
        "                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)\n"
        "            elif head_mask.dim() == 2:\n"
        "                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)\n"
        "        else:\n"
        "            head_mask = [None] * num_hidden_layers\n"
        "        return head_mask\n"
        "\n"
        "    def invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:\n"
        "        # Transformers 5 compatibility shim for unused decoder cross-attention path.\n"
        "        if encoder_attention_mask.dim() == 3:\n"
        "            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]\n"
        "        else:\n"
        "            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]\n"
        "        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)\n"
        "        return (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min\n"
    )
    patched_modeling = original_modeling.replace(
        expected_pytorch_import,
        "from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer\n"
        "try:\n"
        "    from transformers.pytorch_utils import find_pruneable_heads_and_indices\n"
        "except ImportError:\n"
        "    def find_pruneable_heads_and_indices(*args, **kwargs):\n"
        "        raise RuntimeError('head pruning is unsupported by the Transformers 5 compatibility shim')",
    )
    if "    def rebuild_alibi_tensor(\n" in patched_modeling:
        patched_modeling = patched_modeling.replace(
            "    def rebuild_alibi_tensor(\n"
            "        self, size: int, device: Optional[Union[torch.device, str]] = None\n"
            "    ):\n"
            "        # Alibi",
            "    def rebuild_alibi_tensor(\n"
            "        self, size: int, device: Optional[Union[torch.device, str]] = None\n"
            "    ):\n"
            "        if device is None:\n"
            "            device = \"cpu\"  # Transformers 5 may initialize modules on meta; build alibi on CPU.\n"
            "        # Alibi",
        )
    if expected_gradient_checkpoint in patched_modeling:
        patched_modeling = patched_modeling.replace(
            expected_gradient_checkpoint,
            expected_gradient_checkpoint + attention_mask_shims,
        )
    modeling.write_text(patched_modeling, encoding="utf-8")
    localized_auto_map = {
        key: value.split("--", maxsplit=1)[-1]
        for key, value in config.get("auto_map", {}).items()
    }
    if not localized_auto_map:
        raise ValueError("pinned Jina model does not declare its custom code mapping")
    (package / "config.json").write_text(
        json.dumps({**config, "auto_map": localized_auto_map}, indent=2) + "\n",
        encoding="utf-8",
    )
    return package


def load_encoder(
    spec: EncoderSpec,
    *,
    candidate_snapshot: Path | None = None,
    device: str | None = None,
) -> SentenceTransformer:
    """Load MiniLM or the pre-attested, pinned Jina snapshot."""
    kwargs: dict[str, Any] = {}
    model_target: str | Path = spec.model_id
    if spec is CANDIDATE_SPEC:
        if candidate_snapshot is None:
            raise ValueError("Jina must be loaded from a pre-attested pinned local snapshot")
        model_target = candidate_snapshot
        kwargs["trust_remote_code"] = True
        kwargs["local_files_only"] = True
    elif spec.revision is not None:
        kwargs["revision"] = spec.revision
    if device is not None:
        kwargs["device"] = device
    encoder = SentenceTransformer(str(model_target), **kwargs)
    if spec is CANDIDATE_SPEC:
        # Do not inherit a tokenizer's unlimited sentinel as an effective limit.
        encoder.max_seq_length = spec.max_sequence_length
        encoder.tokenizer.model_max_length = spec.max_sequence_length
    dimension = encoder.get_embedding_dimension()
    if dimension != spec.dimension:
        raise ValueError(f"{spec.name}: expected {spec.dimension}-D, got {dimension}-D")
    if spec is CANDIDATE_SPEC:
        commit = _runtime_model_commit(encoder)
        if commit and commit != spec.revision:
            raise ValueError(f"Jina revision mismatch: expected {spec.revision}, got {commit}")
        if encoder.max_seq_length != spec.max_sequence_length:
            raise ValueError("Jina effective max sequence length is not explicitly 8192")
        if encoder.tokenizer.model_max_length != spec.max_sequence_length:
            raise ValueError("Jina tokenizer max sequence length is not explicitly 8192")
    return encoder


def _runtime_model_commit(encoder: SentenceTransformer) -> str | None:
    first_module = next(iter(encoder._modules.values()), None)
    config = getattr(getattr(first_module, "auto_model", None), "config", None)
    return getattr(config, "_commit_hash", None)


def token_lengths_without_clipping(encoder: SentenceTransformer, texts: Iterable[str]) -> list[int]:
    """Measure exact inputs with truncation disabled, including special tokens."""
    return [
        len(encoder.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in texts
    ]


def validate_jina_input_limits(encoder: SentenceTransformer, chunks: list[FrozenChunk]) -> dict[str, int]:
    """Require every frozen chunk and evaluator query to fit Jina's explicit limit."""
    chunk_lengths = token_lengths_without_clipping(encoder, (chunk.text for chunk in chunks))
    query_lengths = token_lengths_without_clipping(encoder, (item["query"] for item in GOLDEN_SET))
    if any(length > CANDIDATE_SPEC.max_sequence_length for length in chunk_lengths):
        raise ValueError("at least one frozen chunk exceeds Jina's explicit 8192-token limit")
    if any(length > CANDIDATE_SPEC.max_sequence_length for length in query_lengths):
        raise ValueError("at least one frozen evaluator query exceeds Jina's explicit 8192-token limit")
    return {
        "max_chunk_tokens": max(chunk_lengths),
        "max_query_tokens": max(query_lengths),
        "chunks_over_limit": sum(length > CANDIDATE_SPEC.max_sequence_length for length in chunk_lengths),
        "queries_over_limit": sum(length > CANDIDATE_SPEC.max_sequence_length for length in query_lengths),
    }


def collection_name(spec: EncoderSpec) -> str:
    return f"exact73_{spec.name.replace('-', '_')}_794851d"


def _point_id(chunk: FrozenChunk, spec: EncoderSpec) -> str:
    return str(uuid5(NAMESPACE_URL, f"exact73:{spec.name}:{chunk.source}:{chunk.chunk_index}"))


def ingest_arm(
    client: QdrantClient, encoder: SentenceTransformer, spec: EncoderSpec, chunks: list[FrozenChunk]
) -> dict[str, Any]:
    """Create and validate a disposable arm collection without production access."""
    name = collection_name(spec)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=spec.dimension, distance=Distance.COSINE),
    )
    started = time.perf_counter()
    vectors = np.asarray(encoder.encode([chunk.text for chunk in chunks], show_progress_bar=False))
    corpus_embedding_seconds = time.perf_counter() - started
    if vectors.shape != (len(chunks), spec.dimension):
        raise ValueError(f"{spec.name}: unexpected vector shape {vectors.shape}")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError(f"{spec.name}: vectors must all be finite and nonzero")
    client.upsert(
        collection_name=name,
        wait=True,
        points=[
            PointStruct(
                id=_point_id(chunk, spec),
                vector=vector.tolist(),
                payload={"source": chunk.source, "chunk_index": chunk.chunk_index, "text": chunk.text},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
    )
    return {
        **validate_arm_collection(client, spec, chunks),
        "corpus_embedding_seconds": corpus_embedding_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
    }


def peak_rss_bytes() -> int:
    """Normalize ru_maxrss, whose units differ between macOS and Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def warm_query_p95_ms(encoder: SentenceTransformer, query: str) -> float:
    """Measure a warmed query encode with no model-side prompt transformation."""
    encoder.encode(query, show_progress_bar=False)
    samples: list[float] = []
    for _ in range(20):
        started = time.perf_counter()
        encoder.encode(query, show_progress_bar=False)
        samples.append((time.perf_counter() - started) * 1000)
    return float(statistics.quantiles(samples, n=20, method="inclusive")[18])


def local_feasibility_failures(collection: dict[str, Any], warm_p95_ms: float) -> list[str]:
    """Apply the fixed local Jina feasibility gate."""
    # Pinned Jina registers [1, n_heads, 8192, 8192] alibi buffers at init; ~6 GB
    # peak RSS on MPS/macOS is expected and is not an OOM failure for this fixture.
    checks = {
        "peak_rss_bytes": (collection["peak_rss_bytes"], 8 * 1024**3),
        "corpus_embedding_seconds": (collection["corpus_embedding_seconds"], 120.0),
        "warm_query_p95_ms": (warm_p95_ms, 500.0),
    }
    return [f"{name}={value:.6f} exceeds {limit:.6f}" for name, (value, limit) in checks.items() if value > limit]


def validate_arm_collection(
    client: QdrantClient, spec: EncoderSpec, chunks: list[FrozenChunk]
) -> dict[str, Any]:
    """Verify Qdrant schema, vectors, identities, and byte-exact payload round-trip."""
    name = collection_name(spec)
    info = client.get_collection(name)
    config = info.config.params.vectors
    if isinstance(config, dict) or config.size != spec.dimension or config.distance != Distance.COSINE:
        raise ValueError(f"{spec.name}: collection schema does not match cosine/{spec.dimension}-D")
    points, offset = client.scroll(name, limit=len(chunks) + 1, with_payload=True, with_vectors=True)
    if offset is not None or len(points) != len(chunks):
        raise ValueError(f"{spec.name}: collection point count does not match frozen fixture")
    by_identity = {
        ((point.payload or {}).get("source"), (point.payload or {}).get("chunk_index")): point
        for point in points
    }
    if len(by_identity) != len(points):
        raise ValueError(f"{spec.name}: duplicate source/chunk_index identity after round-trip")
    expected = {chunk.identity: chunk.text for chunk in chunks}
    actual = {identity: (point.payload or {}).get("text") for identity, point in by_identity.items()}
    if actual != expected:
        raise ValueError(f"{spec.name}: Qdrant payload text differs from frozen fixture")
    vectors = np.asarray([point.vector for point in points])
    if vectors.shape != (len(chunks), spec.dimension) or not np.isfinite(vectors).all():
        raise ValueError(f"{spec.name}: invalid vectors after Qdrant round-trip")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError(f"{spec.name}: zero vector after Qdrant round-trip")
    return {
        "collection": name,
        "point_count": len(points),
        "source_count": len({chunk.source for chunk in chunks}),
        "dimension": spec.dimension,
        "distance": "cosine",
        "payload_round_trip": "byte-identical",
        "finite_nonzero_vectors": len(points),
    }


def _bm25_rankings(chunks: list[FrozenChunk], query: str, depth: int) -> list[FrozenChunk]:
    model = BM25Okapi([chunk.text.lower().split() for chunk in chunks])
    scores = model.get_scores(query.lower().split())
    # The historical score formula is unchanged.  This key only resolves exact
    # equal-score ties deterministically, as permitted by the experiment brief.
    ranked = sorted(
        zip(chunks, scores, strict=True), key=lambda item: (-float(item[1]), item[0].source, item[0].chunk_index)
    )
    return [chunk for chunk, score in ranked if score != 0.0][:depth]


def build_bm25_rankings(chunks: list[FrozenChunk], queries: Iterable[str], depth: int) -> dict[str, list[FrozenChunk]]:
    """Build one frozen BM25 ranking per query for shared use by both dense arms."""
    return {query: _bm25_rankings(chunks, query, depth) for query in queries}


def _dense_rankings(
    client: QdrantClient, encoder: SentenceTransformer, spec: EncoderSpec, query: str, depth: int
) -> list[dict[str, Any]]:
    query_vector = encoder.encode(query, show_progress_bar=False).tolist()
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name(spec), query=query_vector, limit=depth, with_payload=True
        )
        points = response.points if hasattr(response, "points") else response
    else:
        # qdrant-client versions pinned by the historical project provide the
        # same cosine search semantics under ``search``.
        points = client.search(
            collection_name=collection_name(spec), query_vector=query_vector, limit=depth, with_payload=True
        )
    return [
        {
            "source": (point.payload or {})["source"],
            "chunk_index": (point.payload or {})["chunk_index"],
            "text": (point.payload or {})["text"],
            "distance": 1.0 - float(point.score),
        }
        for point in points
    ]


def _rrf(
    dense: list[dict[str, Any]], bm25: list[FrozenChunk], depth: int
) -> list[dict[str, Any]]:
    """Historical RRF-60 with identity-safe, deterministic equal-score ordering."""
    dense_by_identity = {(row["source"], row["chunk_index"]): row for row in dense}
    bm25_by_identity = {chunk.identity: chunk for chunk in bm25}
    scores: dict[tuple[str, int], float] = {}
    for identity in dense_by_identity.keys() | bm25_by_identity.keys():
        score = 0.0
        if identity in dense_by_identity:
            score += 1.0 / (RRF_K + next(i for i, row in enumerate(dense) if (row["source"], row["chunk_index"]) == identity))
        if identity in bm25_by_identity:
            score += 1.0 / (RRF_K + next(i for i, chunk in enumerate(bm25) if chunk.identity == identity))
        scores[identity] = score
    output: list[dict[str, Any]] = []
    for identity, score in sorted(scores.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))[:depth]:
        dense_row = dense_by_identity.get(identity)
        chunk = bm25_by_identity.get(identity)
        output.append(
            {
                "source": identity[0],
                "chunk_index": identity[1],
                "text": dense_row["text"] if dense_row else chunk.text,
                "distance": dense_row["distance"] if dense_row else 0.0,
                "rrf_score": score,
            }
        )
    return output


def evaluate_arm(
    client: QdrantClient,
    encoder: SentenceTransformer,
    spec: EncoderSpec,
    bm25_rankings: dict[str, list[FrozenChunk]],
) -> dict[str, Any]:
    """Run the corrected free evaluator with the frozen query/label semantics."""
    max_depth = max(EVALUATION_DEPTHS) * 2
    in_domain = [item for item in GOLDEN_SET if item["expected_sources"]]
    per_query: list[dict[str, Any]] = []
    aggregate = {k: {"hit": [], "source_recall": [], "ndcg": [], "concept_coverage": [], "concept_pass": [], "unique_sources": [], "duplicate_slots": []} for k in EVALUATION_DEPTHS}
    reciprocal_ranks: list[float] = []
    for item in in_domain:
        dense = _dense_rankings(client, encoder, spec, item["query"], max_depth)
        bm25 = bm25_rankings[item["query"]]
        retrieved = _rrf(dense, bm25, max(EVALUATION_DEPTHS))
        sources = [row["source"] for row in retrieved]
        expected = set(item["expected_sources"])
        query_metrics: dict[str, Any] = {
            "query": item["query"],
            "difficulty": item["difficulty"],
            "expected_sources": item["expected_sources"],
            "retrieved": [{"rank": index, "source": row["source"], "chunk_index": row["chunk_index"]} for index, row in enumerate(retrieved, start=1)],
            "at": {},
        }
        for k in EVALUATION_DEPTHS:
            _, coverage = concept_coverage_at_k(retrieved, item["required_concepts"], k)
            unique, duplicates = source_diversity_at_k(sources, k)
            values = {
                "hit": hit_at_k(sources, expected, k),
                "source_recall": source_recall_at_k(sources, expected, k),
                "ndcg": ndcg_at_k(sources, expected, k),
                "concept_coverage": coverage,
                "concept_pass": float(coverage >= CONCEPT_PASS_THRESHOLD),
                "unique_sources": unique,
                "duplicate_slots": duplicates,
            }
            query_metrics["at"][str(k)] = values
            for name, value in values.items():
                aggregate[k][name].append(value)
        rr = reciprocal_rank(sources, expected, max(EVALUATION_DEPTHS))
        reciprocal_ranks.append(rr)
        query_metrics["mrr_component"] = rr
        per_query.append(query_metrics)
    metrics = {
        f"@{k}": {name: float(sum(values) / len(values)) for name, values in aggregate[k].items()}
        for k in EVALUATION_DEPTHS
    }
    metrics["mrr"] = float(sum(reciprocal_ranks) / len(reciprocal_ranks))
    return {"spec": asdict(spec), "metrics": metrics, "per_query": per_query, "query_count": len(in_domain)}


def baseline_acceptance_failures(metrics: dict[str, Any]) -> list[str]:
    """Return predeclared baseline reproduction failures, without threshold tuning."""
    return _aggregate_failures(metrics)


def candidate_acceptance_failures(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Apply fixed aggregate and per-query no-harm criteria to the GTE candidate."""
    failures = _aggregate_failures(candidate["metrics"])
    baseline_by_query = {item["query"]: item for item in baseline["per_query"]}
    for item in candidate["per_query"]:
        control = baseline_by_query[item["query"]]
        for depth in ("3", "5"):
            candidate_at = item["at"][depth]
            baseline_at = control["at"][depth]
            if candidate_at["source_recall"] < baseline_at["source_recall"]:
                failures.append(f"{item['query']!r}: source-recall loss at @{depth}")
            if baseline_at["concept_pass"] == 1.0 and candidate_at["concept_pass"] == 0.0:
                failures.append(f"{item['query']!r}: concept-pass 1→0 at @{depth}")
            if candidate_at["ndcg"] < baseline_at["ndcg"] - 0.02:
                failures.append(f"{item['query']!r}: nDCG decrease >0.02 at @{depth}")
    return failures


def _aggregate_failures(metrics: dict[str, Any]) -> list[str]:
    checks = {
        ("@3", "hit"): 1.0,
        ("@3", "source_recall"): 0.950,
        ("@3", "ndcg"): 0.946,
        ("@3", "concept_coverage"): 0.822,
        ("@3", "concept_pass"): 0.833,
        ("@5", "hit"): 1.0,
        ("@5", "source_recall"): 0.967,
        ("@5", "ndcg"): 0.954,
        ("@5", "concept_coverage"): 0.867,
        ("@5", "concept_pass"): 0.900,
        ("@1", "hit"): HIT_AT_1_MINIMUM,
        ("mrr", None): 0.978,
    }
    failures = []
    for (depth, metric), minimum in checks.items():
        value = metrics[depth] if depth == "mrr" else metrics[depth][metric]
        if value < minimum:
            label = depth if metric is None else f"{metric}{depth}"
            failures.append(f"{label}={value:.12f} < {minimum:.3f}")
    return failures
