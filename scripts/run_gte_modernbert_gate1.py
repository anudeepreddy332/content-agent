"""Run the bounded, local-only GTE ModernBERT Gate 1 experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

# `python scripts/run_...py` places scripts/ rather than the repository root on
# sys.path. Keep the documented direct invocation usable without packaging this
# bounded experiment as an installed application module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient

from experiments.gte_modernbert import (
    EXPERIMENT_COLLECTION,
    GTE_MODERNBERT_SPEC,
    canonical_documents,
    chunk_corpus,
    corpus_identity,
    ingest_chunks,
    load_encoder,
    validate_collection,
    validate_model_contract,
    warm_query_p95_ms,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default=EXPERIMENT_COLLECTION)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/gte_modernbert_gate1/manifest.json"),
    )
    args = parser.parse_args()

    documents = canonical_documents(PROJECT_ROOT / "kb" / "seed_docs")
    corpus_hash = corpus_identity(documents)
    encoder = load_encoder()
    chunks = chunk_corpus(documents, encoder.tokenizer)
    client = QdrantClient(url=args.qdrant_url)
    ingestion = ingest_chunks(client, encoder, chunks, corpus_hash, args.collection)
    collection = validate_collection(client, documents, len(chunks), corpus_hash, args.collection)
    manifest = {
        "embedding_spec": asdict(GTE_MODERNBERT_SPEC),
        "runtime_model": validate_model_contract(encoder),
        "chunk_contract": {
            "actual_chunk_count": len(chunks),
            "max_model_input_tokens": max(chunk.model_input_tokens for chunk in chunks),
            "empty_chunk_count": sum(not chunk.text.strip() for chunk in chunks),
        },
        "collection": collection,
        "local_performance": {
            **ingestion,
            "warm_query_p95_ms": warm_query_p95_ms(encoder),
        },
    }
    write_manifest(args.manifest, manifest)
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
