"""Run the exact-73 embedding-only MiniLM vs. pinned Jina retrieval gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient  # noqa: E402

from experiments.exact73_jina_retrieval import (  # noqa: E402
    BASELINE_SPEC,
    CANDIDATE_SPEC,
    baseline_acceptance_failures,
    build_bm25_rankings,
    candidate_acceptance_failures,
    evaluate_arm,
    ingest_arm,
    local_feasibility_failures,
    load_encoder,
    load_exact73_fixture,
    prepare_jina_snapshot,
    validate_jina_input_limits,
    warm_query_p95_ms,
)
from scripts.retrieval_eval import GOLDEN_SET  # noqa: E402


def main() -> int:
    output = PROJECT_ROOT / "outputs" / "exact73_jina" / "result.json"
    chunks = load_exact73_fixture(PROJECT_ROOT / "kb" / "seed_docs")
    queries = [item["query"] for item in GOLDEN_SET]
    bm25 = build_bm25_rankings(chunks, queries, depth=10)
    # :memory: makes the collections disposable and physically independent of
    # any configured production Qdrant endpoint.
    client = QdrantClient(":memory:")

    baseline_encoder = load_encoder(BASELINE_SPEC)
    baseline_collection = ingest_arm(client, baseline_encoder, BASELINE_SPEC, chunks)
    baseline = evaluate_arm(client, baseline_encoder, BASELINE_SPEC, bm25)
    baseline_failures = baseline_acceptance_failures(baseline["metrics"])
    if baseline_failures:
        result = {"fixture": {"fingerprint": "verified"}, "baseline": baseline, "failures": baseline_failures}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(output)
        return 2

    candidate_provenance = prepare_jina_snapshot()
    candidate_encoder = load_encoder(
        CANDIDATE_SPEC, candidate_snapshot=Path(candidate_provenance["local_package_path"])
    )
    jina_limits = validate_jina_input_limits(candidate_encoder, chunks)
    candidate_collection = ingest_arm(client, candidate_encoder, CANDIDATE_SPEC, chunks)
    warm_p95_ms = warm_query_p95_ms(candidate_encoder, GOLDEN_SET[0]["query"])
    feasibility_failures = local_feasibility_failures(candidate_collection, warm_p95_ms)
    if feasibility_failures:
        result = {
            "fixture": {"fingerprint": "verified", "chunk_count": len(chunks), "source_count": 20},
            "baseline": baseline,
            "candidate_model_provenance": candidate_provenance,
            "candidate_collection": candidate_collection,
            "jina_input_limits": jina_limits,
            "warm_query_p95_ms": warm_p95_ms,
            "feasibility_failures": feasibility_failures,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(output)
        return 3
    candidate = evaluate_arm(client, candidate_encoder, CANDIDATE_SPEC, bm25)
    failures = candidate_acceptance_failures(baseline, candidate)
    result = {
        "fixture": {"fingerprint": "verified", "chunk_count": len(chunks), "source_count": 20},
        "baseline_collection": baseline_collection,
        "candidate_collection": candidate_collection,
        "candidate_model_provenance": candidate_provenance,
        "jina_input_limits": jina_limits,
        "warm_query_p95_ms": warm_p95_ms,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_failures": failures,
        "classification": "EXACT73-JINA-PASS" if not failures else "EXACT73-JINA-FAIL",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
