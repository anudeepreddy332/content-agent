"""Check archived-snapshot recomputation with public development retrieval.

The check runs public Q01/Q02 through the local and real Qdrant v1.9.2
adapters twice, rebuilds runtime-equivalent snapshots from the resulting raw
and structured evidence, and detects a Qdrant-only mutation.  It deliberately
uses development evaluation rather than manufacturing a holdout qualification.
Direct verifier-fixture tests cover authoritative archive acceptance.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cswp.constants import PRODUCTION_INDEX_DIR  # noqa: E402
from agent.cswp.loader import load_production_index  # noqa: E402
from agent.kb_backend import qdrant_serving  # noqa: E402
from agent.kb_backend.qdrant_serving import clear_serving_cache, create_serving_alias  # noqa: E402
from agent.shadow_qdrant.index import full_rebuild  # noqa: E402
from scripts.phase5d4c_docker_acceptance import (  # noqa: E402
    _docker_available,
    _docker_qdrant,
    _free_port,
)
from scripts.phase5e1_qualification_runner import (  # noqa: E402
    DEFAULT_CONTRACT_PATH,
    ArchiveEvidenceConflict,
    archived_deterministic_execution_snapshot,
    compare_deterministic_executions,
    execute_qualified_backends,
    run_development_evaluation,
)

DEVELOPMENT_FIXTURE = ROOT / "evals/fixtures/retrieval_golden_v2.json"


def _canonical_development_fixture(path: Path) -> None:
    oracle = json.loads(DEVELOPMENT_FIXTURE.read_text(encoding="utf-8"))
    selected = [item for item in oracle["queries"] if item["query_id"] in {"Q01", "Q02"}]
    _representation, units, _by_source = load_production_index()
    source_paths = {Path(unit["source_path"]).stem: unit["source_path"] for unit in units.values()}
    for query in selected:
        for relevant in query["relevant_sources"]:
            source_text = (ROOT / source_paths[relevant["source"]]).read_text(encoding="utf-8")
            for evidence in relevant["evidence"]:
                quote = evidence["quote"]
                start = source_text.find(quote)
                if start < 0 or source_text.find(quote, start + 1) >= 0:
                    raise RuntimeError(
                        f"{query['query_id']}: public evidence has no unique canonical interval"
                    )
                evidence["char_start"] = start
                evidence["char_end"] = start + len(quote)
    path.write_text(
        json.dumps({"schema_version": "retrieval_golden_v2", "queries": selected}),
        encoding="utf-8",
    )


def _recompute_development_determinism(report: dict) -> tuple[bool, bool]:
    """Rebuild archived-style snapshots without invoking release qualification."""

    query_ids = [row["query_id"] for row in report["per_query"]]
    first = archived_deterministic_execution_snapshot(
        report["per_query"], report["backend_per_query"],
        report["raw_backend_outputs"], query_ids,
    )
    second_report = report["execution_2"]
    second = archived_deterministic_execution_snapshot(
        second_report["per_query"], second_report["backend_per_query"],
        second_report["raw_backend_outputs"], query_ids,
    )
    mutated_raw = copy.deepcopy(second_report["raw_backend_outputs"])
    mutated_raw["cswp_qdrant"][0]["hybrid_top10"][0]["chunk_id"] += "__tampered"
    try:
        mutated = archived_deterministic_execution_snapshot(
            second_report["per_query"], second_report["backend_per_query"],
            mutated_raw, query_ids,
        )
    except ArchiveEvidenceConflict:
        mutation_detected = True
    else:
        mutation_detected = bool(compare_deterministic_executions(first, mutated))
    return (
        first == report["determinism"]["execution_1"]
        and second == report["determinism"]["execution_2"],
        mutation_detected,
    )


def main() -> int:
    if not _docker_available():
        print("PHASE-5E3D2C-REAL-ARCHIVE-INTEGRATION-NOT-RUN: Docker unavailable")
        return 2

    with tempfile.TemporaryDirectory(prefix="phase5e3d2c-") as temporary:
        temporary_root = Path(temporary)
        index_dir = temporary_root / "cswp_v1"
        index_dir.mkdir()
        for name in ("manifest.json", "units.jsonl"):
            shutil.copy2(PRODUCTION_INDEX_DIR / name, index_dir / name)
        fixture_path = temporary_root / "public-development-q01-q02.json"
        _canonical_development_fixture(fixture_path)

        port = _free_port()
        container_name = f"content-agent-5e3d2c-{port}"
        with _docker_qdrant(port, container_name) as (url, client, version):
            os.environ["QDRANT_URL"] = url
            qdrant_serving._serving_client = lambda: client  # type: ignore[method-assign]
            clear_serving_cache()
            manifest = full_rebuild(client, index_dir=index_dir)
            create_serving_alias(client, manifest["collection_name"])
            clear_serving_cache()
            report = run_development_evaluation(
                fixture_path=fixture_path,
                contract_path=DEFAULT_CONTRACT_PATH,
                executor=execute_qualified_backends,
                determinism_executor=execute_qualified_backends,
            )
        snapshots_match, qdrant_mutation_detected = _recompute_development_determinism(report)

        complete = (
            version == "1.9.2"
            and manifest["point_count"] == 159
            and report["overall_pass"] is True
            and report["decision_scope"] == "development"
            and report["determinism"]["passed"] is True
            and snapshots_match
            and qdrant_mutation_detected
        )
        print(
            json.dumps(
                {
                    "qdrant_version": version,
                    "point_count": manifest["point_count"],
                    "decision_scope": report["decision_scope"],
                    "snapshots_match_runtime_evidence": snapshots_match,
                    "qdrant_hybrid_mutation_detected": qdrant_mutation_detected,
                    "sealed_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
    print("PHASE-5E3D2C-REAL-ARCHIVE-INTEGRATION-PASS" if complete else "FAIL")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
