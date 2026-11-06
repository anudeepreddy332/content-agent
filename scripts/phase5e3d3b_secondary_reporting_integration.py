"""Validate all frozen secondary layers through public real-adapter scoring."""

from __future__ import annotations

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
from agent.kb_backend import qdrant_serving  # noqa: E402
from agent.kb_backend.qdrant_serving import clear_serving_cache, create_serving_alias  # noqa: E402
from agent.shadow_qdrant.index import full_rebuild  # noqa: E402
from scripts.phase5d4c_docker_acceptance import (  # noqa: E402
    _docker_available,
    _docker_qdrant,
    _free_port,
)
from scripts.phase5e1_qualification_runner import (  # noqa: E402
    BACKENDS,
    DEFAULT_CONTRACT_PATH,
    execute_qualified_backends,
    run_development_evaluation,
)
from scripts.phase5e3d2c_archive_verification_integration import (  # noqa: E402
    _canonical_development_fixture,
)


def _complete_source_matrix(score: dict) -> bool:
    expected_k = {"1", "3", "5"}
    for layer in ("retrieved", "expanded", "packed"):
        metrics = score["secondary_metrics"][layer]["source_rank_metrics"]
        if (
            set(metrics["source_recall_at"]) != expected_k
            or set(metrics["graded_ndcg_at"]) != expected_k
            or not isinstance(metrics["mrr_at_10"], float)
        ):
            return False
    return True


def main() -> int:
    if not _docker_available():
        print("PHASE-5E3D3B-REAL-SECONDARY-INTEGRATION-NOT-RUN: Docker unavailable")
        return 2

    with tempfile.TemporaryDirectory(prefix="phase5e3d3b-") as temporary:
        temporary_root = Path(temporary)
        index_dir = temporary_root / "cswp_v1"
        index_dir.mkdir()
        for name in ("manifest.json", "units.jsonl"):
            shutil.copy2(PRODUCTION_INDEX_DIR / name, index_dir / name)
        fixture_path = temporary_root / "public-development-q01-q02.json"
        _canonical_development_fixture(fixture_path)

        port = _free_port()
        container_name = f"content-agent-5e3d3b-{port}"
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

        determinism = report["determinism"]
        matrix_complete = all(
            _complete_source_matrix(snapshot[backend]["per_query_scoring"])
            for execution in ("execution_1", "execution_2")
            for snapshot in determinism[execution].values()
            for backend in BACKENDS
        )
        complete = (
            version == "1.9.2"
            and manifest["point_count"] == 159
            and report["overall_pass"] is True
            and report["decision_scope"] == "development"
            and determinism["passed"] is True
            and determinism["mismatches"] == []
            and matrix_complete
        )
        print(
            json.dumps(
                {
                    "qdrant_version": version,
                    "point_count": manifest["point_count"],
                    "secondary_matrix_complete": matrix_complete,
                    "determinism_passed": determinism["passed"],
                    "decision_scope": report["decision_scope"],
                    "sealed_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
    print("PHASE-5E3D3B-REAL-SECONDARY-INTEGRATION-PASS" if complete else "FAIL")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
