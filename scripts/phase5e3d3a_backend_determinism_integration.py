"""Exercise complete two-run backend determinism using public Q01/Q02 only."""

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
    archive_is_successful,
    execute_qualified_backends,
    run_qualification,
    runtime_git_sha,
    sha256_file,
)
from scripts.phase5e3d2c_archive_verification_integration import (  # noqa: E402
    _canonical_development_fixture,
)


def main() -> int:
    if not _docker_available():
        print("PHASE-5E3D3A-REAL-DETERMINISM-INTEGRATION-NOT-RUN: Docker unavailable")
        return 2

    with tempfile.TemporaryDirectory(prefix="phase5e3d3a-") as temporary:
        temporary_root = Path(temporary)
        index_dir = temporary_root / "cswp_v1"
        index_dir.mkdir()
        for name in ("manifest.json", "units.jsonl"):
            shutil.copy2(PRODUCTION_INDEX_DIR / name, index_dir / name)
        fixture_path = temporary_root / "public-development-q01-q02.json"
        archive = temporary_root / "two-run-archive"
        _canonical_development_fixture(fixture_path)

        port = _free_port()
        container_name = f"content-agent-5e3d3a-{port}"
        with _docker_qdrant(port, container_name) as (url, client, version):
            os.environ["QDRANT_URL"] = url
            qdrant_serving._serving_client = lambda: client  # type: ignore[method-assign]
            clear_serving_cache()
            manifest = full_rebuild(client, index_dir=index_dir)
            create_serving_alias(client, manifest["collection_name"])
            clear_serving_cache()
            report = run_qualification(
                fixture_path=fixture_path,
                contract_path=DEFAULT_CONTRACT_PATH,
                archive_root=archive,
                write_archive=True,
                executor=execute_qualified_backends,
                authoritative=True,
                approved_execution_sha=runtime_git_sha(),
                approved_fixture_sha256=sha256_file(fixture_path),
                real_adapter_mode=True,
            )

        determinism = report["determinism"]
        complete = (
            version == "1.9.2"
            and manifest["point_count"] == 159
            and report["overall_pass"] is True
            and determinism["passed"] is True
            and determinism["mismatches"] == []
            and set(determinism["execution_1"]) == {"Q01", "Q02"}
            and determinism["execution_1"] == determinism["execution_2"]
            and all(
                set(snapshot) == set(BACKENDS)
                for snapshot in determinism["execution_1"].values()
            )
            and archive_is_successful(archive)
        )
        print(
            json.dumps(
                {
                    "qdrant_version": version,
                    "point_count": manifest["point_count"],
                    "determinism_passed": determinism["passed"],
                    "mismatch_count": len(determinism["mismatches"]),
                    "complete_archive_semantically_verified": archive_is_successful(archive),
                    "sealed_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
    print("PHASE-5E3D3A-REAL-DETERMINISM-INTEGRATION-PASS" if complete else "FAIL")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
