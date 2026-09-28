"""Check semantic archive verification with public development retrieval only.

The check runs public Q01/Q02 through the local and real Qdrant v1.9.2
adapters.  It then creates hash-valid copies whose determinism or second
execution evidence is missing, proving that semantic verification—not merely
the archive manifest digests—controls authoritative archive success.
"""

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
    archive_is_successful,
    archive_verification_failure,
    canonical_json_dumps,
    execute_qualified_backends,
    run_qualification,
    runtime_git_sha,
    sha256_file,
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


def _refresh_manifest(archive: Path) -> None:
    path = archive / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifacts"] = {
        item.name: {"sha256": sha256_file(item), "bytes": item.stat().st_size}
        for item in sorted(archive.iterdir())
        if item.is_file() and item.name != "manifest.json"
    }
    path.write_text(canonical_json_dumps(manifest) + "\n", encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json_dumps(value) + "\n", encoding="utf-8")


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
        archive = temporary_root / "complete-archive"
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

        failed_determinism = temporary_root / "failed-determinism"
        missing_second = temporary_root / "missing-second-execution"
        shutil.copytree(archive, failed_determinism)
        shutil.copytree(archive, missing_second)
        determinism = json.loads((failed_determinism / "determinism.json").read_text())
        determinism["passed"] = False
        _write_json(failed_determinism / "determinism.json", determinism)
        parity = json.loads((failed_determinism / "parity.json").read_text())
        parity["determinism"] = determinism
        _write_json(failed_determinism / "parity.json", parity)
        _refresh_manifest(failed_determinism)

        executions = json.loads((missing_second / "executions.json").read_text())
        _write_json(missing_second / "executions.json", executions[:1])
        per_query = json.loads((missing_second / "per_query.json").read_text())
        per_query["execution_2"] = None
        _write_json(missing_second / "per_query.json", per_query)
        second_parity = json.loads((missing_second / "parity.json").read_text())
        second_parity["execution_2"] = None
        _write_json(missing_second / "parity.json", second_parity)
        _write_json(missing_second / "execution_2_raw_backend_outputs.json", None)
        _refresh_manifest(missing_second)

        complete = (
            version == "1.9.2"
            and manifest["point_count"] == 159
            and report["overall_pass"] is True
            and archive_is_successful(archive)
            and archive_verification_failure(archive) is None
            and not archive_is_successful(failed_determinism)
            and archive_verification_failure(failed_determinism) == "determinism_not_passed"
            and not archive_is_successful(missing_second)
            and archive_verification_failure(missing_second) == "missing_or_failed_execution_evidence"
        )
        print(
            json.dumps(
                {
                    "qdrant_version": version,
                    "point_count": manifest["point_count"],
                    "complete_archive_semantically_verified": archive_is_successful(archive),
                    "failed_determinism_reason": archive_verification_failure(failed_determinism),
                    "missing_second_execution_reason": archive_verification_failure(missing_second),
                    "sealed_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
    print("PHASE-5E3D2C-REAL-ARCHIVE-INTEGRATION-PASS" if complete else "FAIL")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
