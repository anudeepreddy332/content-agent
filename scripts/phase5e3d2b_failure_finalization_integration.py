"""Exercise failure-safe finalization through both real retrieval adapters.

The check starts a fresh local Qdrant v1.9.2 server, rebuilds the frozen
159-unit collection, and runs only public development Q01/Q02.  It injects a
malformed ranking into the already-obtained Q02 adapter result; production
retrieval output and the sealed holdout are never changed or accessed.
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
    _run_evaluation,
    execute_qualified_backends,
    runtime_git_sha,
    sha256_file,
)

DEVELOPMENT_FIXTURE = ROOT / "evals/fixtures/retrieval_golden_v2.json"


def _canonical_development_fixture(path: Path) -> None:
    oracle = json.loads(DEVELOPMENT_FIXTURE.read_text(encoding="utf-8"))
    selected = [item for item in oracle["queries"] if item["query_id"] in {"Q01", "Q02"}]
    _representation, units, _by_source = load_production_index()
    source_paths = {
        Path(unit["source_path"]).stem: unit["source_path"] for unit in units.values()
    }
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


def main() -> int:
    if not _docker_available():
        print("PHASE-5E3D2B-REAL-FAILURE-INTEGRATION-NOT-RUN: Docker unavailable")
        return 2

    with tempfile.TemporaryDirectory(prefix="phase5e3d2b-") as temporary:
        temporary_root = Path(temporary)
        index_dir = temporary_root / "cswp_v1"
        index_dir.mkdir()
        for name in ("manifest.json", "units.jsonl"):
            shutil.copy2(PRODUCTION_INDEX_DIR / name, index_dir / name)
        fixture_path = temporary_root / "public-development-q01-q02.json"
        archive = temporary_root / "failure-archive"
        _canonical_development_fixture(fixture_path)

        port = _free_port()
        container_name = f"content-agent-5e3d2b-{port}"
        with _docker_qdrant(port, container_name) as (url, client, version):
            os.environ["QDRANT_URL"] = url
            qdrant_serving._serving_client = lambda: client  # type: ignore[method-assign]
            clear_serving_cache()
            manifest = full_rebuild(client, index_dir=index_dir)
            create_serving_alias(client, manifest["collection_name"])
            clear_serving_cache()

            def malformed_after_retrieval(query: str):
                outputs = execute_qualified_backends(query)
                if query == "query key value in self-attention":
                    outputs["cswp_qdrant"]["hybrid_top10"] = None
                return outputs

            # This is a failure-only regression over a temporary archive.  It
            # exercises internal finalization mechanics after the real adapters
            # have already returned malformed public-development data; it is
            # not an authoritative release entry point and cannot emit PASS.
            report = _run_evaluation(
                fixture_path=fixture_path,
                contract_path=DEFAULT_CONTRACT_PATH,
                archive_root=archive,
                write_archive=True,
                executor=malformed_after_retrieval,
                authoritative=True,
                approved_execution_sha=runtime_git_sha(),
                approved_fixture_sha256=sha256_file(fixture_path),
                real_adapter_mode=True,
            )

        execution = report["executions"][0]
        journal = [
            json.loads(line)
            for line in (archive / "execution_journal.jsonl").read_text().splitlines()
        ]
        complete = (
            version == "1.9.2"
            and manifest["point_count"] == 159
            and report["terminal_status"] == "INCOMPLETE"
            and report["overall_pass"] is False
            and execution["failure"]["query_id"] == "Q02"
            and execution["failure"]["stage"] == "canonicalization"
            and len(execution["per_query"]) == 1
            and len(execution["raw_backend_outputs"]["cswp_qdrant"]) == 2
            and execution["raw_backend_outputs"]["cswp_qdrant"][1]["hybrid_top10"] is None
            and journal[-1]["terminal_status"] == "INCOMPLETE"
        )
        print(
            json.dumps(
                {
                    "qdrant_version": version,
                    "point_count": manifest["point_count"],
                    "terminal_status": report["terminal_status"],
                    "disposition": report["disposition"],
                    "failure": execution["failure"],
                    "completed_query_count": len(execution["per_query"]),
                    "failing_raw_qdrant_retained": (
                        execution["raw_backend_outputs"]["cswp_qdrant"][1]["hybrid_top10"]
                        is None
                    ),
                    "sealed_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
    print("PHASE-5E3D2B-REAL-FAILURE-INTEGRATION-PASS" if complete else "FAIL")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
