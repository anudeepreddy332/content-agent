"""Run one public-development query through the actual qualification adapters.

This is intentionally a standalone Docker check: it starts a fresh Qdrant
v1.9.2 server, rebuilds only the frozen 159-unit CSWP collection, and never
reads a sealed holdout or calls a paid provider.
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
from agent.kb_backend.qdrant_serving import (  # noqa: E402
    clear_serving_cache,
    create_serving_alias,
)
from agent.shadow_qdrant.index import full_rebuild  # noqa: E402
from scripts.phase5d4c_docker_acceptance import (  # noqa: E402
    _docker_available,
    _docker_qdrant,
    _free_port,
)
from scripts.phase5e1_qualification_runner import (  # noqa: E402
    DEFAULT_CONTRACT_PATH,
    execute_qualified_backends,
    expected_runtime_identity,
    load_contract,
    run_development_evaluation,
    validate_real_adapter_preflight,
)

DEVELOPMENT_FIXTURE = ROOT / "evals/fixtures/retrieval_golden_v2.json"


def _public_development_fixture(path: Path) -> list[dict]:
    oracle = json.loads(DEVELOPMENT_FIXTURE.read_text(encoding="utf-8"))
    queries = [
        item for item in oracle["queries"] if item["query_id"] in {"Q01", "Q02"}
    ]
    if [item["query_id"] for item in queries] != ["Q01", "Q02"]:
        raise RuntimeError("public development Q01/Q02 fixture selection changed")
    _representation, units, _by_source = load_production_index()
    source_paths = {
        Path(unit["source_path"]).stem: unit["source_path"] for unit in units.values()
    }
    for query in queries:
        for relevant in query["relevant_sources"]:
            source_path = source_paths[relevant["source"]]
            source_text = (ROOT / source_path).read_text(encoding="utf-8")
            for evidence in relevant["evidence"]:
                quote = evidence["quote"]
                start = source_text.find(quote)
                if start < 0 or source_text.find(quote, start + 1) >= 0:
                    raise RuntimeError(
                        f"{query['query_id']}: public evidence quote has no unique canonical source interval"
                    )
                evidence["char_start"] = start
                evidence["char_end"] = start + len(quote)
    fixture = {"schema_version": "retrieval_golden_v2", "queries": queries}
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return queries


def main() -> int:
    if not _docker_available():
        print("PHASE-5E3D1-REAL-ADAPTER-INTEGRATION-BLOCKED: Docker unavailable")
        return 2

    with tempfile.TemporaryDirectory(prefix="phase5e3d1-") as temporary:
        root = Path(temporary)
        index_dir = root / "cswp_v1"
        index_dir.mkdir()
        for name in ("manifest.json", "units.jsonl"):
            shutil.copy2(PRODUCTION_INDEX_DIR / name, index_dir / name)
        fixture_path = root / "public-development-q01.json"
        queries = _public_development_fixture(fixture_path)

        port = _free_port()
        container_name = f"content-agent-5e3d1-{port}"
        with _docker_qdrant(port, container_name) as (url, client, version):
            os.environ["QDRANT_URL"] = url
            qdrant_serving._serving_client = lambda: client  # type: ignore[method-assign]
            clear_serving_cache()
            manifest = full_rebuild(client, index_dir=index_dir)
            create_serving_alias(client, manifest["collection_name"])
            clear_serving_cache()

            # This is deliberately called independently of the development
            # runner: release preflight remains a real, mandatory gate only
            # for the authoritative API, while this probe stays public/dev.
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            _canonical_map, preflight = validate_real_adapter_preflight(
                expected_runtime_identity(fixture, load_contract(DEFAULT_CONTRACT_PATH))
            )

            report = run_development_evaluation(
                fixture_path=fixture_path,
                contract_path=DEFAULT_CONTRACT_PATH,
                executor=execute_qualified_backends,
            )

    scores = report["per_query"]
    if (
        version != "1.9.2"
        or preflight["cswp_local"]["unit_count"] != 159
        or preflight["cswp_qdrant"]["point_count"] != 159
        or not report["backend_parity"]["passed"]
        or [score["query_id"] for score in scores] != ["Q01", "Q02"]
        or any(
            not score["coverage"]["retrieved"]
            or score["coverage"]["retrieved"] != score["coverage"]["expanded"]
            or score["coverage"]["retrieved"] != score["coverage"]["packed"]
            for score in scores
        )
    ):
        print(json.dumps(report, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "query_ids": [query["query_id"] for query in queries],
                "qdrant_version": version,
                "covered_spans": {score["query_id"]: score["coverage"] for score in scores},
                "preflight": preflight,
                "backend_parity": report["backend_parity"],
                "sealed_holdout_accessed": False,
            },
            sort_keys=True,
        )
    )
    print("PHASE-5E3D1-REAL-ADAPTER-INTEGRATION-PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
