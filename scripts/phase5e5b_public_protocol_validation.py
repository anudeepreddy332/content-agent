"""Public Q01/Q02 production orchestration with fresh Qdrant 1.9.2 only."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.cswp.constants import PRODUCTION_INDEX_DIR  # noqa: E402
from agent.kb_backend import qdrant_serving  # noqa: E402
from agent.shadow_qdrant.index import full_rebuild  # noqa: E402
from scripts import phase5e1_qualification_runner as runner  # noqa: E402
from scripts.phase5d4c_docker_acceptance import _docker_available, _docker_qdrant, _free_port, _parity_report  # noqa: E402
from scripts.phase5e3d1_real_adapter_integration import _public_development_fixture  # noqa: E402


def main() -> int:
    if not _docker_available():
        print("PUBLIC-PROTOCOL-QDRANT-BLOCKED: Docker unavailable")
        return 2
    with tempfile.TemporaryDirectory(prefix="phase5e5b-public-") as temporary:
        root = Path(temporary)
        root.chmod(0o700)
        index = root / "index"
        index.mkdir()
        for name in ("manifest.json", "units.jsonl"):
            shutil.copy2(PRODUCTION_INDEX_DIR / name, index / name)
        fixture_path = root / "public-q01-q02.json"
        _public_development_fixture(fixture_path)
        fixture = json.loads(fixture_path.read_text())
        fixture["holdout"] = True  # Mechanism eligibility; public oracle only.
        fixture_path.write_text(json.dumps(fixture))
        fixture_path.chmod(0o600)
        approval_path = root / "public-approval.json"
        approval_path.write_text(json.dumps({
            "schema_version": runner.RELEASE_APPROVAL_SCHEMA,
            "approved_execution_sha": runner.runtime_git_sha(),
            "approved_fixture_sha256": runner.sha256_file(fixture_path),
            "fixture_source": "operator_file", "fixture_path": str(fixture_path),
            "fixture_schema": fixture["schema_version"],
            "frozen_contract_identity": runner.EXPECTED_CONTRACT_SHA256,
        }))
        approval_path.chmod(0o600)
        # The public operator seam approves the exact working candidate bytes
        # before the local validation commit. It never changes real approval.
        approved_controls = {
            path: runner.sha256_file(path)
            for path in (Path(runner.__file__), runner.DEFAULT_CONTRACT_PATH, Path(__file__))
        }

        def public_control_binding(_approval):
            if any(runner.sha256_file(path) != digest for path, digest in approved_controls.items()):
                raise runner.QualificationHarnessError("public approved control bytes changed")

        archive = root / "attempt"
        port = _free_port()
        with _docker_qdrant(port, f"content-agent-5e5b-{port}") as (url, client, version):
            os.environ["QDRANT_URL"] = url
            qdrant_serving.clear_serving_cache()
            manifest = full_rebuild(client, index_dir=index)
            qdrant_serving.create_serving_alias(client, manifest["collection_name"])
            with patch.object(qdrant_serving, "_serving_client", lambda: client), patch.object(
                runner, "_fixed_operator_approval_path", lambda: approval_path
            ), patch.object(runner, "DEFAULT_ARCHIVE_ROOT", archive):
                qdrant_serving.clear_serving_cache()
                with patch.object(runner, "validate_approved_execution_controls", public_control_binding):
                    report = runner.run_authoritative_qualification()
                    verdict = runner.verify_authoritative_release(archive)
                records = [json.loads(line) for line in (archive / "execution_journal.jsonl").read_text().splitlines()]
                assert report["state"] == "PASS", report
                assert verdict == "PHASE-5E1A-HOLDOUT-PASS", verdict
                assert [row["query_id"] for row in records if row["event"] == "query_completed"] == ["Q01", "Q02", "Q01", "Q02"]
                assert len([row for row in records if row["event"] == "backend_invocation_intent"]) == 8
                assert not (archive / "oracle_fixture.json").exists()
                parity = _parity_report(client)
                assert parity["gating_count"] == 33
                assert parity["local_packed_recall"] == parity["qdrant_packed_recall"] == 0.93939394
                assert all(not parity[key] for key in ("seed_mismatches", "expanded_mismatches", "packed_id_mismatches", "packed_fingerprint_mismatches", "packed_regressions"))
                assert all(not rows for rows in parity["rank_parity_mismatches"].values())
                initial = json.loads((archive / "pre_retrieval_preflight.json").read_text())["evidence"]
                final = json.loads((archive / "final_live_preflight.json").read_text())["evidence"]
                assert initial["cswp_qdrant"]["point_count"] == final["cswp_qdrant"]["point_count"] == 159
                print(json.dumps({"qdrant_version": version, "point_count": 159, "mechanism": report, "completed_verdict": verdict, "public_queries": ["Q01", "Q02"], "parity": parity, "sealed_access": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
