"""Offline re-score of immutable Call-A provider run under provider-eval oracle v2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.call_a_provider_eval_v2 import rescore_live_run_artifact  # noqa: E402

DEFAULT_RUN_ARTIFACT = (
    REPO_ROOT
    / "outputs/call_a_provider_qualification/runs/call_a_run_20260915T161944Z_1c1d81ac/run_artifact.json"
)
EXPECTED_SOURCE_DIGEST = "56873347a00a7473c491cea856de9e7188f6b0af86dbd7841157d33d600dfcf5"


def main() -> int:
    rescore = rescore_live_run_artifact(
        run_artifact_path=DEFAULT_RUN_ARTIFACT,
        expected_artifact_digest=EXPECTED_SOURCE_DIGEST,
    )
    printable = {
        k: v
        for k, v in rescore.items()
        if k not in {"case_results_v2"}
    }
    printable["case_dispositions_v2"] = [
        {"case_id": r["case_id"], "disposition": r.get("disposition"), "source_v1": r.get("source_disposition_v1")}
        for r in rescore["case_results_v2"]
    ]
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0 if rescore["bounded_qualification_v2"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
