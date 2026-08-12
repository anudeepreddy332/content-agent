"""Run the exact-73 unique-membership crossover diagnostic ($0, stored rankings)."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exact73_unique_membership_diagnostic import (  # noqa: E402
    run_unique_membership_diagnostic,
)


def main() -> int:
    output_dir = PROJECT_ROOT / "outputs" / "exact73_unique_membership"
    output_dir.mkdir(parents=True, exist_ok=True)
    first = run_unique_membership_diagnostic(PROJECT_ROOT)
    second = run_unique_membership_diagnostic(PROJECT_ROOT)

    rankings_match = first.get("summary_fingerprint") == second.get("summary_fingerprint")
    metrics_match = first.get("crossover_aggregates") == second.get("crossover_aggregates")
    classification_match = first.get("classification") == second.get("classification")
    first["repeatability"] = {
        "summary_fingerprint_identical": rankings_match,
        "metrics_identical": metrics_match,
        "classification_identical": classification_match,
        "rankings_and_metrics_identical": rankings_match
        and metrics_match
        and classification_match,
    }

    payload = json.dumps(first, indent=2) + "\n"
    result_path = output_dir / "result.json"
    result_path.write_text(payload, encoding="utf-8")
    digest = sha256(payload.encode("utf-8")).hexdigest()
    (output_dir / "result.sha256").write_text(digest + "\n", encoding="utf-8")
    print(result_path)
    print(digest)
    print("classification=", first.get("classification"))

    if str(first.get("classification", "")).startswith(
        "UNIQUE-MEMBERSHIP-DIAGNOSTIC-INVALID"
    ):
        return 2
    if not (rankings_match and metrics_match and classification_match):
        print("repeatability failed:", first["repeatability"])
        return 3
    print("rankings_and_metrics_identical=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
