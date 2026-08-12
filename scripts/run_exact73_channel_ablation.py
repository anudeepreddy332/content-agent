"""Run the exact-73 seven-arm retrieval-channel ablation (FREE, deterministic)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exact73_channel_ablation import run_ablation  # noqa: E402


def main() -> int:
    output_dir = PROJECT_ROOT / "outputs" / "exact73_channel_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_ablation(PROJECT_ROOT)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result_path)

    repeat = run_ablation(PROJECT_ROOT)
    repeat_path = output_dir / "result_repeat.json"
    repeat_path.write_text(json.dumps(repeat, indent=2) + "\n", encoding="utf-8")

    fp_match = result["ranking_fingerprints"] == repeat["ranking_fingerprints"]
    metrics_match = all(
        result["arms"][arm]["metrics"] == repeat["arms"][arm]["metrics"]
        for arm in result["arms"]
    )
    result["repeatability"] = {
        "ranking_fingerprints_match": fp_match,
        "metrics_match": metrics_match,
        "byte_identical": fp_match and metrics_match,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if result["fusion_reproduction_failures"]:
        print("fusion_reproduction_failures:", result["fusion_reproduction_failures"])
        return 2
    if not fp_match or not metrics_match:
        print("repeatability failed")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
