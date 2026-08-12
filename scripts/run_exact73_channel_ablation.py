"""Run the exact-73 seven-arm retrieval-channel ablation (FREE, deterministic)."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exact73_channel_ablation import run_ablation  # noqa: E402

REQUIRED_RANKING_ARMS = (
    "bm25_only",
    "minilm_dense_only",
    "gte_dense_only",
    "jina_dense_only",
    "minilm_rrf",
    "gte_rrf",
    "jina_rrf",
)


def main() -> int:
    output_dir = PROJECT_ROOT / "outputs" / "exact73_channel_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_ablation(PROJECT_ROOT)

    repeat = run_ablation(PROJECT_ROOT)
    fps = result["ranking_fingerprints"]
    missing = [arm for arm in REQUIRED_RANKING_ARMS if arm not in fps]
    fp_match = (
        not missing
        and fps == repeat["ranking_fingerprints"]
        and set(fps) == set(REQUIRED_RANKING_ARMS)
    )
    metrics_match = all(
        result["arms"][arm]["metrics"] == repeat["arms"][arm]["metrics"]
        for arm in REQUIRED_RANKING_ARMS
    )
    result["repeatability"] = {
        "ranking_fingerprints_cover_all_seven_arms": not missing and len(fps) == 7,
        "ranking_fingerprints_match": fp_match,
        "metrics_match": metrics_match,
        "rankings_and_metrics_identical": fp_match and metrics_match,
        "missing_fingerprint_arms": missing,
    }

    result_path = output_dir / "result.json"
    payload = json.dumps(result, indent=2) + "\n"
    result_path.write_text(payload, encoding="utf-8")
    digest = sha256(payload.encode("utf-8")).hexdigest()
    (output_dir / "result.sha256").write_text(digest + "\n", encoding="utf-8")
    repeat_path = output_dir / "result_repeat.json"
    repeat_path.write_text(json.dumps(repeat, indent=2) + "\n", encoding="utf-8")
    print(result_path)
    print(digest)

    if result["fusion_reproduction_failures"]:
        print("fusion_reproduction_failures:", result["fusion_reproduction_failures"])
        return 2
    if not fp_match or not metrics_match:
        print("repeatability failed:", result["repeatability"])
        return 3
    print("rankings_and_metrics_identical=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
