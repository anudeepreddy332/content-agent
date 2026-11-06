"""Provision and verify the pinned all-MiniLM-L6-v2 snapshot for CI/runtime.

Downloads the exact revision when network is available, verifies artifact
hashes, and refuses to resolve "latest". Safe to run repeatedly; exits 0 when
the pinned snapshot is present and identity-checked.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase5a0_baseline import (  # noqa: E402
    MODEL_CACHE_DIRNAME,
    MODEL_ID,
    MODEL_REVISION,
    Phase5A0Error,
    resolve_local_model_snapshot,
)

MODEL_SAFETENSORS_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
TOKENIZER_JSON_SHA256 = (
    "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
)
REQUIRED_FILES = ("model.safetensors", "tokenizer.json", "config.json")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def snapshot_dir() -> Path:
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / MODEL_CACHE_DIRNAME
        / "snapshots"
        / MODEL_REVISION
    )


def verify_snapshot(snapshot: Path) -> None:
    for name in REQUIRED_FILES:
        path = snapshot / name
        if not path.is_file():
            raise Phase5A0Error(f"missing required artifact: {name}")
    observed_weights = sha256_file(snapshot / "model.safetensors")
    if observed_weights != MODEL_SAFETENSORS_SHA256:
        raise Phase5A0Error(
            "model.safetensors hash drift: "
            f"observed={observed_weights} expected={MODEL_SAFETENSORS_SHA256}"
        )
    observed_tokenizer = sha256_file(snapshot / "tokenizer.json")
    if observed_tokenizer != TOKENIZER_JSON_SHA256:
        raise Phase5A0Error(
            "tokenizer.json hash drift: "
            f"observed={observed_tokenizer} expected={TOKENIZER_JSON_SHA256}"
        )
    if snapshot.name != MODEL_REVISION:
        raise Phase5A0Error(
            f"snapshot revision mismatch: observed={snapshot.name} "
            f"expected={MODEL_REVISION}"
        )


def download_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    cache_root = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=False,
    )
    return Path(cache_root)


def provision(*, verify_only: bool = False) -> Path:
    try:
        snapshot = resolve_local_model_snapshot()
        verify_snapshot(snapshot)
        return snapshot
    except Phase5A0Error:
        if verify_only:
            raise
    if verify_only:
        raise Phase5A0Error(
            "frozen MiniLM snapshot is unavailable locally; refusing network resolution"
        )
    snapshot = Path(download_snapshot())
    verify_snapshot(snapshot)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Require an already-provisioned snapshot; do not download.",
    )
    args = parser.parse_args()
    try:
        snapshot = provision(verify_only=args.verify_only)
    except Phase5A0Error as exc:
        print(f"PROVISION FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PROVISION OK: "
        f"{MODEL_ID}@{MODEL_REVISION} at {snapshot}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
