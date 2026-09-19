"""Pinned local MiniLM encoder for production CSWP retrieval."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_CACHE_DIRNAME = "models--sentence-transformers--all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class EncoderError(RuntimeError):
    """Local MiniLM snapshot unavailable or invalid."""


def resolve_local_model_snapshot(explicit: Path | None = None) -> Path:
    import os

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        candidates.append(
            Path(hf_home)
            / "hub"
            / MODEL_CACHE_DIRNAME
            / "snapshots"
            / MODEL_REVISION
        )
    candidates.append(
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / MODEL_CACHE_DIRNAME
        / "snapshots"
        / MODEL_REVISION
    )
    for candidate in candidates:
        if (candidate / "model.safetensors").is_file() and (
            candidate / "tokenizer.json"
        ).is_file():
            return candidate.resolve()
    raise EncoderError(
        "frozen MiniLM snapshot is unavailable locally; refusing network resolution"
    )


@lru_cache(maxsize=1)
def get_encoder():
    from sentence_transformers import SentenceTransformer

    snapshot = resolve_local_model_snapshot()
    return SentenceTransformer(str(snapshot), local_files_only=True)
