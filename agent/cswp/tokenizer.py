"""Pinned MiniLM tokenizer backend for CSWP token accounting."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from agent.cswp.constants import BASELINE_MANIFEST, MINILM_TOKENIZER_FIXTURE, ROOT
from agent.cswp.errors import StructuralError
from agent.cswp.identity import identity, sha


class MiniLMTokenizer:
    """The frozen local tokenizer backend, with truncation and padding disabled."""

    def __init__(self, snapshot=None):
        from tokenizers import Tokenizer

        embedding = json.loads(BASELINE_MANIFEST.read_text())["embedding"]
        revision = embedding["local_resolved_revision"]
        snapshot = Path(snapshot) if snapshot else MINILM_TOKENIZER_FIXTURE
        for name, expected in embedding["tokenizer_file_sha256"].items():
            path = snapshot / name
            if not path.is_file() or sha(path.read_bytes()) != expected:
                raise StructuralError(f"frozen tokenizer unavailable or changed: {name}")
        self.backend = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        self.backend.no_truncation()
        self.backend.no_padding()
        self.tokenizer_id = identity("tokenizer", embedding["tokenizer_file_sha256"])
        self.manifest = {
            "model_id": embedding["model_id"],
            "model_revision": revision,
            "file_sha256": embedding["tokenizer_file_sha256"],
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "tokenizer_id": self.tokenizer_id,
            "truncation": False,
            "padding": False,
        }
        if self.count("hello", specials=True) - self.count("hello") != 2:
            raise StructuralError("expected exactly two MiniLM special tokens")

    def count(self, text, specials=False):
        return len(self.backend.encode(text, add_special_tokens=specials).ids)

    def offsets(self, text):
        return self.backend.encode(text, add_special_tokens=False).offsets
