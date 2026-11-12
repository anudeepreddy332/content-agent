"""Offline Candidate B representation only. Thin wrapper over agent.cswp."""

from __future__ import annotations

import importlib

from agent.cswp.constants import (
    ABC_CONTRACT as CONTRACT,
    BASELINE_MANIFEST as BASELINE,
    CHUNKER_VERSION,
    PARSER_NAME,
    PARSER_VERSION,
    ROOT,
    SERIALIZATION_VERSION,
)
from agent.cswp.errors import NetworkForbidden, StructuralError
from agent.cswp.identity import canonical_json, identity, sha
from agent.cswp.offline import install_offline_guard
from agent.cswp.source import Source
from agent.cswp.structural import (
    build_corpus,
    chunk_document,
    parse_document,
    serialize,
    validate_document,
)
from agent.cswp.tokenizer import MiniLMTokenizer

LIMIT = 254
ShadowError = StructuralError

__all__ = [
    "BASELINE",
    "CHUNKER_VERSION",
    "CONTRACT",
    "LIMIT",
    "MiniLMTokenizer",
    "NetworkForbidden",
    "PARSER_NAME",
    "PARSER_VERSION",
    "ROOT",
    "SERIALIZATION_VERSION",
    "ShadowError",
    "Source",
    "build_corpus",
    "canonical_json",
    "chunk_document",
    "identity",
    "importlib",
    "install_offline_guard",
    "parse_document",
    "serialize",
    "sha",
    "validate_document",
]


def main():
    from agent.cswp.structural import main as structural_main

    structural_main()


if __name__ == "__main__":
    main()
