"""Production Contiguous Structural Window Packing (CSWP) ingest compiler."""

from agent.cswp.compiler import (
    attach_neighbors,
    compile_and_write,
    compile_corpus,
    strip_neighbor_fields,
    validate_neighbors,
    write_index,
)
from agent.cswp.constants import (
    COMPILER_VERSION,
    CURRENT_CORPUS_REPRESENTATION_FINGERPRINT,
    PRODUCTION_INDEX_DIR,
)
from agent.cswp.errors import CSWPError, StructuralError
from agent.cswp.loader import load_production_index, production_index_available
from agent.cswp.packer import pack
from agent.cswp.structural import build_corpus
from agent.cswp.tokenizer import MiniLMTokenizer

__all__ = [
    "COMPILER_VERSION",
    "CSWPError",
    "CURRENT_CORPUS_REPRESENTATION_FINGERPRINT",
    "MiniLMTokenizer",
    "PRODUCTION_INDEX_DIR",
    "StructuralError",
    "attach_neighbors",
    "build_corpus",
    "compile_and_write",
    "compile_corpus",
    "load_production_index",
    "pack",
    "production_index_available",
    "strip_neighbor_fields",
    "validate_neighbors",
    "write_index",
]
