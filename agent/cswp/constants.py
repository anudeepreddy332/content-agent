"""Frozen paths and version pins for the production CSWP compiler."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_MANIFEST = ROOT / "reports/phase5/phase5a0/baseline_a_manifest.json"
ABC_CONTRACT = ROOT / "evals/fixtures/phase5a0_abc_contract.json"
CSWP_CONTRACT = ROOT / "evals/fixtures/phase5a2e_cswp_contract.json"
MINILM_TOKENIZER_FIXTURE = ROOT / "evals/fixtures/phase5a1_minilm_tokenizer"
PRODUCTION_INDEX_DIR = ROOT / "kb/indexes/cswp_v1"
UNITS_FILE = "units.jsonl"
MANIFEST_FILE = "manifest.json"
HISTORICAL_CSWP_MANIFEST = (
    ROOT / "reports/phase5/phase5a2e/candidate_cswp_manifest.json"
)

PARSER_NAME = "markdown-it-py"
PARSER_VERSION = "4.2.0"
CHUNKER_VERSION = "structural-source-spans-v1"
SERIALIZATION_VERSION = "title-breadcrumb-exact-child-v1"
PACKING_VERSION = "cswp-v1"
COMPILER_VERSION = "phase5d2-cswp-compiler-v1"
CANDIDATE = "CSWP"
CONTENT_TOKEN_LIMIT = 254
TOTAL_TOKEN_LIMIT = 256

# Current frozen 20-document seed corpus qualification fingerprint.
CURRENT_CORPUS_REPRESENTATION_FINGERPRINT = (
    "439ccdf81e2aff1bc0bf3448734cb71f774bc8d3e7619dd1e4b51b2685b79bb3"
)
