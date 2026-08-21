"""
Central configuration for content-agent.
Home for all tunable constants.

Changing REFLECTION_THRESHOLD: raises/lowers bar for self-revision.
Changing GROUNDING_FLOOR: raises/lowers mandatory rewrite threshold. 0.60 is the minimum
    acceptable grounding — below this, the draft has too many unsourced claims.
Changing MAX_ITERATIONS: controls max draft-revise loops. 2 is the sweet spot —
    more than 2 loops rarely improves quality, just burns tokens.
Changing COST_GATE_USD: hard abort if a single run exceeds this.
    $0.10 is generous for a DeepSeek run. Raise only if you deliberately add more nodes.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# DeepSeek pricing as of 2025 (USD per 1M tokens)
# Source: https://api-docs.deepseek.com/quick_start/pricing
DEEPSEEK_INPUT_COST_PER_M = 0.27    # prompt tokens
DEEPSEEK_OUTPUT_COST_PER_M = 1.10   # completion tokens


LLM_TIMEOUT_S = 120   # Per-request DeepSeek timeout. Draft calls are the longest
                      # (~30-60s at max_tokens 4000+); 120s kills only genuine hangs.
                      # APITimeoutError is in the tenacity retry set, so a hung
                      # call is killed and retried, not fatal.

# LLM behavior
DRAFT_TEMPERATURE = 0.3     # Low = factual consistency. Raise to 0.5 if drafts feel robotic.
REFLECT_TEMPERATURE = 0.1   # Very low = deterministic scoring. Do not raise above 0.2

# Pipeline gates
MAX_ITERATIONS = 2          # Max draft-revise loops before forcing HITL
REFLECTION_THRESHOLD = 7    # Reflection score below this triggers rewrite (soft signal)
GROUNDING_FLOOR = 0.60      # Grounding score below this forces rewrite (hard floor)
UVR_THRESHOLD = 0.15        # Unverified-claim rate gate. Value frozen; routing fail-closed.
COST_GATE_USD = 0.10            # Abort run if total cost exceeds this

# ChromaDB
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./kb/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "machinist_evergreen")
KB_N_RESULTS = 5            # How many KB chunks to retrieve per query

# Qdrant – step 4 migration target
# QDRANT_URL points to local docker during dev, Qdrant Cloud in production
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "machinist_evergreen")
# Embedding dimension for all-MiniLM-L6-v2 — must match the model used in save_to_kb
QDRANT_EMBEDDING_DIM = 384


# Tavily
TAVILY_MAX_RESULTS = 5      # Per query. 3 queries × 5 results = up to 15 sources (deduped)
TAVILY_MIN_AVG_SCORE = 0.5  # Force cache bypass if first-pass avg Tavily score is below this.
                            # Tavily scores: 0.9+ = highly relevant, 0.5-0.7 = moderate,
                            # <0.5 = low relevance. Cached results scoring below 0.5 indicate
                            # stale or off-topic sources that will produce poor grounding.
                            # Raise to 0.6 if stale-cache refreshes are too infrequent.
                            # Lower to 0.4 if legitimate low-scoring topics are refreshing unnecessarily.

# Git integration
THEMACHINIST_REPO_PATH = os.getenv("THEMACHINIST_REPO_PATH", "/Users/anudeep/PycharmProjects/themachinist-website")
MAX_TAGS_TO_KEEP = 5        # Prune older tags beyond this count

# Reproducibility (M6a)
# PROMPT_VERSION is a content hash over the four runtime prompt files, computed
# once at import time. Any edit to any of them changes the version automatically
# — no manual bump, no silent re-baselining. Motivation: M3 metric discontinuity
# (CatBoost blind UVR 0.06 vs 0.50 across verify-prompt versions).
# PROMPT_HASHES exposes per-file hashes so telemetry shows WHICH prompt changed:
# grounding/SV numbers are cross-comparable iff verify_system hashes match.
# Fails loud (FileNotFoundError) at import if a prompt file is missing —
# consistent with the startup-validation philosophy.
import hashlib
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent

_PROMPT_FILES = [
    "prompts/draft_system.md",
    "prompts/verify_system.md",
    "prompts/reflect_system.md",
    "prompts/html_template.md",
]

def _prompt_hash(rel_path: str) -> str:
    """12-hex-char SHA-256 prefix of a prompt file's raw bytes."""
    return hashlib.sha256((_PROJECT_ROOT / rel_path).read_bytes()).hexdigest()[:12]

PROMPT_HASHES = {_Path(p).stem: _prompt_hash(p) for p in _PROMPT_FILES}

PROMPT_VERSION = "sha-" + hashlib.sha256(
    "|".join(f"{k}:{v}" for k, v in sorted(PROMPT_HASHES.items())).encode("utf-8")
).hexdigest()[:12]

