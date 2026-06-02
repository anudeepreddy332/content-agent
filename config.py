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

# LLM behavior
DRAFT_TEMPERATURE = 0.3     # Low = factual consistency. Raise to 0.5 if drafts feel robotic.
REFLECT_TEMPERATURE = 0.1   # Very low = deterministic scoring. Do not raise above 0.2

# Pipeline gates
MAX_ITERATIONS = 2          # Max draft-revise loops before forcing HITL
REFLECTION_THRESHOLD = 7    # Reflection score below this triggers rewrite (soft signal)
GROUNDING_FLOOR = 0.60      # Grounding score below this forces rewrite (hard floor)
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

# Git integration
THEMACHINIST_REPO_PATH = os.getenv("THEMACHINIST_REPO_PATH", "/Users/anudeep/PycharmProjects/themachinist-website")
MAX_TAGS_TO_KEEP = 5        # Prune older tags beyond this count

# Reproducibility
PROMPT_VERSION = "v1.0"     # Increment when prompts/draft_system.md changes

