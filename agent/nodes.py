""""
agent/nodes.py
--------------
All node and edge functions for the content-agent pipeline.

Day 23 status:
    COMPLETE: draft_node, retrieve_node
    STUBBED:  verify_node, reflect_node, hitl_node, html_gen_node, git_node

Each node signature: (state: AgentState) -> dict
    Nodes return ONLY the keys they update.
    LangGraph merges the return dict into the existing state.
    Never return the full state — only your changes.

Vulnerabilities to watch:
    - draft_node: DeepSeek may return malformed JSON. Handled with try/except + fallback.
    - retrieve_node: Tavily may return 0 results on niche topics. KB may be empty.
      Both are handled — node degrades gracefully, does not crash.
    - draft_node loops back from hitl if feedback is given. On loop, iterations increments.
      Max iterations enforced in route_after_reflect, not here.
"""

import os
import json
import time
import uuid
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from agent.state import AgentState, DraftSections
from tools.web_search import web_search
from tools.query_kb import query_kb
from config import (
    DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL,
    DRAFT_TEMPERATURE,
    MAX_ITERATIONS,
    REFLECTION_THRESHOLD,
    GROUNDING_FLOOR,
    COST_GATE_USD,
    DEEPSEEK_INPUT_COST_PER_M,
    DEEPSEEK_OUTPUT_COST_PER_M,
)

# DeepSeek client
# Initialized once at module load. api_key read at call time (not cached at import)
# so that test_resilience.py invalid-key tests work correctly (Phase 3 lesson).

def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )

def _load_system_prompt() -> str:
    """Load draft system prompt from prompts/draft_system.md."""
    prompt_path = Path("prompts/draft_system.md")
    if not prompt_path:
        raise FileNotFoundError(f"Draft system prompt not found at {prompt_path}")

    # Strip the comment header lines (lines starting with #) before the actual prompt
    lines = prompt_path.read_text(encoding="utf-8").splitlines()
    content_lines = [l for l in lines if not l.startswith("#")]
    return "\n".join(content_lines).strip()

def _cost(usage) -> float:
    """Compute DeepSeek API cost from a usage object."""
    return (
        usage.prompt_tokens / 1_000_000 * DEEPSEEK_INPUT_COST_PER_M
        +
        usage.completion_tokens / 1_000_000 * DEEPSEEK_OUTPUT_COST_PER_M
    )



























