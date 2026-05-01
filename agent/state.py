"""
agent/state.py
--------------
Single source of truth for the content-agent state.
Every node reads from and writes to this TypedDict.
"""

from typing import TypedDict, Literal

class DraftSections(TypedDict):
    problem_framing: str
    technical_dive: str
    code_snippets: str
    takeaways: str


class AgentState(TypedDict):
    # Input
    topic: str                  # e.g. "Linear & Logistic Regression"
    slug: str                   # e.g. "linear-logistic-regression"
    series_context: str         # e.g. "Family 01 — Linear Models · supervised-learning-models.html"
    card_id: str                # e.g. "01-A"

    # Draft
    draft_sections: DraftSections        # {problem_framing, technical_dive, code_snippets, takeaways}
    draft_markdown: str         # Full assembled draft

    # Retrieval
    web_sources: list           # [{title, url, content, score}]
    kb_results: list            # [{text, source, distance}]

    # Verification
    grounding_report: list      # [{claim, source_url, confidence, status}]
    grounding_score: float      # mean confidence across all claims

    # Reflection
    reflection_score: int       # 1–10
    reflection_notes: str
    iterations: int             # max 2

    # HITL
    hitl_status: Literal["pending", "approved", "rejected", "feedback"]
    hitl_feedback: str | None

    # Output
    html_output: str | None
    html_filename: str | None
    branch_name: str | None
    git_status: Literal["not_started", "pushed", "merged", "tagged_and_merged", "failed"] | None

    # Telemetry
    run_id: str
    total_tokens: int
    total_cost_usd: float
    latency_ms: dict            # {draft, retrieve, verify, reflect, html_gen, git}
