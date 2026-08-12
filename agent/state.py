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
    category: str               # P2.2 Learning Log category: concept-exploration | project-deep-dives | field-notes

    # Draft
    draft_sections: DraftSections        # {problem_framing, technical_dive, code_snippets, takeaways}
    draft_markdown: str         # Full assembled draft

    # Retrieval
    web_sources: list           # [{title, url, content, score}]
    kb_results: list            # bounded evidence windows assembled from ranked child chunks
    kb_context_stats: dict      # candidate depth, source diversity, per-consumer context budgets

    # Verification
    grounding_report: list      # [{claim, source_url, confidence, status, specificity,
    #   source_kind, source_ref, kb_chunk_candidates?}]
    grounding_score: float      # mean confidence across all claims
    verification_status: Literal["not_started", "completed", "parse_failed", "skipped_cost_gate", "upstream_failed"]

    # Reflection
    reflection_score: int       # 1–10
    reflection_notes: str
    iterations: int             # max 2

    # HITL
    hitl_status: Literal["pending", "approved", "rejected", "feedback"]
    hitl_feedback: str | None
    html_review_status: Literal["approved", "rejected", "changes"] | None   # P2 post-render gate
    html_feedback: str | None   # P2: design/layout note for html_revise — NEVER content


    # Output
    html_output: str | None
    html_filename: str | None
    branch_name: str | None
    git_status: Literal["not_started", "pushed", "merged", "tagged_and_merged", "failed"] | None

    # Telemetry
    run_id: str
    prompt_version: str
    total_tokens: int
    total_cost_usd: float
    latency_ms: dict            # {draft, retrieve, verify, reflect, html_gen, git}

    # Per-iteration verify metrics (M4 instrumentation): one entry per verify pass.
    # Without this, iteration 1's grounding report is overwritten by iteration 2
    # and the revise loop's effect is invisible in telemetry.
    iteration_metrics: list

    # M4: count of unverified claims injected into the current draft call
    # (0 on iteration 1 and in the control arm). Records injection REALITY,
    # not intent — guard against the M2 interpolation-bug class.
    m4_feedback_claims: int

    # Error log
    error_log: list[str]
