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
    kb_results: list            # qualified: packed CSWP units + provenance; legacy: [{text, source, ...}]

    # Verification
    grounding_report: list      # [{claim, source_url, confidence, status, specificity,
    #   source_kind, source_ref, kb_chunk_candidates?, material, claim_type,
    #   section, anchor_quote, occurrences, requires_citation}]
    grounding_score: float      # DEPRECATED as routing authority (Phase 4 Slice 1):
    #   compatibility/observability only (API/telemetry/benchmark/UI). Derived from
    #   engine statuses (not LLM confidence); cannot determine semantic acceptance
    #   or revision routing.
    verification_status: Literal[
        "not_started",
        "completed",
        "parse_failed",
        "verification_error",
        "skipped_cost_gate",
        "upstream_failed",
        "inventory_failed",     # Phase 4 Slice 2A: Call-A inventory/anchor failure
    ]

    # Phase 4 Slice 2A: canonical claim inventory for the CURRENT draft version.
    # {schema_version, run_id, iteration, draft_sha256, brief_requirements,
    #  claims[{claim_id, claim_text, anchor_quote, occurrences, anchor_validity,
    #  section, claim_type, material, materiality_reason_code,
    #  materiality_rationale, materiality_override, requires_citation,
    #  satisfies_req_ids, specificity, call_b_eligible}], satisfied_req_ids,
    #  counts}. Rebuilt from scratch on every verify pass; never reused across
    #  draft versions. Completeness vs the true draft content remains UNKNOWN.
    claim_inventory: dict | None

    # Phase 4 Slice 2A/2b schema foundation: fixed brief-derived requirements,
    # external to any single draft's claims: [{req_id, kind, mandatory,
    # description}]. A mandatory req linked to a claim forces material=true
    # (deterministic override). Full coverage gating is Slice 2b.
    brief_requirements: list

    # Reflection
    reflection_score: int       # 1–10
    reflection_notes: str
    reflection_provenance: dict # judge output vs deterministic fallback
    iterations: int             # max 2

    # HITL
    hitl_status: Literal["pending", "approved", "rejected", "feedback"]
    hitl_feedback: str | None
    html_review_status: Literal["approved", "rejected", "changes"] | None   # P2 post-render gate
    html_feedback: str | None   # P2: design/layout note for html_revise — NEVER content


    # Output
    html_output: str | None
    html_filename: str | None
    article_body_html: str | None
    html_sha256: str | None
    html_policy_version: str | None
    approved_html_sha256: str | None
    git_commit_sha: str | None
    publish_expected_remote_sha: str | None
    publish_observed_remote_sha: str | None
    publish_status: str | None
    publish_error: str | None
    published_remote_sha: str | None
    published_live_url: str | None
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
    # Phase 4 Slice 1: blocker-bearing rows (weak/unverified) are injected via a
    # separate targeted block and recorded in
    # semantic_trace revision_linkage.semantic_obligations; this count stays
    # unverified-only for M4 telemetry comparability.
    m4_feedback_claims: int

    # Error log
    error_log: list[str]
    # Safe policy decision metadata. Never contains rejected model text.
    policy_diagnostics: list[dict]

    # P0-2b slice 2B: append-only reconstructable draft/verify evidence.
    # Persistence only; not an acceptance input.
    semantic_trace: dict
