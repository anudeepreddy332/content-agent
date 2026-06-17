# PROJECT STATUS

_Last updated: 2026-06-16 — PRODUCTION FREEZE (v5-freeze)_

## Current Phase
POST-FREEZE phase P2 (HITL hardening) — in progress on feature/p2-html-hitl (off v5-freeze).
Frozen baseline (v5-freeze) remains the production reference.

## Bird's-eye view
retrieve → draft → verify → reflect → [revise ≤2] → HITL → html_gen → git. Deployed as a
non-root container (FastAPI + SqliteSaver durable HITL, bearer auth) with network-isolated
Qdrant, single-VM compose. SV primary / UVR≤0.15 gate / prompt hash sha-6687240c8cd8. Publish
= HITL + local merge + supervised human push; rollback = one documented command.

## Completed   (append)
- Grounding arc M1-M5, M6a (prompt hashing). B1-B8 (DECISIONS 2026-06-12 … 2026-06-16).

## Currently active
P2.1 — second HITL gate (content-frozen layout review). retrieve → draft → verify → reflect →
[revise ≤2] → hitl (CONTENT) → html_gen → hitl_html (LAYOUT) → [approve→git | request_changes→
html_revise→hitl_html | reject→END]. No production HITL bypass. Content-freeze enforced by a
word-multiset guard in html_revise_node. Next P2 items: (2) index.html Learning Log auto-update
in git_node; (3) MAX_ITERATIONS grounding experiment.

## Current blocker
None. Highest-risk item: B4 durable HITL (Days 4–6); pre-registered fallback = CLI-in-container, decision end of Day 6.

## Repository state
- main: v5-freeze. Active: feature/p2-html-hitl.
- New: html_revise_node, prompts/html_revise_system.md, state fields html_review_status +
  html_feedback. New env/cost: one LLM call per layout-revision request.
- Standing limitations unchanged (FREEZE.md). B4 registry volatility now spans two pause points.
