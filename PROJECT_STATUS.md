# PROJECT STATUS

_Last updated: 2026-06-17 — POST-FREEZE P2 series closed_

## Current Phase
POST-FREEZE phase P2 — COMPLETE. P2.1 (content-frozen HTML HITL gate + html_revise),
P2.2 (index.html Learning Log auto-update), P2.3 (MAX_ITERATIONS experiment — REJECT, stays 2)
all closed. main at v6-p2.2-index-auto-update; P2.3 produced no code change (experiment reverted).

## Bird's-eye view
retrieve → draft → verify → reflect → [revise ≤2] → HITL → html_gen → git. Deployed as a
non-root container (FastAPI + SqliteSaver durable HITL, bearer auth) with network-isolated
Qdrant, single-VM compose. SV primary / UVR≤0.15 gate / prompt hash sha-6687240c8cd8. Publish
= HITL + local merge + supervised human push; rollback = one documented command.

## Completed   (append)
- Grounding arc M1-M5, M6a (prompt hashing). B1-B8 (DECISIONS 2026-06-12 … 2026-06-16).

## Currently active
None on this branch. P2 series is closed (see DECISIONS 2026-06-17, P2.2 + P2.3 entries). This
branch (feature/p2.3-max-iterations) carries only the reverted experiment toggle (commit
d301ed2) and the analysis scripts kept for the record. Next workstream is the demo front-end,
which lives on its own branch: feature/demo-ui (see that branch's PROJECT_STATUS.md).

## Current blocker
None.

## Repository state
- main: v6-p2.2-index-auto-update.
- This branch: code revert only (d301ed2, MAX_ITERATIONS back to 2) + scripts/p2_3_analyze.py,
  scripts/p2_3_reach.py kept for the record. No production code change.
- Standing limitations unchanged (FREEZE.md): B4 registry volatility (spans 2 pause points),
  5-tag rollback window, local-merge-no-auto-push, single-worker throughput.

## Operating points (confirmed)
MAX_ITERATIONS=2, REFLECTION_THRESHOLD=7, GROUNDING_FLOOR=0.60, COST_GATE_USD=0.10.
SV primary / UVR<=0.15 gate. prompt_version sha-6687240c8cd8. HITL mandatory on BOTH gates.
