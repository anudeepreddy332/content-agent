# PROJECT STATUS

_Last updated: 2026-06-16 — PRODUCTION FREEZE (v5-freeze)_

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
DEMO (P-demo) — interactive live front-end. CODE COMPLETE + TESTED, operator steps remain.
- Built: SSE streaming surface on the API (api/server.py, additive /ui/runs* + GET
  /ui/runs/{id}/events) + self-contained SPA (static/index.html) served at GET /. Shows each
  node live, both HITL gates inline, published-link panel. See DECISIONS 2026-06-17 (P-demo).
- Tested: tests/test_api_stream.py (3 tests); full suite 43 passed ($0); local boot smoke OK.
  Frozen poll API + all prior tests untouched; SQLite single-worker invariant preserved.
- REMAINING (operator, needs accounts/credentials + one paid run):
  1. Netlify: deploy ~/tmp/tmw-fork as a dedicated demo site (publish dir = repo root).
  2. Local rehearsal: launch with GIT_PUSH_ENABLED=true THEMACHINIST_REPO_PATH=<fork>
     API_BEARER_TOKEN=<tok>; run a topic end-to-end through both gates; `git push` the fork.
  3. AWS deploy of the validated container (docker-compose.prod.yml) for a public URL.
- Branch note: this work now lives on feature/demo-ui (off main, beabceb). The reverted P2.3
  toggle + analysis scripts moved to their own feature/p2.3-max-iterations branch. Tag v6-demo
  on merge.

## Current blocker
None. Highest-risk item: B4 durable HITL (Days 4–6); pre-registered fallback = CLI-in-container, decision end of Day 6.

## Repository state
- main: v6-p2.2-index-auto-update. Standing limitations unchanged (FREEZE.md):
  B4 registry volatility (now spans 2 pause points), 5-tag rollback window,
  local-merge-no-auto-push, single-worker throughput.

## Operating points (confirmed)
MAX_ITERATIONS=2, REFLECTION_THRESHOLD=7, GROUNDING_FLOOR=0.60, COST_GATE_USD=0.10.
SV primary / UVR<=0.15 gate. prompt_version sha-6687240c8cd8. HITL mandatory on BOTH gates.