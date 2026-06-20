# PROJECT STATUS

_Last updated: 2026-06-20 — LangSmith default-on + token usage/cost attached (feature/langsmith-fixes)_

## Current Phase
POST-FREEZE phase P2 — COMPLETE. P2.1 (content-frozen HTML HITL gate + html_revise),
P2.2 (index.html Learning Log auto-update), P2.3 (MAX_ITERATIONS experiment — REJECT, stays 2)
all closed; P2.3 produced no code change (experiment reverted). P-demo (interactive SSE UI +
cloud publish) is COMPLETE and LIVE on EC2. A production-readiness audit and its do-now fixes
are merged to main (v7-audit-fixes). A repo cleanup (file reorg + docstring/dead-code pass) and
this docs-sync pass are on unmerged feature branches awaiting review.

## Bird's-eye view
retrieve → draft → verify → reflect → [revise ≤2] → HITL → html_gen → git. Deployed as a
non-root container (FastAPI + SqliteSaver durable HITL, bearer auth) with network-isolated
Qdrant, single-VM compose. SV primary / UVR≤0.15 gate / prompt hash sha-6687240c8cd8. Publish
= HITL + local merge + supervised human push; rollback = one documented command.

## Completed   (append)
- Grounding arc M1-M5, M6a (prompt hashing). B1-B8 (DECISIONS 2026-06-12 … 2026-06-16).
- P2.1 second HITL gate, P2.2 index.html auto-update, P2.3 MAX_ITERATIONS reject (2026-06-17).
- P-demo: SSE streaming UI (api/server.py additive /ui/runs* + static/index.html), cloud
  publish endpoint, live rehearsal passed (2026-06-18). See DECISIONS 2026-06-17/06-18.
- Production-readiness audit (docs/PRODUCTION_READINESS.md) + its 4 do-now fixes: PR-only
  grounding-regression CI gate, uptime-check runbook + ERROR-level publish-failure log,
  query_kb.py structlog migration, Docker Hub deploy path documented. Tagged v7-audit-fixes.
- **Cloud deploy LIVE (2026-06-19):** EC2 + Caddy (TLS reverse proxy) + Docker Hub
  (`anudeepreddy332/content-agent:demo`, built via `docker buildx --platform linux/arm64
  --push`, pulled on the box per docs/deploy/DEPLOY_DEMO.md's primary path) + sslip.io for TLS
  without real DNS. Live demo: https://54-221-24-43.sslip.io — the app container has no direct
  public port; Caddy is the only public surface (docker-compose.demo.yml `!reset` on `ports`).
- Repo cleanup (feature/cleanup, unmerged): 93 tracked files (down from ~140) — deleted 9
  tmp/ scratch scripts + 29 stale benchmark JSON dumps + 2 dead docs + 1 orphaned prompt;
  relocated root markdown into docs/ + docs/deploy/, retrieval-baseline evidence + kept-for-
  record scripts into docs/archive/ + scripts/archive/ + tools/archive/; fixed 5 docstring/
  behavior mismatches and removed 3 dead imports in scripts/+tools/. Docker build re-verified
  green after the reorg.
- Docs-sync, repo cleanup, and opt-in LangSmith tracing (LANGSMITH_TRACING=1 AND-gate) all
  merged to main.
- **LangSmith default-on + token usage/cost (2026-06-20, feature/langsmith-fixes, unmerged):**
  `setup_langsmith_tracing()` now enables on `LANGCHAIN_API_KEY`+`LANGCHAIN_PROJECT` alone (no
  flag needed); `LANGSMITH_TRACING="0"` force-off override added. `_get_client()` wraps the
  DeepSeek client with `langsmith.wrappers.wrap_openai()` when tracing is on; `_llm_call()`
  attaches DeepSeek's real per-token cost to each traced run's `usage_metadata` (LangSmith has
  no built-in `deepseek-chat` price entry). See DECISIONS 2026-06-20. Off-path verified
  zero-`langsmith`-import; full suite green (62 passed); smoke test passed with tracing off.

## Currently active
feature/langsmith-fixes (unmerged) is the tip of work — see Completed above and DECISIONS.md.

## Current blocker
None. Reminder carried from feature/cleanup: the Docker image was rebuilt and verified to
build locally after the reorg, but if any further code changes land before the image is
re-pushed to Docker Hub, re-push and re-pull-test on EC2 before trusting the live demo URL.

## Repository state
- main: v7-audit-fixes (production-readiness audit + do-now fixes + cheatsheet docs merged).
- feature/cleanup and feature/docs-sync: both unmerged, this status reflects their content.
- Current file tree (post-cleanup): see README.md's "Project structure" section for the full
  layout — docs/ (CASE_STUDY, cheatsheets, audit report), docs/deploy/ (DEPLOY*, RECOVERY),
  docs/archive/ (gate reports, retrieval-eval evidence), scripts/archive/ + tools/archive/
  (kept-for-record / blocked-dependency code). Dockerfile/compose/Caddyfile stayed at root.
- scripts/archive/p2_3_analyze.py, scripts/archive/p2_3_reach.py kept for the record (moved
  into scripts/archive/ during the feature/cleanup reorg); no production code change from P2.3.
- Standing limitations unchanged (FREEZE.md): B4 registry volatility (now spans 2 pause
  points), 5-tag rollback window, local-merge-no-auto-push, single-worker throughput.

## Operating points (confirmed)
MAX_ITERATIONS=2, REFLECTION_THRESHOLD=7, GROUNDING_FLOOR=0.60, COST_GATE_USD=0.10.
SV primary / UVR<=0.15 gate. prompt_version sha-6687240c8cd8. HITL mandatory on BOTH gates.
