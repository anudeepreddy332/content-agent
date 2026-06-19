# content-agent — PRODUCTION FREEZE (v5-freeze, 2026-06-16)

## What is frozen
Single-agent LangGraph content pipeline: retrieve → draft → verify → reflect →
[revise ≤2] → HITL → html_gen → git, deployed as an authenticated FastAPI service in a
non-root container alongside a network-isolated Qdrant. Code is frozen at tag
v5-b6b8-complete; this commit (v5-freeze) is documentation only.

## Deployed
- Single VM, docker compose -f docker-compose.prod.yml. API on 127.0.0.1:8000 (SSH tunnel).
- Live publish target: themachinist.org via git_node LOCAL merge + supervised `git push`.
- Prompt baseline hash: sha-6687240c8cd8 (metrics comparable only at this hash).

## How to operate
- Deploy/recover: docs/deploy/DEPLOY.md, docs/deploy/RECOVERY.md.
- Publish: HITL approval, then a human `git push origin main`. The agent CANNOT push on its own.
- Rollback: `scripts/rollback_publish.sh <slug>` + supervised push.

## Four standing limitations (first-class, not hidden)
1. B4 registry volatility — paused runs return 404 after an app restart; the SqliteSaver
   checkpoint survives on the outputs volume but the in-memory registry is not rehydrated.
   Drain awaiting_review runs before any restart/deploy.
2. Rollback window (tag-based) — git_node prunes to the newest 5 `v-` tags. The revert-based
   rollback in rollback_publish.sh does not depend on the tag and is unbounded.
3. Publish posture — git_node does a LOCAL merge only; every live publish needs a human push.
   Intentional safeguard, not a gap.
4. API throughput — single-worker executor (max_workers=1); runs queue during compute. Fine
   for one-article-at-a-time editorial use; vertical scaling is post-freeze.

## Resume-from-here
Sources of truth: agent.md, DECISIONS.md, milestone HANDOFF files. PROJECT_STATUS.md reflects
FROZEN state. Deferred work (NEW phases): B9 autonomy, tracing, M5b chunk tagging, Docling
ingest, literal B6 staging branch, OpenClaw WhatsApp trigger. Do not reopen locked decisions
without new evidence (DECISIONS.md).