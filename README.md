# content-agent

**A supervised, single-operator LangGraph content pipeline that researches a topic, drafts a
grounded HTML article, verifies claims, and requires human review before publication.**

> **Current engineering status:** enterprise production is **BLOCKED**. The P0-1
> active-content/credential boundary is validated and integrated within its scoped contract;
> current deployment identity is not established. Phase-5 HOLDOUT-V2 retrieval qualification
> passed only at `e0cfe0b566134f848feb731a777f315eb1456dd2`. The reviewed application-only
> descendant `2a1d95f98b2b240750288924116f8a77b3492250` has green push CI, but the October-5
> real application E2E failed and was rejected at Gate 1. The corrected workflow contract is
> frozen; implementation and initial HTML content-equivalence guard are pending. External pilot
> is **NOT READY**. Read `PROJECT_STATUS.md` and `architecture.md` for current state and contract.

## 🎥 Live Demo

[**Watch the full end-to-end demo →**](https://www.youtube.com/watch?v=gJttMm90ugM)

A grounded, human-in-the-loop LLM pipeline: topic → hybrid retrieval → draft → claim-level
grounding verification → reflection → human approval → live publish.

📄 [Full case study](https://themachinist.org/content-agent)

---

## What this does

Takes a topic as input and supports supervised article drafting, verification, HTML review,
and guarded Git preparation for the website repository or demo fork. Successful end-to-end
completion is not guaranteed; the October-5 run produced a malformed draft, failed claim
inventory extraction, and never generated final article HTML or published.

Historical implemented topology: Retrieve → Draft → Verify → Reflect → bounded automatic
redraft → HITL (content) → HTML Gen →
HITL (layout) → Git (local merge only — a human always does the actual `git push`).
This topology reached Gate 1 with failed quality in the October-5 run. The accepted correction
requires valid draft → accepted verification → genuine current reflection score ≥7 before
Gate 1, with two drafts per quality episode and separately bounded content/layout feedback.
That workflow correction is **IMPLEMENTATION PENDING**, not an E2E PASS.
Source-aware drafting (retrieve runs before draft, not after) was locked at M3 — see
`DECISIONS.md`, 2026-06-09.

## Canonical engineering state

Read these in order before changing code:

| File | Authoritative role |
| --- | --- |
| `PROJECT_STATUS.md` | Concise current state, priority, blockers, and authorized mission. |
| `architecture.md` | Accepted architecture contract, including the frozen P0-1 implementation boundary. |
| `DECISIONS.md` | Append-only material decision history; superseded conclusions remain visible. |
| `docs/EXPERIMENT_LEDGER.md` | Compact index of meaningful experiments and release evidence. |
| `FREEZE.md` | Historical v5 freeze record; not current status. |

Repository evidence is authoritative. Agent or conversation memory is not. Update these files only
after a material validated state transition, not after every discussion.

---

## Setup

```bash
# 1. Clone and enter
git clone https://github.com/anudeepreddy332/content-agent
cd content-agent

# 2. Create virtual env and install deps
uv venv
source .venv/bin/activate
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env — add DEEPSEEK_API_KEY and TAVILY_API_KEY

# 4. (Optional) Start Qdrant for legacy ingest or cswp_qdrant serving
docker-compose up -d

# 5. (Optional) Seed the legacy Qdrant collection
uv run python scripts/ingest.py --source kb/seed_docs/
```

**Local inference without Qdrant:** the default `KB_BACKEND=cswp_local` serves the
production CSWP index from `kb/indexes/cswp_v1/` in memory. No Qdrant container is
required for pipeline runs, inference smoke, or pytest. Set `KB_BACKEND=cswp_qdrant`
only when a durable Qdrant instance has the `kb_cswp_serving` alias pointed at a
qualified CSWP collection (production compose sets this automatically).

```bash
# Provision pinned MiniLM and run zero-provider inference smoke (cswp_local default)
uv run python scripts/provision_minilm_snapshot.py
uv run python scripts/inference_smoke.py
```

---

## Run

```bash
# Draft an article (interactive HITL)
uv run python main.py run --topic "Gradient Descent"

# With series context (for supervised-learning-models.html cards)
uv run python main.py run \
  --topic "Linear & Logistic Regression" \
  --card-id "01-A" \
  --series "Family 01 — Linear Models · supervised-learning-models.html"

# Benchmark mode (auto-approve, no git push)
uv run python main.py run --topic "Gradient Descent" --auto

# Interactive demo SPA + API, locally (dry-run publish unless PUBLISH_TARGET=demo)
uv run python main.py serve --host 127.0.0.1   # then open http://127.0.0.1:8000/
```

A live client rehearsal that publishes an article must use `docs/CHEATSHEET_LOCAL.md`
(`PUBLISH_TARGET=demo`, fork clone only, never themachinist.org). Unset `PUBLISH_TARGET`
disables local merge and remote push.

For copy-paste command sequences beyond the basics above — running the full interactive
demo locally end-to-end, or deploying/operating the EC2 + Caddy + Docker Hub cloud demo — see:
- `docs/CHEATSHEET_LOCAL.md` — local server, SPA walkthrough, where telemetry/articles land.
- `docs/CHEATSHEET_AWS.md` — EC2 deploy, Docker Hub build/push, DNS via sslip.io, what
  persists across a reboot.

(`docs/deploy/DEPLOY.md` and `docs/deploy/DEPLOY_DEMO.md` are the full runbooks the cheat
sheets are distilled from, if you need the complete picture or are setting up from scratch.)

### Optional: LangSmith tracing

Off by default, additive to the existing structlog JSON logs. Set `LANGSMITH_TRACING=1` +
`LANGCHAIN_API_KEY` + `LANGCHAIN_PROJECT` (all three, or it stays off) to get cross-node trace
visualization in LangSmith. See `docs/LANGSMITH.md` for setup and exactly what it does/doesn't
change.

---

## Project structure

```
content-agent/
├── PROJECT_STATUS.md        ← Current state. Read this first.
├── architecture.md          ← Accepted architecture contract
├── DECISIONS.md             ← Append-only material decisions
├── FREEZE.md                ← Historical v5 freeze record
├── main.py                  ← CLI entry point
├── Dockerfile  Caddyfile  docker-compose.{yml,prod.yml,demo.yml}
├── agent/
│   ├── state.py             ← AgentState TypedDict
│   ├── graph.py             ← LangGraph state machine
│   └── nodes.py             ← All node implementations
├── api/
│   └── server.py            ← FastAPI HITL API + demo SSE/publish surface
├── static/
│   └── index.html           ← Self-contained demo SPA, served at GET /
├── tools/
│   ├── web_search.py        ← Tavily wrapper with 7-day file cache
│   ├── query_kb.py          ← Qdrant + BM25 hybrid retrieval with RRF
│   ├── save_to_kb.py        ← Qdrant ingest with chunking
│   └── archive/
│       └── document_ingest.py.archived  ← Docling multi-format parser, runtime-blocked (§13)
├── observability/
│   └── logger.py            ← structlog JSON logger
├── kb/
│   ├── seed_docs/           ← 20 enriched .md files (committed)
│   ├── qdrant_data/         ← Qdrant storage (gitignored)
│   └── chroma_db/           ← legacy ChromaDB store (gitignored)
├── prompts/
│   ├── draft_system.md      ← Draft node system prompt
│   ├── verify_system.md     ← Verify node system prompt
│   ├── reflect_system.md    ← Reflect node system prompt
│   ├── html_template.md     ← HTML generation template
│   └── html_revise_system.md ← Layout-only revision prompt (P2)
├── evals/
│   ├── verifier_golden_test.py  ← CI grounding-regression gate
│   ├── prompt_evals/        ← Prompt-level schema-stability checks
│   └── topics.json          ← 20 evaluation topics
├── scripts/
│   ├── smoke_test.py        ← Single end-to-end validation run
│   ├── benchmark.py         ← 20-topic benchmark harness
│   ├── retrieval_eval.py    ← Retrieval recall@k evaluation
│   ├── ingest.py            ← KB seed script (multi-format)
│   ├── rollback_publish.sh  ← Revert a published article's merge
│   └── archive/             ← Concluded one-off experiment analysis, kept for the record
├── tests/                   ← pytest suite, $0/mocked, runs in CI on every push
├── .github/workflows/       ← ci.yml (lint+test+eval-gate), eval.yml (manual full sweep)
├── docs/
│   ├── EXPERIMENT_LEDGER.md  ← compact canonical evidence index
│   ├── CASE_STUDY.md  CHEATSHEET_AWS.md  CHEATSHEET_LOCAL.md  PRODUCTION_READINESS.md
│   ├── deploy/           ← DEPLOY.md, DEPLOY_DEMO.md, RECOVERY.md
│   └── archive/          ← historical gate reports + retrieval-eval evidence
└── outputs/                  ← all gitignored runtime artifacts
    ├── runs/                ← Per-run telemetry JSON
    ├── articles/            ← Generated HTML articles (local archive)
    ├── checkpoints.sqlite   ← Durable HITL graph state (SqliteSaver)
    └── tavily_cache/        ← Tavily result cache
```

---

## Build steps

| Step | Status | Deliverable |
|------|--------|-------------|
| 1 — Logging + core nodes | ✓ complete | structlog, verify_node, reflect_node, smoke_test passing |
| 2 — Full pipeline | ✓ complete | html_gen_node, git_node, main.py CLI |
| 3 — Evals + benchmark | ✓ complete | Prompt evals, retrieval golden set (historical legacy hit@3 = 100%; corrected source metrics in DECISIONS.md), 5-round benchmark |
| 4 — Qdrant + Docling | ✓ complete | Qdrant migration, BM25+RRF, multi-format ingest (Docling later dropped, M6) |
| 5 — API + fault injection | ✓ complete | FastAPI durable HITL API (B4), 5 fault modes tested (B2) |
| 6 — Docker + AWS | ✓ complete | Containerized (B5), tested on EC2 + Caddy + Docker Hub (see demo video above) |

Step labels are historical and do not imply current enterprise-release approval. See
`PROJECT_STATUS.md` for the current priority, blockers, and authorized mission.
