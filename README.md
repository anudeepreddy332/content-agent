# content-agent

**A production-grade, single-agent LangGraph pipeline that researches a topic, drafts a
grounded HTML article, verifies every claim against its sources, and publishes it — with a
human approving both the content and the rendered layout before anything goes live.**

## 🎥 Live Demo

[**Watch the full end-to-end demo →**](https://www.youtube.com/watch?v=gJttMm90ugM)

A grounded, human-in-the-loop LLM pipeline: topic → hybrid retrieval → draft → claim-level
grounding verification → reflection → human approval → live publish.

📄 [Full case study](https://themachinist.org/content-agent)

---

## What this does

Takes a topic as input. Produces a publish-ready HTML article for themachinist.org,
pushed to a feature branch on the themachinist-website repo (or its demo fork).

Pipeline: Retrieve → Draft → Verify → Reflect → [revise ≤2] → HITL (content) → HTML Gen →
HITL (layout) → Git (local merge only — a human always does the actual `git push`).
Source-aware drafting (retrieve runs before draft, not after) was locked at M3 — see
`DECISIONS.md`, 2026-06-09.

Read `architecture.md` before touching any code.

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

# 4. Start Qdrant (Docker required)
docker-compose up -d

# 5. Seed the knowledge base
uv run python scripts/ingest.py --source kb/seed_docs/

# 6. Verify KB is populated
uv run python scripts/ingest.py --stats
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

# Interactive demo SPA + API, locally
uv run python main.py serve   # then open http://localhost:8000/
```

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
├── architecture.md          ← Contract. Read this first.
├── DECISIONS.md / PROJECT_STATUS.md / FREEZE.md   ← canonical living docs
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

Step labels are historical; see PROJECT_STATUS.md for the current milestone state (M1-M6,
B1-B9, P2, P-demo all complete) and agent.md for the full roadmap.
