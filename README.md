# content-agent

**Phase 4a — AI/Tech Article Drafting Pipeline**
Part of the multi-phase agentic AI engineering journey → themachinist.org

---

## What this does

Takes a topic as input. Produces a publish-ready HTML article for themachinist.org,
pushed to a feature branch on the themachinist-website repo.

Pipeline: Draft → Retrieve → Verify → Reflect → HITL → HTML Gen → Git push

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
```

---

## Project structure

```
content-agent/
├── architecture.md          ← Contract. Read this first.
├── main.py                  ← CLI entry point
├── docker-compose.yml       ← Qdrant local dev service
├── agent/
│   ├── state.py             ← AgentState TypedDict
│   ├── graph.py             ← LangGraph state machine
│   └── nodes.py             ← All node implementations
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
│   └── qdrant_data/         ← Qdrant storage (gitignored)
├── prompts/
│   ├── draft_system.md      ← Draft node system prompt
│   ├── verify_system.md     ← Verify node system prompt
│   ├── reflect_system.md    ← Reflect node system prompt
│   └── html_template.md     ← HTML generation template
├── evals/
│   ├── prompt_evals/        ← Prompt-level regression tests
│   └── topics.json          ← 20 evaluation topics
├── scripts/
│   ├── smoke_test.py        ← Single end-to-end validation run
│   ├── benchmark.py         ← 20-topic benchmark harness
│   ├── retrieval_eval.py    ← Retrieval recall@k evaluation
│   └── ingest.py            ← KB seed script (multi-format)
└── outputs/
    ├── runs/                ← Per-run telemetry JSON (gitignored)
    ├── articles/            ← Generated HTML articles (gitignored)
    ├── tavily_cache/        ← Tavily result cache (gitignored)
    └── retrieval-baseline-chromadb/  ← ChromaDB baseline for comparison
```

---

## Build steps

| Step | Status | Deliverable |
|------|--------|-------------|
| 1 — Logging + core nodes | ✓ complete | structlog, verify_node, reflect_node, smoke_test passing |
| 2 — Full pipeline | ✓ complete | html_gen_node, git_node, main.py CLI |
| 3 — Evals + benchmark | ✓ complete | Prompt evals, retrieval golden set (100% recall@3), 5-round benchmark |
| 4 — Qdrant + Docling | ✓ complete | Qdrant migration, BM25+RRF, multi-format ingest |
| 5 — API + fault injection | in progress | FastAPI, 5 fault modes tested |
| 6 — Docker + AWS | planned | Containerized deployment |

---

## Phase 4a → 4b gate

Multi-agent (Phase 4b) is only justified if evaluation data shows the single agent
is hitting systematic limits. See `architecture.md § 10` for the exact criteria.
Gate report: `docs/archive/retrieval-baseline-chromadb/final_phase4a_gate_report.md`