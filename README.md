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

# 2. Create virtual env and install deps (uv)
uv venv
source .venv/bin/activate
uv pip install -e .

# 3. Configure environment
cp .env.example .env
# Edit .env — add DEEPSEEK_API_KEY and TAVILY_API_KEY

# 4. Seed the knowledge base
mkdir -p kb/seed_docs
# Add .md or .txt files to kb/seed_docs/
uv run python scripts/ingest.py --source kb/seed_docs/

# 5. Verify KB is populated
uv run python scripts/ingest.py --stats
```

---

## Run

```bash
# Draft an article (interactive HITL)
uv run python main.py run \
  --topic "Linear & Logistic Regression" \
  --card-id "01-A" \
  --series "Family 01 — Linear Models · supervised-learning-models.html"
```

---

## Project structure

```
content-agent/
├── architecture.md          ← Contract. Read this first.
├── main.py                  ← Entry point (Day 26)
├── agent/
│   ├── state.py             ← AgentState TypedDict
│   ├── graph.py             ← LangGraph state machine (Day 23)
│   └── nodes.py             ← All node implementations (Days 23–25)
├── tools/
│   ├── web_search.py        ← Tavily wrapper ✓
│   ├── query_kb.py          ← ChromaDB query ✓
│   ├── save_to_kb.py        ← ChromaDB ingest ✓
│   └── verify_claim.py      ← Claim grounding (Day 24)
├── kb/
│   ├── seed_docs/           ← Your evergreen .md files (committed)
│   └── chroma_db/           ← Local vector DB (gitignored)
├── prompts/
│   ├── draft_system.md      ← Draft node system prompt (Day 23)
│   └── html_template.md     ← HTML gen template (Day 25)
├── scripts/
│   ├── ingest.py            ← KB seed script ✓
│   └── benchmark.py         ← Eval harness (Day 27)
├── evals/
│   └── topics.json          ← 20 evaluation topics
└── outputs/
    └── runs/                ← Per-run telemetry JSON (gitignored)
```

---

## Phase tracking

| Day | Status | Deliverable |
|-----|--------|-------------|
| 22  | ✓ Done | Architecture contract, scaffold, tools, KB ingest |
| 23  | Next   | LangGraph graph, draft node, retrieve node |
| 24  | —      | Verify node, reflect node |
| 25  | —      | HTML gen node, html_template.md |
| 26  | —      | Git node, main.py CLI |
| 27  | —      | Benchmark harness, 20-run eval |
| 28  | —      | Business layer, case study draft |

---

## Phase 4a → 4b gate

Multi-agent (Phase 4b) is only justified if evaluation data shows the single agent
is hitting systematic limits. See `architecture.md § 10` for the exact criteria.
Data goes in `outputs/phase4a_gate_report.md` after 10+ runs.
