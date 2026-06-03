# architecture.md — Content Agent (Phase 4a)
# The Machinist · AI/Tech Article Drafting Pipeline
# Status: Day 22 — Contract locked before any logic code is written

---

## 1. Big Picture

This agent takes a topic as input and produces a publish-ready HTML article for
themachinist.org, pushed to a feature branch on the themachinist-website repo.

```
[You: topic + intent]
        │
        ▼
[DRAFT NODE] — DeepSeek generates structured 5-section draft
        │
        ▼
[RETRIEVE NODE] — Tavily (web, current facts) + ChromaDB (KB, evergreen concepts)
        │
        ▼
[VERIFY NODE] — Every factual claim scored for source grounding (0.0–1.0)
        │
        ▼
[REFLECT NODE] — Agent scores its own draft (structure + grounding, 1–10)
        │          If score < 7: loop back to DRAFT (max 2 iterations)
        ▼
[HITL GATE] — You see: full draft + source grounding report + reflection score
        │      You: approve / reject / give feedback
        ▼
[HTML GEN NODE] — Produces themachinist.org-compliant HTML (matches supervised-learning-models.html spec)
        │
        ▼
[GIT NODE] — Pushes to feature/article-<slug> branch
        │      Diffs vs main:
        │        • New file only → merge + push
        │        • Existing files changed → tag current main (keep last 5 tags) → merge
        ▼
[DONE — article is live on Netlify via GitHub → Netlify CD]
```

---

## 2. State Schema (locked)

```python
from typing import TypedDict, Literal

class AgentState(TypedDict):
    # ── Input ──
    topic: str                        # e.g. "Linear & Logistic Regression"
    slug: str                         # e.g. "linear-logistic-regression"
    series_context: str               # e.g. "Family 01 — Linear Models · supervised-learning-models.html"
    card_id: str                      # e.g. "01-A" (matches sl-card-num in parent page)

    # ── Draft ──
    draft_sections: dict              # {problem_framing, technical_dive, code_snippets, takeaways}
    draft_markdown: str               # Full assembled draft in markdown

    # ── Retrieval ──
    web_sources: list[dict]           # [{title, url, snippet, relevance_score}]
    kb_results: list[dict]            # [{text, source, distance}]

    # ── Verification ──
    grounding_report: list[dict]      # [{claim, source_url, confidence: 0.0–1.0, status: verified|weak|unverified}]
    grounding_score: float            # mean confidence across all claims

    # ── Reflection ──
    reflection_score: int             # 1–10
    reflection_notes: str             # Agent's own critique
    iterations: int                   # Draft revision count (max 2)

    # ── HITL ──
    hitl_status: Literal["pending", "approved", "rejected", "feedback"]
    hitl_feedback: str | None

    # ── Output ──
    html_output: str | None           # Final HTML string
    html_filename: str | None         # e.g. "linear-logistic-regression.html"
    branch_name: str | None           # e.g. "feature/article-linear-logistic-regression"
    git_status: Literal["not_started", "pushed", "merged", "tagged_and_merged", "failed"] | None

    # ── Telemetry ──
    run_id: str                       # UUID, set at entry
    total_tokens: int
    total_cost_usd: float
    latency_ms: dict                  # {draft, retrieve, verify, reflect, html_gen, git}
```

---

## 3. Node Definitions

### 3.1 draft_node
- **Input**: `topic`, `series_context`, `card_id`, `hitl_feedback` (on revision)
- **Action**: Calls DeepSeek to produce `draft_sections` dict with keys:
  `problem_framing`, `technical_dive`, `code_snippets`, `takeaways`
- **Output**: Populates `draft_sections`, `draft_markdown`, increments `iterations`
- **System prompt**: `prompts/draft_system.md`
- **Cost tracking**: Records token usage → `total_tokens`, `total_cost_usd`

### 3.2 retrieve_node
- **Input**: `topic`, `draft_markdown`
- **Action**:
  1. Tavily search: 5 queries derived from topic + draft content → web sources
  2. ChromaDB query: evergreen concept retrieval from local KB
- **Output**: Populates `web_sources`, `kb_results`
- **Tools used**: `web_search` (Tavily), `query_kb` (ChromaDB)

### 3.3 verify_node
- **Input**: `draft_markdown`, `web_sources`, `kb_results`
- **Action**: Extracts factual claims from draft. For each claim, matches to a source,
  assigns `confidence` score (0.0–1.0) and `status` (verified/weak/unverified).
  Computes `grounding_score` as mean confidence.
- **Output**: Populates `grounding_report`, `grounding_score`

### 3.4 reflect_node
- **Input**: `draft_markdown`, `grounding_report`, `grounding_score`
- **Action**: DeepSeek self-evaluates the draft on:
  - Structure adherence (5 sections present and coherent)
  - Technical depth appropriate for themachinist.org audience
  - Grounding quality (sourced vs unsourced claims)
  - Clarity and human readability (not AI-slop)
  Scores 1–10. Reflection score alone does NOT gate the draft.
  Composite gate: force rewrite if EITHER:
    - reflection_score < 7 AND grounding_score < 0.75
      - grounding_score < 0.60 (regardless of reflection score)
  Max 2 iterations. After 2, proceed to HITL regardless — human decides.
  Rationale: LLMs inflate self-scores. Grounding score is objective (source-backed).
  Use grounding as the hard floor, reflection as a soft signal.
- **Output**: Populates `reflection_score`, `reflection_notes`

### 3.5 hitl_node
- **Input**: Full state
- **Action**: Displays to user:
  1. Full `draft_markdown` (readable in terminal)
  2. Source grounding report (claim → source → confidence)
  3. Reflection score + agent notes
  User inputs: `a` (approve) / `r` (reject) / `f` (feedback, then types it)
- **Output**: Sets `hitl_status`, `hitl_feedback`

### 3.6 html_gen_node
- **Input**: `draft_sections`, `web_sources`, `grounding_report`, `topic`, `slug`, `card_id`
- **Action**: Generates a complete HTML file following the exact spec in `dev_agent.md`:
  - Anti-flicker script first in `<head>`
  - Full SEO block (title, description, OG, Twitter, canonical)
  - Shared CSS/JS from `./shared.css` / `./shared.js`
  - Orange accent Learning Log page style (matching `supervised-learning-models.html` CSS variables)
  - Page-specific CSS in `<style>` block (no duplication of shared.css)
  - `id="main"` on `<main>`
  - Breadcrumb: Learning Log → Concept Exploration → [Article Title]
  - Back link → `index.html#learning-log`
  - LD+JSON schema block
  - All external links: `target="_blank" rel="noopener noreferrer"`
  - Pre-ship checklist verified programmatically
- **Output**: Populates `html_output`, `html_filename`
- **Template**: `prompts/html_template.md` (locked to supervised-learning-models structure)

### 3.7 git_node
- **Input**: `html_output`, `html_filename`, `slug`
- **Action**:
  1. Writes HTML file to `../themachinist-website/` (sibling directory)
  2. Creates branch: `feature/article-<slug>`
  3. Commits with message: `feat: add <topic> article [content-agent]`
  4. Diffs `feature` vs `main`:
     - If ONLY new file added → merge to main + push
     - If any existing file changed → tag current main as `v-<YYYYMMDD>-<slug>`
       → merge to main + push → prune tags keeping last 5 only
  5. Cleans up feature branch after merge
- **Output**: Sets `branch_name`, `git_status`

---

## 4. Tool Signatures

```python
# tools/web_search.py
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Tavily search. Returns [{title, url, content, score}]"""

# tools/query_kb.py
def query_kb(query: str, n_results: int = 5) -> list[dict]:
    """ChromaDB query. Returns [{text, source, distance}]"""

# tools/save_to_kb.py
def save_to_kb(text: str, source: str, metadata: dict) -> bool:
    """Ingest approved article back into KB. Called after HITL approval."""

# tools/verify_claim.py
def verify_claim(claim: str, sources: list[dict]) -> dict:
    """Score claim against sources. Returns {confidence: float, source_url: str, status: str}"""
```

---

## 5. KB Design

- **Storage**: Qdrant (Docker), `/kb/qdrant_data/` — run via `docker-compose up -d`
- **Collection**: `machinist_evergreen`
- - **Legacy**: ChromaDB data preserved at `/kb/chroma_db/` — retrieval baseline at `outputs/retrieval-baseline-chromadb/`
- **Content**: AI/ML concepts that do not change — definitions, formulas, architectures, theory
- **NOT in KB**: Recent events, model releases, benchmark results, paper findings (→ Tavily)
- **Chunk size**: 400 tokens, 50-token overlap
- **Embedding model**: `all-MiniLM-L6-v2` via sentence-transformers (384-dim, local, free)
- **Multi-format ingest**: `tools/document_ingest.py` — Docling-based parser for PDF, DOCX, PPTX, XLSX, MD, TXT, HTML
- **Self-improvement**: Every HITL-approved article is ingested into KB post-publish
- **Seed content**: `scripts/ingest.py` — point at `/kb/seed_docs/` folder of `.md` or `.txt` files

---

## 6. Output Schema (HTML generator contract)

Article sections map to HTML as follows:

| Section | HTML element |
|---|---|
| Title | `<h1>` in `.sl-header` |
| Problem Framing | First `<h2>` + `<div class="sl-definition">` |
| Technical Deep-Dive | Subsequent `<h2>` sections with `<p>`, `<ul>`, callouts |
| Code Snippets | `<pre><code>` blocks with language label |
| Takeaways | Final `<h2>` with `.callout.callout-key` block |
| Sources | Footer-adjacent `<div class="sl-sources">` with numbered citations |

---

## 7. Telemetry Requirements (eval harness input)

Every run writes a JSON record to `outputs/runs/`:

```json
{
  "run_id": "uuid",
  "topic": "...",
  "timestamp": "ISO8601",
  "iterations": 1,
  "reflection_score": 8,
  "grounding_score": 0.84,
  "claims_verified": 12,
  "claims_weak": 2,
  "claims_unverified": 1,
  "hitl_status": "approved",
  "total_tokens": 4821,
  "total_cost_usd": 0.0029,
  "latency_ms": {
    "draft": 3200,
    "retrieve": 1800,
    "verify": 900,
    "reflect": 2100,
    "html_gen": 1400,
    "git": 800
  }
}
```

---

## 8. Git Integration Rules (from dev_agent.md, enforced in git_node)

- Branch naming: `feature/article-<slug>`
- Never commit directly to `main`
- Commit message format: `feat: add <topic> article [content-agent]`
- Diff check: `git diff main...<branch> --name-only`
- Tag format: `v-<YYYYMMDD>-<slug>` (keep last 5, prune older)
- After merge: delete feature branch

---

## 9. Evaluation Topics (20 runs for Phase 4a entry gate to Phase 4b)

These are real articles you will publish on themachinist.org, in order:

| # | Topic | Series Context |
|---|---|---|
| 01 | Linear & Logistic Regression | 01-A · Family 01 Linear Models |
| 02 | Ridge, Lasso & ElasticNet | 01-B · Family 01 Linear Models |
| 03 | Support Vector Machines | 01-C · Family 01 Linear Models |
| 04 | K-Nearest Neighbors | 02-A · Family 02 Instance-Based |
| 05 | Naive Bayes | 03-A · Family 03 Probabilistic |
| 06 | Decision Trees | 04-A · Family 04 Tree-Based |
| 07 | Random Forest | 04-B · Family 04 Tree-Based |
| 08 | XGBoost | 04-C · Family 04 Tree-Based |
| 09 | CatBoost | 04-E · Family 04 Tree-Based |
| 10 | Feedforward Neural Networks | 05-A · Family 05 Neural Networks |
| 11 | Gradient Descent (deep dive) | Concept Exploration standalone |
| 12 | Backpropagation | Concept Exploration standalone |
| 13 | Attention & Transformers | Concept Exploration standalone |
| 14 | RAG — Retrieval Augmented Generation | Concept Exploration standalone |
| 15 | LangGraph — State Machines for Agents | Agentic AI standalone |
| 16 | ReAct Agent Pattern | Agentic AI standalone |
| 17 | Embedding Models & Vector Search | Concept Exploration standalone |
| 18 | Prompt Engineering for Production | Agentic AI standalone |
| 19 | Agentic AI in Real Production | Agentic AI standalone |
| 20 | Multi-Agent Systems — When & Why | Agentic AI standalone |

---

## 10. Phase 4a → Phase 4b Entry Gate

Phase 4b (multi-agent) is ONLY justified if, after 10+ runs, at least one of:
- Reflection score < 7 on > 30% of runs (single agent can't self-correct reliably)
- Grounding score < 0.75 on > 30% of runs (verify node is overwhelmed)
- HITL rejection rate > 40% (draft quality insufficient)
- Run time > 5 min per article consistently (parallelism would help)

Document the gate decision in `outputs/phase4a_gate_report.md` with data.

---

## 11. Future Integration Hook (OpenClaw)

The pipeline entry point is `main.py run --topic "..." --series "..."`.
When OpenClaw is set up, the WhatsApp trigger maps directly to this CLI interface.
Deterministic tasks (git operations, file I/O, HTML templating) → OpenClaw.
Probabilistic tasks (drafting, reflection, grounding) → DeepSeek via this pipeline.
No changes to the pipeline needed — only a new entry point wrapper.

---

## 12. Build Steps (Phase 4a)

| Step | Branch | Status | Deliverable |
|------|--------|--------|-------------|
| 1 | feature/step1-logging-verify-reflect | ✓ merged | structlog, real verify_node, real reflect_node, smoke_test |
| 2 | feature/step2-full-pipeline | ✓ merged | html_gen_node (Option B), git_node, main.py CLI |
| 3 | feature/step3-evals-benchmark | ✓ merged | Prompt evals, retrieval eval (100% recall@3), 5-round benchmark, gate report |
| 4 | feature/step4-qdrant-docling | ✓ merged | Qdrant migration, BM25+RRF, Docling ingest, retrieval_eval_qdrant |
| 5 | feature/step5-api-fault-injection | in progress | FastAPI, failure injection (5 fault modes) |
| 6 | — | planned | Docker, docker-compose app service, AWS |