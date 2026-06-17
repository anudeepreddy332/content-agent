# architecture.md — Content Agent (Phase 4A)
## The Machinist · AI/Tech Article Drafting Pipeline
## Status: Phase 4A implemented and reliable (100/100 benchmark runs, 0 failures).
##         Contract updated 2026-06-07 to reflect verified implementation reality.
##         Active milestone: M1 (retrieval freshness baseline). Roadmap: see agent.md (M1–M6, B1–B9).

---

## 0. How to read this file

This is the system contract. Sections describe what the code actually does today, verified against the implementation during the 2026-06-07 production-readiness audit. Where current behavior is a known gap with a scheduled fix, it is flagged inline and collected in §13. Do not treat aspirational items as built.

---

## 1. Big Picture

This agent takes a topic as input and produces a publish-ready HTML article for
themachinist.org, optionally pushed to a feature branch on the themachinist-website repo.

```
[You: topic + intent]
        │
        ▼
[DRAFT NODE] — DeepSeek generates a structured 4-section draft
        │       (NOTE: drafting is currently BLIND to retrieval — it writes
        │        before any source is fetched. M2 will make this source-aware.)
        ▼
[RETRIEVE NODE] — Tavily (web, with 7-day cache + freshness gate)
        │          + Qdrant KB (dense + BM25 + RRF, evergreen concepts)
        ▼
[VERIFY NODE] — Every factual claim extracted and scored for source grounding (0.0–1.0)
        │
        ▼
[REFLECT NODE] — Agent scores its own draft (1–10, advisory only)
        │          Composite gate: rewrite if grounding < 0.60, OR
        │          (reflection < 7 AND grounding < 0.75). Max 2 iterations.
        ▼
[HITL GATE 1 — CONTENT] full draft + grounding report + reflection + warnings
        │      approve / reject / feedback→draft   (auto-approve via HITL_AUTO_APPROVE=1)
        ▼
[HTML GEN NODE] — Produces themachinist.org-compliant HTML
        │          (3 sections rendered deterministically, technical_dive via LLM)
        ▼
[HITL GATE 2 — LAYOUT] rendered HTML; content is FROZEN here
        │      approve→git / request_changes→html_revise→(loop) / reject→END
        ▼
[GIT NODE] — Dry-run by default. Only acts when GIT_PUSH_ENABLED=true.
        │      Writes to themachinist-website repo, feature/article-<slug> branch,
        │      diffs vs main, tags-before-merge on changed files, prunes to last 5 tags.
        ▼
[DONE — article HTML in outputs/articles/; live on Netlify only if git push was enabled]
```

---

## 2. State Schema (as implemented in agent/state.py)

```python
from typing import TypedDict, Literal

class DraftSections(TypedDict):
    problem_framing: str
    technical_dive: str
    code_snippets: str
    takeaways: str

class AgentState(TypedDict):
    # Input
    topic: str
    slug: str
    series_context: str
    card_id: str

    # Draft (4 sections, not 5)
    draft_sections: DraftSections
    draft_markdown: str

    # Retrieval
    web_sources: list   # [{title, url, content, score}]      (from Tavily)
    kb_results: list    # [{text, source, distance, rrf_score}] (from Qdrant+BM25+RRF)

    # Verification
    grounding_report: list   # [{claim, source_url, confidence, status}]
    grounding_score: float   # mean confidence across all claims

    # Reflection
    reflection_score: int    # 1–10 (advisory)
    reflection_notes: str
    iterations: int          # max 2

    # HITL
    hitl_status: Literal["pending", "approved", "rejected", "feedback"]
    hitl_feedback: str | None
    html_review_status: Literal["approved", "rejected", "changes"] | None  
    html_feedback: str | None   # P2 layout note for html_revise 

    # Output
    html_output: str | None
    html_filename: str | None
    branch_name: str | None
    git_status: Literal["not_started", "pushed", "merged", "tagged_and_merged", "failed"] | None

    # Telemetry
    run_id: str
    prompt_version: str       # currently static "v1.0" — see §13
    total_tokens: int
    total_cost_usd: float
    latency_ms: dict          # {draft, retrieve, verify, reflect, html_gen, git}

    # Error log (non-fatal errors accumulate here, surfaced at HITL)
    error_log: list[str]
```

---

## 3. Node Definitions (verified against agent/nodes.py)

### 3.1 draft_node
- Input: `topic`, `series_context`, `card_id`, `hitl_feedback` (on revision).
- Action: Calls DeepSeek with `prompts/draft_system.md` to produce a 4-key `draft_sections`
  dict (`problem_framing`, `technical_dive`, `code_snippets`, `takeaways`). Parses JSON,
  strips code fences, degrades gracefully on parse failure (preserves raw for debugging).
- Output: `draft_sections`, `draft_markdown`, increments `iterations`, accrues tokens/cost/latency.
- Current reality: does NOT read `web_sources` or `kb_results`. The draft is written before
  retrieval and on every revision iteration. This is the leading root cause of unsourced
  claims and is the target of M2 (grounding-aware drafting).

### 3.2 retrieve_node
- Input: `topic`, `draft_sections.problem_framing` (preview for KB query), `error_log`.
- Action:
  1. Tavily search over 3 fixed query angles (`explained technical`,
     `failure modes limitations production`, `implementation Python example`),
     deduped by URL, sorted by score, top 10.
  2. Freshness gate: if first-pass sources are sparse (<3) or low average Tavily score
     (< `TAVILY_MIN_AVG_SCORE`, currently 0.5), re-run all queries with `force_refresh=True`.
  3. Qdrant KB query (dense + BM25, fused via RRF), top 5.
- Output: `web_sources`, `kb_results`, `latency_ms`, `error_log`.
- Tools: `web_search` (Tavily), `query_kb` (Qdrant). Tavily errors are caught per-query
  into `error_log` and do not crash the node.

### 3.3 verify_node
- Input: `draft_markdown`, `web_sources`, `kb_results`.
- Action: DeepSeek extracts every verifiable claim and assigns `source_url`, `confidence`
  (0.0–1.0), and `status` (verified/weak/unverified) against the provided sources.
  Verification is inline in this node (there is no separate `verify_claim` tool).
  `prompts/verify_system.md` excludes self-referential code-description claims and instructs
  single-extraction of duplicate claims; `_deduplicate_grounding_report` removes string-level
  duplicates (difflib at 0.85). `grounding_score` is the mean confidence.
  Cost gate: if total cost ≥ `COST_GATE_USD`, returns empty report and skips the call.
- Output: `grounding_report`, `grounding_score`, tokens/cost/latency.

### 3.4 reflect_node
- Input: `draft_markdown`, `grounding_report` summary, `grounding_score`.
- Action: DeepSeek self-evaluates structure, technical depth, grounding, and clarity, returning
  a 1–10 score plus notes (`prompts/reflect_system.md`). Cost-gated like verify.
- Gate (in `route_after_reflect`, not in this node): force a rewrite if
  `grounding_score < GROUNDING_FLOOR` (0.60, hard floor), OR
  (`reflection_score < REFLECTION_THRESHOLD` (7) AND `grounding_score < 0.75`).
  Always proceed to HITL once `iterations >= MAX_ITERATIONS` (2) or the cost gate trips.
  Rationale: LLMs inflate self-scores, so grounding is the hard floor and reflection is advisory.
- Output: `reflection_score`, `reflection_notes`, tokens/cost/latency.

### 3.5 hitl_node
- Input: full state.
- Action: Renders the draft, a grounding table (claim → status → confidence → source),
  the reflection score and notes, and any `error_log` warnings (rich console). Prompts for
  `a` (approve) / `r` (reject) / `f` (feedback). `HITL_AUTO_APPROVE=1` bypasses the prompt
  (used by `--auto` and the benchmark).
- Output: `hitl_status`, `hitl_feedback`.
- Current reality: blocking `input()`, graph compiled without a checkpointer, so HITL is
  in-process only (no resume across restarts). Durable HITL is scheduled for B4 (API).

### 3.6 html_gen_node
- Input: `draft_sections`, `web_sources`, `grounding_report`, `topic`, `slug`, `series_context`.
- Action: Loads `prompts/html_template.md`. Renders `problem_framing`, `code_snippets`, and
  `takeaways` deterministically in Python (no LLM); renders `technical_dive` via a dedicated
  DeepSeek HTML-conversion call. Builds citations from the grounding report (falling back to
  top web sources). Substitutes template placeholders, fixes LD+JSON entity escaping, and
  validates programmatically (DOCTYPE, `id="main"`, `<h1>`, no unreplaced `{{...}}`).
  Validation warnings go to `error_log`, they do not block. Cost-gated.
- Output: `html_output`, `html_filename` (written to `outputs/articles/`, suffixed on collision).

### 3.6.5 hitl_html_node + html_revise_node (P2)
- hitl_html_node: gate 2. Reviews rendered HTML for design/structure/formatting/positioning.
  approve→git, reject→END, request_changes→html_revise. CLI/API/auto-approve modes mirror hitl_node.
- html_revise_node: one temperature-0 LLM pass applying the layout note to html_output, then
  loops to hitl_html. Content-freeze guard (visible-word-multiset) discards any drifting
  revision and keeps the original. No production path bypasses either gate.

### 3.7 git_node
- Input: `html_output`, `html_filename`, `slug`, `topic`.
- Action: Dry-run unless `GIT_PUSH_ENABLED=true`. When enabled: writes HTML into the repo at
  `THEMACHINIST_REPO_PATH`, creates `feature/article-<slug>`, commits, diffs vs `main`,
  tags `v-<YYYYMMDD>-<slug>` before merge when existing files change (pruning to last 5 tags),
  merges with `--no-ff`, deletes the feature branch, and restores the original branch in a
  `finally` block. Every git operation is wrapped so failure logs to `error_log` and never
  crashes the pipeline (HTML is already saved locally).
- Output: `branch_name`, `git_status` ("dry_run" / "merged" / "tagged_and_merged" / "failed").

---

## 4. Tool Signatures (as implemented in tools/)

```python
# tools/web_search.py
def web_search(query: str, max_results: int = 5, force_refresh: bool = False) -> list[dict]:
    """Tavily search with a 7-day file cache. Returns [{title, url, content, score}].
       force_refresh bypasses and overwrites the cache."""

# tools/query_kb.py
def query_kb(query: str, n_results: int = 5) -> list[dict]:
    """Qdrant dense + BM25, fused via Reciprocal Rank Fusion (k=60).
       Returns [{text, source, distance, rrf_score}]. [] if Qdrant unreachable or empty."""

# tools/save_to_kb.py
def save_to_kb(text: str, source: str, metadata: dict | None = None) -> bool:
    """Qdrant ingest with 400/50 tiktoken chunking. Invalidates the BM25 index.
       NOTE: not yet wired into the pipeline (see §13, self-improvement)."""

# tools/document_ingest.py
def ingest_document(path: str | Path) -> list[dict]:
    """Docling multi-format parser (PDF/DOCX/PPTX/XLSX/HTML) + .md/.txt fast path.
       NOTE: Docling is non-functional on Python 3.14 (see §13). .md/.txt work today."""
```

Note: the previously listed `tools/verify_claim.py` does not exist. Verification is inline in `verify_node`.

---

## 5. KB Design

- Storage: Qdrant (Docker), `/kb/qdrant_data/`, started via `docker-compose up -d`.
- Collection: `machinist_evergreen`.
- Legacy: ChromaDB data preserved at `/kb/chroma_db/`; retrieval baseline at `outputs/retrieval-baseline-chromadb/`.
- Retrieval: dense (all-MiniLM-L6-v2, 384-dim, cosine) + BM25, fused via RRF (k=60).
- Content: evergreen AI/ML concepts (definitions, formulas, architectures, theory).
- NOT in KB: recent events, model releases, benchmark numbers, paper findings (those go to Tavily).
- Chunking: 400 tokens, 50-token overlap (tiktoken cl100k_base), consistent across save and ingest.
- Multi-format ingest: implemented in `tools/document_ingest.py` but blocked on the runtime (§13).
- Self-improvement: ingesting HITL-approved articles back into the KB is planned, not yet wired.
- Seed content: `scripts/ingest.py` against `/kb/seed_docs/` (20 `.md` seed docs committed).
- Retrieval health (verified): recall@1 0.967, recall@3 1.0, recall@5 1.0, concept-hit 0.867,
  OOS rejection 0.8 on a 35-query adversarial golden set (`outputs/retrieval_eval_qdrant.json`).

---

## 6. Output Schema (HTML generator contract)

| Section | HTML element | Rendered by |
|---|---|---|
| Title | `<h1>` in `.sl-header` | template substitution |
| Problem Framing | first `<h2>` + `<div class="sl-definition">` | Python (deterministic) |
| Technical Deep-Dive | `<h2>` sections with `<p>`, `<ul>`, callouts | LLM HTML conversion |
| Code Snippets | `<pre><code>` blocks with language label | Python (deterministic) |
| Takeaways | final `<h2>` with `.callout.callout-key` | Python (deterministic) |
| Sources | `<div class="sl-sources">` numbered citations | Python from grounding report |

Programmatic pre-ship checks: DOCTYPE present, `id="main"` present, `<h1>` present, no unreplaced `{{PLACEHOLDER}}` tokens. Failures are logged as warnings, not hard blocks.

---

## 7. Telemetry Requirements (eval harness input)

Every run writes a JSON record to `outputs/runs/<run_id>.json` (written even on a top-level crash):

```json
{
  "run_id": "uuid",
  "topic": "...",
  "slug": "...",
  "timestamp": "ISO8601",
  "prompt_version": "v1.0",
  "iterations": 1,
  "reflection_score": 8,
  "reflection_notes": "...",
  "grounding_score": 0.84,
  "hitl_status": "approved",
  "git_status": "dry_run",
  "total_tokens": 4821,
  "total_cost_usd": 0.0029,
  "latency_ms": { "draft": 0, "retrieve": 0, "verify": 0, "reflect": 0, "html_gen": 0, "git": 0 },
  "error_log": [],
  "claims_verified": 12,
  "claims_weak": 2,
  "claims_unverified": 1,
  "grounding_breakdown": {
    "unverified_no_source": 1,
    "unverified_has_source": 0,
    "weak_count": 2,
    "mean_confidence_verified": 0.9,
    "mean_confidence_unverified": 0.1
  },
  "grounding_report": [ { "claim": "...", "source_url": "...", "confidence": 0.9, "status": "verified" } ]
}
```

Primary experiment metric is claim-level unverified-rate, not `grounding_score` (the scalar has high run-to-run variance; see DECISIONS.md).

---

## 8. Git Integration Rules (enforced in git_node)

- Publish is opt-in: `GIT_PUSH_ENABLED=true` required, otherwise dry-run.
- Branch naming: `feature/article-<slug>`. Never commit directly to `main`.
- Commit message: `feat: add <topic> article [content-agent]`.
- Diff vs `main` decides strategy: new file only → merge; existing files changed → tag then merge.
- Tag format: `v-<YYYYMMDD>-<slug>`, prune to last 5.
- After merge: delete feature branch; restore the original branch even on failure.

---

## 9. Evaluation Topics (matches evals/topics.json)

The 20 benchmark topics in `evals/topics.json` are the canonical set (ids 1–20). They match the
list previously documented here. Four are current grounding-failure topics under investigation:
CatBoost (#9), ReAct (#16), Embedding Models (#17), Multi-Agent Systems (#20).

---

## 10. Phase 4A → 4B Entry Gate (SUPERSEDED)

The original numeric gate (reflection < 7 on >30%, grounding < 0.75 on >30%, HITL rejection
> 40%, runtime > 5 min) is retained for history but is superseded by the milestone roadmap and
the locked multi-agent decision.

Locked decision (DECISIONS.md, 2026-06-07): multi-agent (Phase 4B specialization) is DEFERRED.
Evidence: retrieval is healthy (recall@3 = 1.0), many topics already reach 0.85–0.94 grounding
single-agent, and the failure mode is over-claiming / thin sources, which a multi-agent verifier
would not fix. Revisit only if, after M2 (source-aware drafting) and fresh retrieval, grounding
still fails on >30% of runs while adequate evidence is demonstrably present in the retrieved set.

The roadmap that replaces this gate lives in agent.md: Phase 4A milestones M1–M6, Phase 4B
milestones B1–B9 (B = deployment, automation, autonomy gating, not multi-agent).

---

## 11. Future Integration Hook (OpenClaw)

Entry point is `main.py run --topic "..." --series "..."`. A future OpenClaw WhatsApp trigger
maps directly to this CLI. Deterministic tasks (git, file I/O, HTML templating) → OpenClaw;
probabilistic tasks (drafting, reflection, grounding) → DeepSeek via this pipeline. No pipeline
change needed, only a new entry-point wrapper. Note: any non-CLI entry point must first close
the path-sanitization gap in §13 (B1).

---

## 12. Build Status

Step labels below are historical. Current planning uses the milestone IDs in agent.md.

| Step | Status | Deliverable |
|------|--------|-------------|
| 1 | merged | structlog, real verify_node, real reflect_node, smoke_test |
| 2 | merged | html_gen_node, git_node (dry-run guarded), main.py CLI |
| 3 | merged | prompt evals, retrieval golden set (recall@3 = 1.0), 5-round / 100-run benchmark, gate report |
| 4 | merged | Qdrant migration, BM25 + RRF, document_ingest (Docling, runtime-blocked), retrieval_eval_qdrant |
| 5 | in progress | FastAPI wrapper, failure injection (5 fault modes) — now tracked as B4 / B2 |
| 6 | planned | app container, compose app service, cloud — now tracked as B5 |

Active milestone: M1 (retrieval freshness baseline). Next: M2 (grounding-aware drafting).

---

## 13. Known contradictions and current-reality notes (with scheduled fix)

These are real gaps between this contract's intent and current behavior. Each has an owning milestone.

1. Drafting is blind to retrieval (draft runs before retrieve, never reads sources). Leading
   root cause of unsourced claims. Fix: M2 (grounding-aware drafting).
2. Multi-format ingest is non-functional on the declared runtime: `pyproject.toml` requires
   Python >=3.14, Docling requires 3.11–3.13. Only `.md`/`.txt` ingest works today.
   Fix: M6 (pin runtime to 3.13 or drop Docling and document `.md`/`.txt`-only).
3. `prompt_version` is a static "v1.0" stamped into every run even though prompts have changed,
   which breaks exact run reconstruction. Fix: M6 (content-hash versioning).
4. `topic` → `slug` → filename is not sanitized for `/` or `..`; safe for single-user CLI,
   a path-traversal risk under any API/automation entry point. Fix: B1.
5. HITL is in-process only (blocking input, no checkpointer), so it cannot resume across
   restarts. Fix: B4 (durable, async-safe HITL behind the API).
6. Self-improvement (ingesting approved articles back into the KB) is described in §5 but not
   wired into the graph. Track under a later milestone before enabling autonomy (B9).