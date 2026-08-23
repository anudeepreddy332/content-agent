# PRODUCTION_READINESS_V2.md

Independent production-deployment audit. Every finding below is anchored to a file path,
function name, and line number/excerpt that was opened and read during this session.
No claim is inherited from any prior audit, README, or other markdown doc.

---

## FILES REVIEWED

| File | Lines |
|---|---|
| agent/nodes.py | 1729 |
| agent/graph.py | 139 |
| agent/state.py | 73 |
| api/server.py | 488 |
| config.py | 100 |
| main.py | 268 |
| prompts/draft_system.md | 47 |
| prompts/verify_system.md | 47 |
| prompts/reflect_system.md | 15 |
| prompts/html_revise_system.md | 16 |
| prompts/html_template.md | 545 |
| tools/query_kb.py | 391 |
| tools/web_search.py | 131 |
| tools/save_to_kb.py | 165 |
| observability/logger.py | 13 |
| observability/tracing.py | 54 |
| tests/conftest.py | 90 |
| tests/test_api.py | 144 |
| tests/test_api_stream.py | 140 |
| tests/test_api_publish.py | 119 |
| tests/test_failure_injection.py | 196 |
| tests/test_index_update.py | 34 |
| tests/test_slug.py | 24 |
| tests/test_tracing.py | 95 |
| evals/verifier_golden_test.py | 115 |
| evals/prompt_evals/test_draft_prompt.py | 101 |
| evals/prompt_evals/test_reflect_prompt.py | 146 |
| evals/prompt_evals/test_verify_prompt.py | 84 |
| scripts/benchmark.py | 197 |
| scripts/retrieval_eval.py | 433 |
| scripts/smoke_test.py | 117 |
| scripts/preflight_check.py | 52 |
| scripts/check_telemetry_fields.py | 54 |
| scripts/rollback_publish.sh | 59 |
| .github/workflows/ci.yml | 91 |
| .github/workflows/eval.yml | 60 |
| docker-compose.prod.yml | 37 |
| docker-compose.demo.yml | 43 |
| docker-compose.yml (root) | 26 |
| Dockerfile | 40 |
| Caddyfile | 3 |
| pyproject.toml | 32 |
| .gitignore | 72 |
| .env.example | 14 |
| docs/archive/retrieval-eval-qdrant/retrieval_eval_qdrant.json | 1112 |
| docs/archive/retrieval-baseline-chromadb/retrieval_eval_baseline.json | 1113 |
| docs/archive/retrieval-baseline-chromadb/final_phase4a_gate_report.md | 65 |
| outputs/retrieval_eval.json | 1112 |
| docs/deploy/DEPLOY.md | 60 |
| docs/deploy/DEPLOY_DEMO.md | 195 |
| docs/deploy/RECOVERY.md | 38 |
| outputs/runs/9ecfb907-1be5-491a-b549-15e0db537d6c.json | 898 |
| outputs/runs/smoke-7c0d72af.json | 932 |
| outputs/runs/228ace7c-a801-4cbf-bd1c-e40ad2f1a115.json | 658 |
| outputs/runs/a622e5f3-398f-4946-bd78-f6366d8ad763.json | 682 |
| outputs/runs/a87b8482-8ffd-4778-b5dc-7b9c0b97e393.json | 761 |

## FILES NOT FOUND

- `.env.template` — checked at project root; only `.env.example` exists (used instead, per Step 2 instructions).

All other files listed in Step 2 of the audit spec were found and read in full.

## AUDIT BLOCKERS

- No `outputs/benchmark_results/*.json` files exist in this checkout (directory does not exist — confirmed via `ls outputs/benchmark_results/` → "No such file or directory"). The Step 4 "Pipeline Reliability Metrics" instruction to compute mean `wall_time_s` from "benchmark JSON files in docs/archive/ and outputs/runs/" cannot be fully satisfied: `wall_time_s` is a field written only by `scripts/benchmark.py`'s aggregate report (`scripts/benchmark.py:90`, `:130`), not by `main.py:_write_telemetry`. A field-presence check across all 458 files in `outputs/runs/*.json` confirms **zero** files contain a `wall_time_s` key. This is marked AUDIT INCOMPLETE in the Metrics section below rather than estimated.
- `docs/archive/` contains no raw benchmark-run JSON (only the two `retrieval_eval*.json` files and three `*_phase4a_gate_report.md` markdown summaries). Step 4's "Total runs found across all benchmark JSON files" is answered using `outputs/runs/*.json` instead, with this substitution stated explicitly.

---

## INVENTORY FINDINGS (Step 1 output, verbatim)

```
=== git log ===
94e2a9f Removed Phase 4a → 4b gate.
1383386 Merge docs/readme-demo-update: add live demo video, drop dead sslip.io link
0b36e51 docs(readme): drop redundant journey tagline from demo section
82231d4 docs(readme): add live demo video section, drop dead sslip.io link
d573377 Merge final audit: updated PRODUCTION_READINESS.md and lint cleanups
3044718 chore: lint cleanups (import ordering, unused imports, f-string fixes)
ba27d15 docs(audit): final production-readiness audit on merged main
9a1ee17 Merge langsmith-fixes: default-on tracing, token/cost in traces
05dd010 feat(observability): LangSmith default-on when key+project set; attach token usage/cost to traces
1010959 Merge langsmith: opt-in LangGraph tracing, env-gated, no metric re-baseline
3d76e23 Merge docs-sync: update README, PROJECT_STATUS, DECISIONS, architecture, agent.md, deploy docs
7ecdeb4 Merge cleanup: reorg project tree, delete stale files, fix imports
962155d feat(observability): opt-in LangSmith tracing, OFF by default
5e7f400 docs(decisions): record cloud publish endpoint, EC2 deploy, LangSmith plan
f970097 docs(readme): add overview, live demo URL, cheat-sheet pointers, fix pipeline order
c3c15ef docs(status): reflect cloud deploy, audit/cleanup/docs-sync state
b989795 docs(deploy): fix DEPLOY_DEMO.md's PRODUCTION_READINESS.md path after cleanup
cbc6a0a docs(gitignore): cover fork-clone/ and docker-compose.image.yml
951e556 docs(architecture): fix stale pipeline order — retrieve before draft (M3 locked)
a11e2a6 chore(cleanup): scripts/+tools/ docstring accuracy, dead code, comment audit

=== git branch -a ===
  docs/readme-demo-update
* main
  remotes/origin/HEAD -> origin/main
  remotes/origin/docs/readme-demo-update
  remotes/origin/main

=== git status ===
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean

=== bare print() calls (agent/ api/ tools/ scripts/ observability/) ===
agent/nodes.py:1041:            print("Enter a, r, or f.")
scripts/ingest.py: 16 occurrences (CLI utility, lines 48-191)
tools/save_to_kb.py:82,127,159,165
scripts/benchmark.py: 12 occurrences (lines 40-189, CLI progress output)
scripts/preflight_check.py: 6 occurrences (lines 19-53)
scripts/check_telemetry_fields.py: 5 occurrences (lines 33-55)
scripts/smoke_test.py: 3 occurrences (lines 105-110)
scripts/archive/p2_3_reach.py, scripts/archive/m4_analyze.py, scripts/archive/p2_3_analyze.py: archived analysis scripts
scripts/retrieval_eval.py: 14 occurrences (lines 385-430, CLI report output)

=== LANGCHAIN_TRACING / LANGSMITH references ===
.env.example:12-13 (commented-out optional config)
main.py:27,223-224 (setup_langsmith_tracing import + call)
observability/tracing.py: full module (gating logic)
agent/nodes.py:63,112 (langsmith.wrappers.wrap_openai, langsmith.run_helpers.get_current_run_tree — both lazily imported only when tracing is enabled)
api/server.py:52,57,60 (setup_langsmith_tracing import + call before build_graph())
tests/test_tracing.py: full file (gating behavior tests)

=== sys.exit in evals/ scripts/ ===
evals/verifier_golden_test.py:116
evals/prompt_evals/test_draft_prompt.py:68,90,93
evals/prompt_evals/test_reflect_prompt.py:107,135,138
evals/prompt_evals/test_verify_prompt.py:77
scripts/smoke_test.py:108,115
scripts/benchmark.py:188
scripts/check_telemetry_fields.py:34,39,46,49,52
scripts/ingest.py:177
scripts/preflight_check.py:20,34,40,51
scripts/archive/p2_3_analyze.py, scripts/archive/m4_analyze.py: no sys.exit (archived analysis only)

=== max_workers / ThreadPoolExecutor ===
api/server.py:10 (docstring rationale)
api/server.py:34 (import)
api/server.py:68: EXECUTOR = ThreadPoolExecutor(max_workers=1)

=== wc -l core files ===
    1729 agent/nodes.py
     139 agent/graph.py
      73 agent/state.py
     488 api/server.py
     100 config.py
     268 main.py
    2797 total
```

---

## AREA 1: RETRIEVAL

### a) Hybrid retrieval implementation
- Dense retrieval: `tools/query_kb.py:query_kb()` (lines 281–354) embeds the query with `SentenceTransformer("all-MiniLM-L6-v2")` (`_get_encoder`, line 73) and calls `client.search(...)` against Qdrant (lines 309–316). Distance is computed as `distance = round(1.0 - hit.score, 4)` (line 323) — cosine similarity converted to a distance convention.
- BM25: implemented in `tools/query_kb.py:_build_bm25()` (lines 103–168), using `rank_bm25.BM25Okapi` over documents scrolled from Qdrant's payload store (`client.scroll`, lines 137–143). Query-time scoring in `_bm25_query()` (lines 182–215).
- RRF fusion: `tools/query_kb.py:_reciprocal_rank_fusion()` (lines 219–277), `k=60` (default param, line 222). Exact formula, quoted from the docstring (lines 234–238):
  > "Each document gets score = sum of 1/(k + rank) across both rankers... Rank is 0-indexed."
- Embedding model: `all-MiniLM-L6-v2`, 384 dimensions (`config.py:54`, `tools/query_kb.py:73`, `tools/save_to_kb.py:55`).

### b) Retrieval eval harness — `scripts/retrieval_eval.py`
- Measures `recall@k` (k=1,3,5 default), `concept_hit_rate`, and `out_of_scope_rejection_rate` against a fixed 35-query `GOLDEN_SET` (lines 26–264): 30 in-domain queries (easy/medium/hard) + 5 out-of-scope queries.
- `run_retrieval_eval()` (lines 266–372) calls `query_kb()` directly — it measures the live KB state, not a frozen fixture.
- **No `sys.exit` anywhere in this file** — confirmed by the Step 1 grep (`scripts/retrieval_eval.py` does not appear in the `sys.exit` grep output) and by reading the file in full: `main()` (lines 375–430) only prints warnings for in-domain misses (lines 413–417) and out-of-scope false positives (lines 425–430); it never exits non-zero. **This harness does not gate anything** — it is purely a measurement/reporting tool, not enforced in CI (`.github/workflows/*.yml` do not invoke `scripts/retrieval_eval.py` — confirmed by reading both workflow files in full).

### c) Retrieval metrics — `docs/archive/retrieval-eval-qdrant/retrieval_eval_qdrant.json`
Exact top-level values (read in full):
```
"recall@1": 0.967, "recall@3": 1.0, "recall@5": 1.0,
"concept_hit_rate": 0.867, "out_of_scope_rejection_rate": 0.8
```
`per_query` contains 35 entries (`in_domain_queries: 30`, `concept_eligible_queries: 30`, `out_of_scope_queries: 5` — confirmed via `len(d['per_query'])` = 35).

### d) Runtime retrieval quality gate in agent/nodes.py
There is **no recall/concept/out-of-scope gate** in the production pipeline. The only runtime check is a Tavily **source-quality refresh trigger**, not a retrieval-quality gate against the KB:
```python
# agent/nodes.py:462-479
avg_score = (sum(r.get("score", 0.0) for r in web_sources) / len(web_sources) if web_sources else 0.0)
needs_refresh = len(web_sources) < 3 or avg_score < TAVILY_MIN_AVG_SCORE
if needs_refresh:
    ...
    web_sources = []
    seen_urls = set()
    for query in queries:
        results = web_search(query, max_results=5, force_refresh=True)
```
This re-fetches from Tavily live (bypassing cache) if the first pass is sparse/low-scoring — it never halts the pipeline or rejects the run. `query_kb()` results (the KB/Qdrant side) have **no quality gate at all** in `retrieve_node` (`agent/nodes.py:509–512`): whatever `query_kb()` returns (including `[]`) is used as-is.

### e) Tavily cache — `tools/web_search.py`
- Cache dir: `CACHE_DIR = Path("outputs/tavily_cache")` (line 25), created via `CACHE_DIR.mkdir(parents=True, exist_ok=True)` (line 26).
- TTL: `CACHE_TTL_DAYS = 7` (line 27); enforced in `_load_cache()` (lines 35–49) by comparing file mtime age in days; expired entries are deleted (`path.unlink()`, line 42).
- `.gitignore` covers it: `outputs/` is excluded wholesale (`.gitignore:51`, comment: "Generated at runtime; outputs/ is a bind mount in prod, never tracked.") — `outputs/tavily_cache/` falls under this rule; there is no separate explicit entry for it.

### f) Source deduplication
Implemented in `retrieve_node`, by URL, across all 3 web queries:
```python
# agent/nodes.py:433-434, 449-452
seen_urls = set()
web_sources = []
...
for r in results:
    if r["url"] not in seen_urls:
        seen_urls.add(r["url"])
        web_sources.append(r)
```
KB-side deduplication is implicit in RRF fusion (`_reciprocal_rank_fusion`, `tools/query_kb.py:249–252`) which merges dense/BM25 hits keyed by exact text content — duplicate chunks collapse into one scored entry. Grounding-report claim deduplication (a separate concern, post-verification) is in `_deduplicate_grounding_report` (`agent/nodes.py:617–660`), using `difflib.SequenceMatcher` at a 0.85 similarity threshold.

### g) Out-of-scope rejection
**Only implemented in the eval harness, not at runtime.** `scripts/retrieval_eval.py` computes `out_of_scope_rejection_rate` by checking whether the top retrieved distance exceeds `min_distance_threshold` (lines 347–353), but this logic does not exist anywhere in `agent/nodes.py` or `tools/query_kb.py`. In production, `query_kb()` always returns its top-n results regardless of how poor the match is — there is no distance-threshold reject path. An out-of-scope topic would silently retrieve weak KB matches rather than being flagged.

---

## AREA 2: GENERATION

### a) Draft node — `agent/nodes.py:draft_node` (lines 205–385)
- Model: `DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")` (`config.py:20`).
- Temperature: `DRAFT_TEMPERATURE = 0.3` (`config.py:35`).
- Prompt file: `prompts/draft_system.md`, loaded via `_load_system_prompt()` (`agent/nodes.py:128–137`), which strips comment lines (`#`-prefixed) before use.
- `max_tokens=4000` (explicit in the call).
- Exact LLM call:
```python
# agent/nodes.py:317-323
response = _llm_call(
    client,
    model=DEEPSEEK_MODEL,
    messages=messages,
    temperature=DRAFT_TEMPERATURE,
    max_tokens=4000,
)
```

### b) Verify node — grounding_score / SV / UVR computation
`agent/nodes.py:verify_node` (lines 721–857). Grounding score:
```python
# agent/nodes.py:791-795
if grounding_report:
    grounding_score = sum(r.get("confidence", 0) for r in grounding_report) / len(grounding_report)
else:
    grounding_score = 0
```
SV ("substantive and verified") and the UVR component counts:
```python
# agent/nodes.py:801-810
n_verified = sum(1 for r in grounding_report if r.get("status") == "verified")
n_weak = sum(1 for r in grounding_report if r.get("status") == "weak")
n_unverified = sum(1 for r in grounding_report if r.get("status") == "unverified")
n_substantive = sum(1 for r in grounding_report if r.get("specificity") == "substantive")
n_substantive_verified = sum(
    1 for r in grounding_report
    if r.get("specificity") == "substantive" and r.get("status") == "verified"
)
```
UVR itself is computed and persisted per-iteration (not as a top-level state key) in the `iteration_metrics` append:
```python
# agent/nodes.py:836-838
"N": len(grounding_report),
"uvr": round(n_unverified / max(len(grounding_report), 1), 3),
"grounding_score": round(grounding_score, 3),
```
`grounding_score` is a mean of per-claim `confidence` values from the verifier LLM — it is **not** the same quantity as `verified_fraction` (computed separately in `main.py:_write_telemetry`, line 109: `len(verified) / max(len(report), 1)`). Both numbers appear in telemetry under different keys (`grounding_score` top-level vs. `grounded_depth.verified_fraction`) and should not be conflated.

### c) Reflect node — routing logic in `route_after_reflect`
`agent/nodes.py:1665–1706`. Exact thresholds, read directly from `config.py`:
- `MAX_ITERATIONS = 2` (`config.py:39`)
- `REFLECTION_THRESHOLD = 7` (`config.py:40`)
- `GROUNDING_FLOOR = 0.60` (`config.py:41`)

```python
# agent/nodes.py:1686-1706
if iterations >= MAX_ITERATIONS:
    return "hitl"
if state.get("total_cost_usd", 0) >= COST_GATE_USD:
    return "hitl"
hard_floor_fail = grounding_score < GROUNDING_FLOOR
soft_fail = reflection_score < REFLECTION_THRESHOLD and grounding_score < 0.75
if hard_floor_fail or soft_fail:
    return "draft"
return "hitl"
```
Note the **iteration ceiling check runs first** and unconditionally returns `"hitl"` once `iterations >= MAX_ITERATIONS=2`, regardless of how low `grounding_score` is — see Production Risk R3 below, evidenced by an actual run.

### d) HTML gen node
`agent/nodes.py:html_gen_node` (lines 1209–1356). Template loaded from `prompts/html_template.md`, with the fenced code block extracted:
```python
# agent/nodes.py:1218-1224
template_raw = Path("prompts/html_template.md").read_text(encoding='utf-8')
if "```html" in template_raw:
    template = template_raw.split("```html", 1)[1]
    template = template.split("```", 1)[0].strip()
```
Three sections (`problem_framing`, `code_snippets`, `takeaways`) are rendered deterministically in Python (no LLM call) via `_render_problem_framing`, `_render_code_snippets`, `_render_takeaways` (lines 1044–1090). Only `technical_dive` goes through an LLM call, in `_render_technical_dive_via_llm` (lines 1093–1127): `model=DEEPSEEK_MODEL, temperature=0.1, max_tokens=3000`.

### e) html_revise_node content-freeze guard
`agent/nodes.py:html_revise_node` (lines 1366–1421). Exact Counter-based comparison and threshold:
```python
# agent/nodes.py:1399-1410
before, after = Counter(_visible_words(original)), Counter(_visible_words(revised))
drift = sum((before - after).values()) + sum((after - before).values())
looks_like_html = ("<html" in revised.lower()) or revised.lower().startswith("<!doctype")
errors = list(state.get("error_log", []))
if drift > 2 or not looks_like_html:
    errors.append(f"html_revise: revision DISCARDED (content drift={drift} words / valid_html={looks_like_html}); kept original")
    out_html = original
else:
    out_html = revised
```
`_visible_words` (lines 1359–1363) strips tags and lowercases, then `re.findall(r"\w+", ...)` tokenizes. Drift threshold is **>2 words** (word-multiset symmetric difference) before the revision is discarded.

### f) Cost gate
`COST_GATE_USD = 0.10` (`config.py:42`). Enforced independently in three nodes, each returning a degraded/empty result instead of raising:
```python
# agent/nodes.py:727-732 (verify_node)
if state.get("total_cost_usd", 0) >= COST_GATE_USD:
    return {"grounding_report": [], "grounding_score": 0.0, "latency_ms": {...}}
# agent/nodes.py:879-882 (reflect_node)
if state.get("total_cost_usd", 0) >= COST_GATE_USD:
    return {"reflection_score": 7, "reflection_notes": "Cost gate — skipped reflect", "latency_ms": {...}}
# agent/nodes.py:1213-1216 (html_gen_node)
if state.get("total_cost_usd", 0) >= COST_GATE_USD:
    return {"html_output": None, "html_filename": None, "latency_ms": {...}}
```
Also checked in the router (`agent/nodes.py:1692–1694`, `route_after_reflect`) to force `"hitl"` instead of another revise loop. `draft_node` itself has **no cost-gate check** (confirmed by reading the full function, lines 205–385) — its own docstring says so explicitly (line 227: "Cost gate exceeded: does NOT check here — checked in route_after_reflect"), meaning a single draft call can still push `total_cost_usd` over the gate before the gate is ever evaluated.

### g) Iteration limit
`MAX_ITERATIONS = 2` (`config.py:39`), enforced in `route_after_reflect` (`agent/nodes.py:1687–1689`, quoted above in 2c).

### h) LLM error handling — `_llm_call` and tenacity
```python
# agent/nodes.py:68-78
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        RateLimitError, APIConnectionError, APITimeoutError, InternalServerError,
    )),
    reraise=True,
)
def _llm_call(client: OpenAI, **kwargs):
    response = client.chat.completions.create(**kwargs)
    ...
```
Retried exceptions: `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `InternalServerError`. Max 3 attempts, exponential backoff 2s→4s→8s (capped 10s). `reraise=True` — after exhausting retries the original exception propagates. `AuthenticationError` and `BadRequestError` are explicitly **not** retried (per the docstring, lines 90–92) — confirmed in `tests/test_failure_injection.py:test_llm_call_no_retry_on_auth_error` (lines 10–14), which asserts exactly 1 call.
`LLM_TIMEOUT_S = 120` (`config.py:29`), applied at the OpenAI client level: `OpenAI(... timeout=LLM_TIMEOUT_S, max_retries=0, ...)` (`agent/nodes.py:51–57`) — `max_retries=0` is deliberate so tenacity is the sole retry layer (comment: "tenacity owns retries; SDK-internal retries would stack (3x2=6 attempts)").

---

## AREA 3: EVALUATION HARNESSES

### a) Golden verifier test — `evals/verifier_golden_test.py`
Fixture: `CLAIMS` (lines 35–66), a 12-entry 2×2 grid (substantive/generic × supported/unsupported) verified one claim at a time against a fixed `SOURCE` text (lines 15–27). Pass criterion + exit:
```python
# evals/verifier_golden_test.py:116
sys.exit(0 if (ground_ok >= 11 and spec_ok >= 10) else 1)
```
Wired into CI: `.github/workflows/ci.yml`, job `eval-gate` (lines 46–92), specifically:
```yaml
# .github/workflows/ci.yml:86-90
- name: "Grounding-regression gate: verifier golden fixture (>=11/12, >=10/12)"
  if: steps.secret_check.outputs.has_key == 'true'
  env:
    DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
  run: uv run python evals/verifier_golden_test.py
```
This job runs **only on `pull_request`** (`if: github.event_name == 'pull_request'`, line 47) and skips cleanly (no failure) if `DEEPSEEK_API_KEY` is absent (lines 56–72) — e.g. on forks. A code comment in the workflow itself (lines 83–85) flags that this is "necessary but not sufficient" for actually blocking a merge — it still requires a branch-protection rule naming `eval-gate` as a required status check, which this audit cannot verify from the repo alone (GitHub branch-protection settings are not in any file read).

### b) Benchmark harness — `scripts/benchmark.py`
`--gate` produces a real exit:
```python
# scripts/benchmark.py:173-189
if gate:
    gate_failures = []
    if aggregate["failed"] > 0:
        gate_failures.append(f"{aggregate['failed']} run(s) failed outright")
    for r in successful:
        ...
        if uvr > 0.15:
            gate_failures.append(f"topic {r['id']:02d} UVR {uvr:.2f} > 0.15")
    if gate_failures:
        print("\nCI GATE: FAIL")
        ...
        sys.exit(1)
    print("\nCI GATE: PASS — all runs succeeded, all UVR <= 0.15")
```
Gates on exactly two conditions: zero outright run failures, and per-run UVR ≤ 0.15. **SV is deliberately not gated** — explicit in-code rationale (lines 170–172): "SV is deliberately NOT gated here — at n=3 the +/-6-7 noise band ... would make an SV threshold flake; it is reported only."

### c) Prompt evals — `evals/prompt_evals/*.py`
All three (`test_draft_prompt.py`, `test_reflect_prompt.py`, `test_verify_prompt.py`) make **real LLM calls** via a real `OpenAI` client against `DEEPSEEK_BASE_URL` with `os.getenv("DEEPSEEK_API_KEY")` (e.g. `test_draft_prompt.py:24-25`, `test_reflect_prompt.py:52-55`, `test_verify_prompt.py:33-34`) — these are not mocked. Each has pass/fail criteria checked against fixed inputs (schema validation, not content-quality scoring) and exits via `sys.exit(0)`/`sys.exit(1)`. **None of the three are referenced in `.github/workflows/ci.yml` or `.github/workflows/eval.yml`** — confirmed by reading both workflow files in full; they exist only as manually-run scripts.

### d) CI enforcement — every `.github/workflows/*.yml`
Two workflow files exist (confirmed via `ls .github/workflows/`):

**`ci.yml`** — triggers: `push` (all branches) and `pull_request` (against `main`).
- `lint`: `uv run ruff check --select E9,F63,F7,F82 .` — fatal-tier only (syntax errors, undefined names). Runs on every push/PR.
- `test`: `uv run pytest tests/ -v` — runs on every push/PR. Comment (lines 33–35) states this suite is "Zero network, zero API cost (verified offline 2026-06-13)."
- `eval-gate`: PR-only, conditionally skips without `DEEPSEEK_API_KEY` (see 3a above). Runs `evals/verifier_golden_test.py` only.
None of these jobs have `continue-on-error` set, so a failure does fail the job — whether that *blocks a merge* depends on GitHub branch-protection configuration not visible in this repo's files.

**`eval.yml`** — trigger: `workflow_dispatch` only (manual, never automatic). Spins up a Qdrant service container (`qdrant/qdrant:v1.9.2`, lines 14–17), ingests the seed KB, then runs three gates in sequence: `evals/verifier_golden_test.py`, `scripts/check_telemetry_fields.py`, `scripts/benchmark.py --limit <N> --gate`. Title comment: "Eval gates (manual — costs ~$0.07 in API credits)" (line 1).

### e) Metrics from actual benchmark output
`docs/archive/retrieval-baseline-chromadb/final_phase4a_gate_report.md` (read in full), R5 (final) row, quoted exactly:
```
Mean grounding | 0.62   Mean reflection | 6.7   Mean cost | $0.0090
Mean wall time | 53s    HTML errors | 0      Pipeline failures | 0
"Hard gates: All passed. Mean cost $0.0090 (threshold $0.10). Mean wall time 53s
(threshold 300s). Zero HTML errors. 100/100 runs successful."
"Quality gates: Mean grounding 0.62 (threshold 0.75 — not met, but trending upward).
Mean reflection 6.7 (threshold 7.0 — not met, but improved)."
```
**This markdown report does not contain a `prompt_version` field anywhere** — confirmed by reading the full 65-line file. The benchmark-run telemetry's `prompt_version` could not be quoted from this file; it would require the raw per-run JSON files from that benchmark sweep, which are not present in `docs/archive/` (only the markdown summary and the two retrieval-eval JSONs are archived there). Marked AUDIT INCOMPLETE for that specific sub-claim.

---

## AREA 4: OBSERVABILITY

### a) Structured logging
`observability/logger.py` (13 lines, read in full):
```python
import structlog, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
structlog.configure(
    processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()],
    logger_factory=structlog.PrintLoggerFactory(),
)
def get_logger(node_name: str):
    return structlog.get_logger().bind(node=node_name)
```
Every node in `agent/nodes.py` calls `get_logger(...)`: confirmed by grep-equivalent reading — `draft_node`, `retrieve_node`, `verify_node`, `reflect_node`, `html_gen_node`, `html_revise_node`, `git_node`, `route_after_reflect`, `route_after_hitl`, `route_after_html_review` all instantiate a `log = get_logger("<node>")` at the top of the function and call `.info`/`.warning`/`.error`. The **one exception**: `hitl_node` (lines 961–1041) and `hitl_html_node` (lines 1131–1191) — neither calls `get_logger` at all; they use `rich.console.Console` for CLI display and rely on the caller (`route_after_hitl`, `route_after_html_review`) to log the resulting decision. HITL nodes themselves are silent in structlog terms.

### b) Tracing — `observability/tracing.py` (54 lines, read in full)
LangSmith tracing is implemented and **env-gated, default-on when configured** (not opt-in via a separate flag):
```python
# observability/tracing.py:36-46
if os.environ.get("LANGSMITH_TRACING") == "0":
    return False
if not os.environ.get("LANGCHAIN_API_KEY"):
    return False
project = os.environ.get("LANGCHAIN_PROJECT")
if not project:
    return False
os.environ["LANGSMITH_TRACING"] = "true"
log.info("tracing.langsmith_enabled", project=project)
return True
```
So tracing turns on automatically once `LANGCHAIN_API_KEY` + `LANGCHAIN_PROJECT` are both set, unless `LANGSMITH_TRACING=0` forces it off — there is no separate "opt-in" toggle required. `tests/test_tracing.py` (95 lines) tests: disabled-by-default with nothing set, disabled when either var is missing, enabled when both are set with no flag, force-off override, `is_tracing_enabled()` reflects state, and that importing `api.server` in a clean subprocess never crashes regardless of tracing env vars (real/dummy credentials).

### c) Telemetry fields — `main.py:_write_telemetry` (lines 32–142)
Every field written to `outputs/runs/<run_id>.json`, read directly from the function body: `run_id`, `topic`, `slug`, `timestamp`, `prompt_version`, `prompt_hashes`, `iterations`, `iteration_metrics`, `m4_feedback_claims`, `experiment_flags` (`m4_force_revise`, `m4_grounding_feedback`, `m4_freeze_cache`), `reflection_score`, `reflection_notes`, `grounding_score`, `hitl_status`, `html_review_status`, `git_status`, `total_tokens`, `total_cost_usd`, `latency_ms`, `error_log`, `claims_verified`, `claims_weak`, `claims_unverified`, `grounding_breakdown` (`unverified_no_source`, `unverified_has_source`, `weak_count`, `mean_confidence_verified`, `mean_confidence_unverified`), `grounding_report`, `web_sources_count`, `kb_results_count`, `web_sources` (truncated to 2000 chars/content), `kb_results` (truncated to 2000 chars/text), `attribution` (`web`, `kb`, `none`, `unresolved`).
Cross-checked against the 5 sampled run files — all fields are present and match in every one of the 5, **with one exception**: `outputs/runs/smoke-7c0d72af.json` has `"prompt_version": "unknown"` instead of a real hash (see Production Risk R4).

### d) Prompt version
Computed in `config.py:91–99`:
```python
def _prompt_hash(rel_path: str) -> str:
    return hashlib.sha256((_PROJECT_ROOT / rel_path).read_bytes()).hexdigest()[:12]
PROMPT_HASHES = {_Path(p).stem: _prompt_hash(p) for p in _PROMPT_FILES}
PROMPT_VERSION = "sha-" + hashlib.sha256(
    "|".join(f"{k}:{v}" for k, v in sorted(PROMPT_HASHES.items())).encode("utf-8")
).hexdigest()[:12]
```
Over 4 files: `draft_system.md`, `verify_system.md`, `reflect_system.md`, `html_template.md` (`config.py:84–89`). It appears in telemetry as `prompt_version` (top-level) and `prompt_hashes` (per-file breakdown) — both written by `_write_telemetry` (`main.py:49,55`). Confirmed live: all 5 sampled runs except the smoke-test one show `"prompt_version": "sha-6687240c8cd8"` with identical `prompt_hashes`.

### e) Per-run reconstructability
**Can be reconstructed** from a single `outputs/runs/<id>.json`: the full draft topic/slug, every claim extracted by the verifier per iteration (via `iteration_metrics[].grounding_report`), the exact web sources and KB chunks retrieved (truncated to 2000 chars each), cost/token/latency breakdown per node, the HITL/HTML-review/git outcome, and the exact prompt hashes in effect.
**Cannot be reconstructed**: the actual rendered HTML output (`html_output` is not part of the telemetry record — only `outputs/articles/<archive_name>.html` has it, a separate file not referenced by run_id in the JSON), the raw LLM responses before JSON-parsing (only the final parsed `grounding_report`/`draft_sections` survive), and — per the B4 limitation — whether/when a human actually clicked approve/reject in the API flow (only the *result* `hitl_status` is recorded, not a timestamped audit trail of the review action itself).

### f) Bare print() calls
From the Step 1 grep (full list above under Inventory Findings). Classification:
- `agent/nodes.py:1041` (`print("Enter a, r, or f.")`) — **production code path**, but only reached in the CLI-interactive HITL branch (`os.environ.get("HITL_MODE") != "api"`), i.e. never during an API-served run. Low impact.
- `tools/save_to_kb.py:82,127,159,165` — **production code path** (ingestion tool used at deploy/setup time, not part of the request-serving pipeline). Not structlog-routed.
- `scripts/ingest.py`, `scripts/benchmark.py`, `scripts/preflight_check.py`, `scripts/check_telemetry_fields.py`, `scripts/smoke_test.py`, `scripts/retrieval_eval.py`, `scripts/archive/*.py` — all CLI/utility scripts, not part of `agent/nodes.py` or `api/server.py`'s runtime request path.

---

## AREA 5: RELIABILITY

### a) LLM retry policy
Quoted in full under Area 2h. Retries: `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `InternalServerError`. 3 attempts, `wait_exponential(multiplier=1, min=2, max=10)` (2s→4s→8s, capped 10s), `reraise=True`.

### b) Timeout
`LLM_TIMEOUT_S = 120` (`config.py:29`). Applied via the OpenAI client constructor: `OpenAI(api_key=..., base_url=DEEPSEEK_BASE_URL, timeout=LLM_TIMEOUT_S, max_retries=0)` (`agent/nodes.py:52–57`).

### c) Empty retrieval
- Qdrant returns 0 results / unreachable: `tools/query_kb.py:query_kb()` — `_collection_exists()` (lines 81–88) catches any exception and returns `False`; `query_kb` then returns `[]` immediately (line 300). `tests/test_failure_injection.py:test_query_kb_qdrant_down_returns_empty` (lines 84–93) verifies this with a `DeadQdrant` stub raising `ConnectionError`.
- Tavily returns 0 results: handled by the source-quality refresh path in `retrieve_node` (Area 1d) — if the post-refresh count is still 0, `web_sources = []` is simply returned; `retrieve_node`'s own docstring (lines 408–412) states: "Both empty: draft proceeds to HITL with a low grounding score — human sees it." `tests/test_failure_injection.py:test_retrieve_tavily_empty_results` (lines 41–59) confirms `web_sources == []`, the KB path still runs, and `error_log` records the low-quality-sources event.

### d) Malformed verifier output
`agent/nodes.py:verify_node`, lines 773–778:
```python
try:
    grounding_report = _extract_json_array(raw)
except (json.JSONDecodeError, ValueError) as e:
    log.error("verify.parse_failed", run_id=state["run_id"], error=str(e), raw_preview=raw[:300])
    grounding_report = []
```
`_extract_json_array` (`agent/nodes.py:140–169`) itself is a 2-layer tolerant parser (fence-stripped `json.loads`, then a `[`...`]` slice-and-parse fallback) before raising. On total failure, `verify_node` degrades to an empty grounding report rather than crashing — confirmed by `tests/test_failure_injection.py:test_verify_malformed_output_degrades_gracefully` (lines 120–129), which also asserts an `iteration_metrics` entry is still appended with `N=0`.

### e) Cost gate behavior
When the cost gate fires in `verify_node`/`reflect_node`/`html_gen_node`, each returns a small dict of degraded defaults (quoted in Area 2f) — LangGraph merges this into state and the graph proceeds along its normal edges (e.g., `verify` → `reflect` still fires even with an empty grounding report). The actual *routing* consequence is in `route_after_reflect` (`agent/nodes.py:1692–1694`): cost-gate-exceeded forces `"hitl"` directly, skipping any further revise loop.

### f) HITL registry on restart
`main.py` has no registry at all — it's a synchronous CLI (`graph.invoke(initial_state)`, line 242), with no persistent run registry concept. `api/server.py` initializes `REGISTRY: dict[str, dict] = {}` (line 69) as an **in-process Python dict**, populated only by `create_run`/`ui_create_run` at request time. There is no startup code in `api/server.py`'s `lifespan()` (lines 254–262) that reads `CHECKPOINTER`/`GRAPH.get_state(...)` to rebuild `REGISTRY` entries for in-flight `thread_id`s — confirmed by reading `lifespan()` in full: it only calls `kb_warmup()` and yields. **REGISTRY is not rehydrated from the SQLite checkpoint on startup.**

### g) Failure injection tests — `tests/test_failure_injection.py` (196 lines, read in full)
Failure modes tested, and what each proves:
1. DeepSeek `AuthenticationError` — not retried, propagates after 1 call (`test_llm_call_no_retry_on_auth_error`).
2. DeepSeek `RateLimitError` — retried exactly 3 times then reraised (`test_llm_call_retries_rate_limit_then_reraises`).
3. DeepSeek `APITimeoutError` — recovers on the 3rd attempt (`test_llm_call_timeout_recovers_on_third_attempt`).
4. `draft_node` auth-error propagation to the crash handler — proves draft does not silently swallow a fatal credential error (`test_draft_auth_error_propagates_to_crash_handler`).
5. Tavily returns empty — KB path still runs, `error_log` captures it, latency split invariant holds (`test_retrieve_tavily_empty_results`).
6. Tavily raises (`ConnectionError`) on every call — captured into `error_log`, never raises out of `retrieve_node` (`test_retrieve_tavily_errors_logged_not_fatal`).
7. Qdrant connection failure — `query_kb` returns `[]`, never raises (`test_query_kb_qdrant_down_returns_empty`).
8. KB down but web up — `retrieve_node` still returns web results (`test_retrieve_survives_kb_down`).
9. Malformed verifier JSON (3 variants: fenced, preamble-wrapped, pure garbage) — extraction layer correctness (`test_extract_json_array_*`).
10. `verify_node` receiving non-JSON — degrades to empty report, not a crash (`test_verify_malformed_output_degrades_gracefully`).
11. `draft_node` receiving non-JSON — preserves the raw text for debugging, marks a `[PARSE ERROR]` (`test_draft_malformed_output_degrades_gracefully`).
12. Cost gate breach before `verify_node`/`reflect_node` LLM calls — proves the LLM is never actually called once the gate is tripped (`test_verify_cost_gate_skips_llm_entirely`, `test_reflect_cost_gate_skips_llm`).
13. Cost gate vs. revise-gate priority — cost gate wins even on a terrible draft (`test_route_cost_gate_forces_hitl_over_revision`).
14. Max-iterations forces HITL regardless of score (`test_route_max_iterations_forces_hitl`).
15. Published filename stability across re-publishes (B6 regression) — `test_published_filename_always_slug_based`.

### h) html_revise content-freeze guard
Quoted in full in Area 2e. `Counter`-based word-multiset symmetric difference, discard threshold `drift > 2` words (or invalid HTML structure).

---

## AREA 6: SECURITY

### a) Auth — `require_auth`, `api/server.py:235–241`
```python
def require_auth(authorization: str = Header(default="")):
    expected = os.environ.get("API_BEARER_TOKEN")
    if not expected:
        raise HTTPException(503, "API_BEARER_TOKEN not configured on the server")
    presented = authorization[7:] if authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(401, "invalid or missing bearer token")
```
Fail-closed: if the server has no token configured, every authenticated route 503s rather than silently allowing access. Comparison uses `hmac.compare_digest` (constant-time).

### b) SSE token — `GET /ui/runs/{id}/events`
`api/server.py:380–389`:
```python
async def ui_events(run_id: str, token: str = ""):
    expected = os.environ.get("API_BEARER_TOKEN")
    if not expected:
        raise HTTPException(503, "API_BEARER_TOKEN not configured on the server")
    if not hmac.compare_digest(token, expected):
        raise HTTPException(401, "invalid or missing token")
```
Token arrives as a **query parameter** (`token: str = ""`), not a header — same `compare_digest` check as `require_auth`. This is documented in-code (lines 382–384): "EventSource cannot set headers, so the bearer token arrives as a query param... (Query-param tokens can land in access logs; acceptable for a single-user demo.)" The same pattern is used for `GET /ui/runs/{id}/preview` (lines 416–426).

### c) Slug sanitization — `main.py:_make_slug` (lines 145–151)
```python
def _make_slug(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower().replace("&", "and"))
    return re.sub(r"-{2,}", "-", slug).strip("-")[:80]
```
Allowlist: only `[a-z0-9-]` survives (everything else collapses to `-`). Max length: 80 characters (`[:80]`). `tests/test_slug.py` (24 lines) directly verifies path-traversal and shell/git-ref-injection neutralization, full output-alphabet closure (`re.fullmatch(r"[a-z0-9-]*", ...)`), the length cap, and that a fully-degenerate topic (`"———"`) yields an empty string (which callers must reject — and do: `main.py:227-228`, `api/server.py:276-277`).

### d) Path traversal in `git_node`
`agent/nodes.py:git_node` (lines 1496–1659). The destination is built as `dest_path = repo_path/filename` (line 1572), where `filename = state.get("html_filename")` — and `html_filename` was set earlier in `html_gen_node` as `filename = f"{state['slug']}.html"` (line 1322). Because `slug` is constrained by `_make_slug`'s allowlist (no `/`, no `..`, no null bytes), `filename` cannot escape `repo_path` by construction. **There is no explicit path-containment check in `git_node` itself** (no `os.path.commonpath`/`resolve().is_relative_to(...)` assertion on `dest_path` against `repo_path`) — safety here is delegated entirely to upstream slug sanitization, not independently re-verified at the point of the filesystem write. `repo_path` itself (`Path(THEMACHINIST_REPO_PATH).resolve()`, line 1552) comes from an operator-set environment variable, not user input.

### e) Topic input sanitization before LLM prompt
**Gap, not implemented.** `state["topic"]` is inserted directly into the `draft_node` user message via f-string interpolation with no escaping/sanitization:
```python
# agent/nodes.py:291-303
user_message = f"""Write a technical article for The Machinist on the following topic.
                    Topic: {state['topic']}
                    ...
```
The only constraint on `topic` anywhere in the system is that `_make_slug(topic)` must not be empty (`main.py:227-228`, `api/server.py:276-277`) — the raw `topic` string itself (which can contain arbitrary text, including prompt-injection-style instructions) is never filtered before reaching the LLM. Both `CreateRun.topic` (Pydantic `Field(min_length=1)`, `api/server.py:245`) and the CLI `--topic` option accept arbitrary strings.

### f) Secrets — `.gitignore` (72 lines, read in full)
```
.env             (line 35)
.venv / venv/ / env/ / ENV/   (lines 36-38)
outputs/         (line 51)
fork-clone/      (line 69)
*.sqlite, *.sqlite-*   (lines 56-57)
docker-compose.image.yml   (line 72)
```
All present.

### g) Network isolation — `docker-compose.prod.yml`
```yaml
# qdrant service, lines 7-19
qdrant:
  # No `ports:` — network isolation. Only services on this compose network reach it.
  ...
# app service, line 29
ports:
  - "127.0.0.1:8000:8000"            # loopback only — access via SSH tunnel, not public
```
Qdrant: no host port binding at all (private to the compose network). App: bound to `127.0.0.1`, not `0.0.0.0` — not reachable from outside the host without an SSH tunnel. `docker-compose.demo.yml` overrides the app's ports with `ports: !reset []` (line 40, "Caddy is the only public entry point now") and adds `caddy` bound to `0.0.0.0:80`/`0.0.0.0:443` (implicit — `"80:80"`, `"443:443"`, lines 21–22, no host IP prefix means all interfaces).

### h) Container user — `Dockerfile`
```dockerfile
# line 32-33
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app/.cache /home/appuser
USER appuser
```
Runs as uid 10001, non-root. Note the `chown` only covers `/app/.cache` and `/home/appuser` — not the full `/app` tree (e.g. `/app/outputs`, the bind-mount target) — `docs/deploy/DEPLOY.md` step 4 separately documents `sudo chown -R 10001:10001 outputs` on the **host** side for the bind-mounted volume.

### i) CORS
**Absent.** No `from fastapi.middleware.cors import CORSMiddleware` or any CORS-related import/`add_middleware` call anywhere in `api/server.py` — confirmed by reading the full 488-line file.

### j) Rate limiting
**Absent.** No rate-limiting middleware, decorator, or manual counter exists anywhere in `api/server.py` — confirmed by reading the full file. The only practical throttle on request volume is the single-worker `ThreadPoolExecutor(max_workers=1)` (Area 8e), which serializes *compute* but does not limit how many HTTP requests (e.g. `POST /runs`) can be queued or how fast.

---

## AREA 7: DEPLOYMENT

### a) Compose files
- `docker-compose.prod.yml`: services `qdrant` (private, named volume `qdrant_data`) + `app` (`build: .`, env_file `.env`, loopback port, `GIT_PUSH_ENABLED: "false"`).
- `docker-compose.demo.yml`: layers on top — adds `caddy` (public 80/443, reverse-proxies to `app:8000`), overrides `app`'s `GIT_PUSH_ENABLED: "true"`, adds `PUBLISH_REMOTE`/`NETLIFY_BASE_URL`/`THEMACHINIST_REPO_PATH: /app/fork`, mounts `./fork-clone:/app/fork`, and clears `app`'s host port via `ports: !reset []`.
- `docker-compose.yml` (root): standalone, Qdrant-only, for raw local dev (see 7b).
- Discrepancy found: `docker-compose.yml` (root) and `docker-compose.prod.yml` both declare `container_name: content-agent-qdrant` for their `qdrant` service (`docker-compose.yml:16`, `docker-compose.prod.yml:10`). Running both simultaneously (e.g. `docker compose up` from the root file while `docker-compose.prod.yml` is also up) would conflict on that container name — this is not called out in either file or in `DEPLOY.md`.

### b) `docker-compose.yml` (root level)
Per its own header comment (lines 1–9, quoted verbatim): "Runs Qdrant locallu for dev / Production: swap image for a managed Qdrant Cloud instance or EC2". It is a **dev convenience** — Qdrant only, with both REST (`6333`) and gRPC (`6334`) ports exposed to the host, data persisted at `./kb/qdrant_data`. It is not referenced anywhere in `docs/deploy/DEPLOY.md` (which uses `docker-compose.prod.yml` exclusively). It does not "conflict" with `docker-compose.prod.yml` in the compose-override sense (they are never passed together via `-f`), but does conflict at the Docker container-name level if both are brought up independently (see 7a).

### c) Caddyfile (3 lines, full file)
```
{$DEMO_DOMAIN} {
    reverse_proxy app:8000
}
```
Domain variable: `$DEMO_DOMAIN`, sourced from the `caddy` service's `environment: DEMO_DOMAIN: "${DEMO_DOMAIN}"` in `docker-compose.demo.yml:27`, which in turn is read from the host's `.env`/shell environment — `docs/deploy/DEPLOY_DEMO.md` step 4 (lines 38–44) documents setting it via `sslip.io`.

### d) Dockerfile
```dockerfile
FROM python:3.14-slim-bookworm        # line 2
...
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app/.cache /home/appuser   # line 32
USER appuser                           # line 33
```
**Not** a multi-stage build — a single `FROM` statement for the entire image (confirmed: only one `FROM` line in the 40-line file). Copies: `pyproject.toml` + `uv.lock` first (dependency layer, cached), then `RUN uv sync --frozen --no-dev` (drops dev deps like pytest/ruff), then bakes the embedding model offline (`RUN python -c "...SentenceTransformer('all-MiniLM-L6-v2')"`, line 26), then `COPY . .` (full source, line 29) — source copied last so dependency layers stay cached across source-only changes.

### e) Rollback — `scripts/rollback_publish.sh` (59 lines, read in full)
Critical revert command:
```bash
# lines 43-47
if git rev-parse -q --verify "${TARGET}^2" >/dev/null 2>&1; then
    git revert -m 1 --no-edit "$TARGET"
else
    git revert --no-edit "$TARGET"
fi
```
Does **not** use `--no-commit` — `git revert --no-edit` commits immediately (no edit prompt, but still a real commit). Data-loss protection: aborts if the working tree isn't clean (line 20: `[ -z "$(git status --porcelain)" ] || { echo "ABORT..."; exit 1; }`) and requires the operator to type the exact slug to confirm (lines 39–40: `read -r -p "Type the slug to confirm revert:..."`). The script explicitly does **not** push (line 57: "Reverted locally. Live rollback requires the supervised push").

### f) Docker Hub path
Documented (not scripted in any committed file) in `docs/deploy/DEPLOY_DEMO.md:56`:
```
docker buildx build --platform linux/arm64 -t anudeepreddy332/content-agent:demo --push .
```
The image tag (`anudeepreddy332/content-agent:demo`) is hardcoded in this doc text only — `docker-compose.image.yml`, which would reference it in a compose file, is explicitly excluded from the repo by `.gitignore:72` and documented as "a deploy-time artifact... recreated per-deploy on the EC2 box; not meant to be tracked here" (`DEPLOY_DEMO.md:61-63`).

### g) GIT_PUSH_ENABLED gate — `agent/nodes.py:git_node`, lines 1529–1539
```python
git_push_enabled = os.environ.get("GIT_PUSH_ENABLED", "false").lower() == "true"
if not git_push_enabled:
    log.info("git.dry_run", run_id=state["run_id"], branch=branch,
             note="GIT_PUSH_ENABLED is not true — skipping git operations")
    return {"branch_name": branch, "git_status": "dry_run", "latency_ms": existing_latency, "error_log": error_log}
```
`false` (default): every git operation is skipped entirely, `git_status` returns `"dry_run"`. `true`: the function proceeds to write the file, commit, and do a **local-only** `--no-ff` merge (lines 1567–1633) — it never calls `git push` anywhere in this function (confirmed by reading the full 1496–1659 range; the only `git push` in the entire codebase is in `api/server.py:ui_publish`, lines 463–467, a separate human-triggered endpoint).

### h) DEPLOY.md vs DEPLOY_DEMO.md discrepancies
- Both correctly cross-reference the B4 registry-volatility limitation (`DEPLOY.md:42-52`, `DEPLOY_DEMO.md:195`) — consistent, not a discrepancy.
- One discrepancy found between git history (Step 1 inventory) and current doc content: the Step 1 `git log` shows commit `82231d4` titled "docs(readme): add live demo video section, drop dead sslip.io link" and `1383386` "Merge docs/readme-demo-update: add live demo video, drop dead sslip.io link" — indicating `sslip.io` was identified as non-functional and removed from the README. `docs/deploy/DEPLOY_DEMO.md:39-44`, however, still actively documents the `sslip.io` workflow as the "Easiest path" for DNS, with no note that it may be unreliable. This is a real discrepancy between the git history (which this audit is permitted to use per Step 1) and the current content of `DEPLOY_DEMO.md`.

---

## AREA 8: OPERATIONAL READINESS

### a) Health endpoint — `api/server.py:268-270`
```python
@app.get("/health")
def health():
    return {"status": "ok"}
```
This performs **zero actual checks** — no Qdrant ping, no SQLite/checkpoint file check, no DeepSeek/Tavily reachability check. It will return `200 {"status":"ok"}` even if Qdrant is fully down, the checkpoint DB file is missing, or `DEEPSEEK_API_KEY` is invalid. It is unauthenticated by design (comment, `Dockerfile:37`: "/health is unauthenticated by design (B4) — safe for the container healthcheck").

### b) Uptime monitoring
**No uptime monitoring exists in the repository** — no UptimeRobot config file, no synthetic-check script, no monitoring-as-code anywhere in the files read. `docs/deploy/DEPLOY_DEMO.md:146-167` documents a **manual, infra-side** setup instruction (create a free external HTTP monitor pointed at `/health`, check for `"ok"` in the body, 5-minute interval) — this is a runbook step for a human to perform outside the repo, not anything committed or automated.

### c) Restart policy — `docker-compose.prod.yml` / `docker-compose.demo.yml`
- `qdrant`: `restart: unless-stopped` (`docker-compose.prod.yml:14`)
- `app`: `restart: unless-stopped` (`docker-compose.prod.yml:35`)
- `caddy`: `restart: unless-stopped` (`docker-compose.demo.yml:30`)
`app`'s restart policy is not overridden in the demo file, so it inherits `unless-stopped` from the base.

### d) Registry volatility
Covered in Area 5f. `REGISTRY` (`api/server.py:69`) is a plain in-memory dict, populated only by request handlers, never rehydrated at startup. Exact impact on a paused run if the server restarts: the SqliteSaver checkpoint (`outputs/checkpoints.sqlite`) still has the full graph state and pending `interrupt()` payload, but `GET /runs/{id}` (and `/ui/runs/{id}`) will 404 (`reg = REGISTRY.get(run_id); if not reg: raise HTTPException(404, ...)`, lines 290–292) because the run_id → thread_id mapping that only ever lived in `REGISTRY` is gone. The run is not resumable through the HTTP API at all post-restart; only direct programmatic access to the checkpointer (same `thread_id`) could resume it, and the API exposes no endpoint to do that.

### e) Single-worker constraint
```python
# api/server.py:34, 68
from concurrent.futures import ThreadPoolExecutor
EXECUTOR = ThreadPoolExecutor(max_workers=1)
```
Rationale stated in-code (lines 8–11): avoids concurrent SQLite access from the checkpointer (`check_same_thread=False` is only safe because of this single-worker invariant). This prevents **any** concurrent graph computation — if two runs are queued, the second's compute segment waits for the first's entire segment (up to the next HITL interrupt) to finish before starting, even though the human-wait time itself doesn't block the executor (`interrupt()` returns control while awaiting review).

### f) Recovery procedure — `docs/deploy/RECOVERY.md` (38 lines, read in full)
Covers the restart-mid-HITL-review case only as an **avoidance instruction**, not a remediation runbook:
```
## B4 durability limitation (carried from DEPLOY.md)
SqliteSaver checkpoints survive container restart (on the ./outputs volume); the API's
in-memory run REGISTRY does not. Drain awaiting_review runs before any restart/deploy.
```
There is no documented step-by-step procedure in this file (or in `DEPLOY.md`) for *recovering* a run that is already orphaned after a restart happened anyway — `DEPLOY.md:47-48` mentions in prose that the run "can be resumed by a process holding the same thread_id" but provides no concrete command, script, or API call to do so. The rest of `RECOVERY.md` covers article rollback (delegates to `scripts/rollback_publish.sh`), service-unhealthy diagnosis (`docker compose logs`, restart), and post-publish verification (`curl` checks) — none of which touch the orphaned-run case beyond the avoidance line quoted above.

---

## STEP 4 — METRICS REPORT

### RETRIEVAL METRICS
From `docs/archive/retrieval-eval-qdrant/retrieval_eval_qdrant.json` (read in full):
```
recall@1: 0.967   recall@3: 1.0   recall@5: 1.0
concept_hit_rate: 0.867   out_of_scope_rejection_rate: 0.8
```
`per_query` entries: **35** (eval set size: 35 queries — 30 in-domain + 5 out-of-scope, confirmed via `total_queries: 35`, `in_domain_queries: 30`, `out_of_scope_queries: 5` in the file's own metadata).
This eval was run against **Qdrant** — confirmed by the directory name (`docs/archive/retrieval-eval-qdrant/`) and by the fact that a separate, distinctly-named ChromaDB baseline file exists for comparison.
ChromaDB baseline, `docs/archive/retrieval-baseline-chromadb/retrieval_eval_baseline.json` (read in full):
```
recall@1: 0.933   recall@3: 1.0   recall@5: 1.0
concept_hit_rate: 0.867   out_of_scope_rejection_rate: 1.0
```
Comparing the two: `recall@1` improved (0.933 → 0.967) under Qdrant, but `out_of_scope_rejection_rate` **regressed** (1.0 → 0.8) — i.e., the Qdrant backend now produces one false-positive in-scope match on a query that should have been rejected as out-of-scope, where the ChromaDB baseline did not. This is a real, evidence-anchored regression that is not gated anywhere in CI (Area 1g, 3d).

### GROUNDING METRICS
From the 5 most recently modified `outputs/runs/*.json` files (read in full):

| run_id | topic | grounding_score | reflection_score | iterations | total_cost_usd | prompt_version | claims_verified | claims_unverified |
|---|---|---|---|---|---|---|---|---|
| 9ecfb907... | Gradient Descent | 0.637 | 7 | 2 | 0.01332 | sha-6687240c8cd8 | 16 | 7 |
| smoke-7c0d72af | Gradient Descent (smoke) | 0.753 | 7 | 2 | 0.01281 | **unknown** | 24 | 6 |
| 228ace7c... | Generative AI vs Agentic AI | 0.738 | 7 | 2 | 0.01019 | sha-6687240c8cd8 | 13 | 4 |
| a622e5f3... | agentic ai roadmap | 0.762 | 7 | 1 | 0.00767 | sha-6687240c8cd8 | 25 | 5 |
| a87b8482... | Test topic | 0.382 | 5 | 2 | 0.01033 | sha-6687240c8cd8 | 9 | 13 |

From `docs/archive/retrieval-baseline-chromadb/final_phase4a_gate_report.md` (R5 final round, quoted exactly above in Area 3e): `mean_grounding = 0.62`, `mean_reflection = 6.7`. The `prompt_version` for that benchmark sweep **could not be quoted** — the markdown report contains no such field (AUDIT INCOMPLETE, stated in Area 3e and the Audit Blockers section).
Topics below 0.50 grounding, quoted exactly from the gate report's "< 0.60 (needs work)" column:
```
Linear/Logistic 0.26   Backpropagation 0.20   Random Forest 0.44   Embedding Models 0.24   Multi‑Agent 0.15
```
The report's stated interpretation (quoted, not interpreted by this audit): "The remaining low scores are retrieval‑coverage problems (thin cached Tavily results), not architectural limitations."

### PIPELINE RELIABILITY METRICS
Computed directly from all 458 files in `outputs/runs/*.json` (no benchmark-aggregate JSON exists in this checkout — see Audit Blockers):
```
Total run files: 458
Parse errors: 0
Files with a "crash" entry in error_log: 3
  - outputs/runs/31d0e287-a57a-4192-860d-f2db273562b6.json -> "pipeline crash: [Errno 32] Broken pipe"
  - outputs/runs/74503e39-2c7a-4c5e-b006-7d13d71f97b0.json -> "pipeline crash: [Errno 32] Broken pipe"
  - outputs/runs/3cb41f50-0f85-46b9-baa2-3862478e8b65.json -> "pipeline crash: Draft system prompt not found at prompts/draft_system.md"
Mean total_cost_usd (n=458): 0.009966   (computed: sum=$4.5643 / 458)
Mean wall_time_s: AUDIT INCOMPLETE — requires: outputs/benchmark_results/*.json (directory does not exist in this checkout); 0/458 outputs/runs/*.json files contain a wall_time_s field.
git_status distribution: {'dry_run': 432, 'merged': 11, 'tagged_and_merged': 7, None: 6, 'not_started': 1, 'failed': 1}
hitl_status distribution: {'approved': 452, 'rejected': 3, 'pending': 3}
```
Computation shown: `mean = sum(d["total_cost_usd"] for d in all 458 files) / 458`, where `total_cost_usd` was read directly from each file's top-level field (no estimation).

---

## STEP 5 — PRODUCTION RISKS

**R1 — CRITICAL — HITL registry not rehydrated on restart**
- Evidence: `api/server.py:69` (`REGISTRY: dict[str, dict] = {}`), `api/server.py:254-262` (`lifespan()` never reads the checkpointer), `api/server.py:290-292` (`get_run` 404s if `run_id` not in `REGISTRY`).
- Impact: any server restart while a run is `awaiting_review` makes that run permanently unreachable via the HTTP API (`GET /runs/{id}` → 404), even though its full graph state survives in `outputs/checkpoints.sqlite`. The only path to recover it is manual: write a script that calls `GRAPH.get_state(...)`/`GRAPH.invoke(Command(resume=...), config)` directly against the same `thread_id`, bypassing the API entirely.
- Fix: in `lifespan()` (`api/server.py:254-262`), before `yield`, iterate the checkpointer's stored thread_ids with pending interrupts (LangGraph exposes this via `CHECKPOINTER` list/iteration APIs) and repopulate `REGISTRY[run_id] = {"status": "awaiting_review", "interrupt_payload": ..., ...}` for each. Effort: ~half a day (needs the exact LangGraph SqliteSaver listing API, plus tests). Risk: low — additive, doesn't change existing happy-path behavior.

**R2 — HIGH — `/health` performs no real checks**
- Evidence: `api/server.py:268-270` (`return {"status": "ok"}`, no checks).
- Impact: an uptime monitor (per `DEPLOY_DEMO.md:146-167`) will report healthy even when Qdrant is unreachable, the checkpoint DB is corrupted/missing, or `DEEPSEEK_API_KEY`/`TAVILY_API_KEY` are invalid — every one of those failure modes only surfaces once a real run is attempted and fails mid-pipeline.
- Fix: in `health()` (`api/server.py:268-270`), add a cheap Qdrant `get_collections()` call and a `CHECKPOINTER`/`_conn` ping, returning 503 if either fails. Keep it unauthenticated (current design intent) but make it load-bearing. Effort: 1-2 hours. Risk: low, but must keep the check cheap (no embedding model calls) to avoid slowing the container healthcheck (`Dockerfile:38-39`, 5s timeout).

**R3 — HIGH — Low-grounding drafts can reach publish with no automated floor at the human gate**
- Evidence: `agent/nodes.py:1687-1689` (`if iterations >= MAX_ITERATIONS: return "hitl"` — fires before the `GROUNDING_FLOOR` check at lines 1696-1703); real run `outputs/runs/a87b8482-8ffd-4778-b5dc-7b9c0b97e393.json` shows `grounding_score: 0.382` (well under `GROUNDING_FLOOR = 0.60`), `iterations: 2`, `hitl_status: "approved"`, `git_status: "merged"`.
- Impact: once `MAX_ITERATIONS` is hit, the pipeline always proceeds to HITL regardless of how low grounding is — by design, the human is the final authority — but there is no server-side hard block preventing approval of a sub-floor draft. The grounding score IS shown to the reviewer (`hitl_node`'s interrupt payload, `agent/nodes.py:978-989`, includes `grounding_score`), so this is a visibility-not-enforcement gap, not a silent one.
- Fix: this is a product decision, not a clear-cut bug — flagging for awareness rather than prescribing a code change without sign-off, since CLAUDE.md's locked decisions state HITL is the mandatory, final gate by design. If a harder floor is wanted, it would be a new check in `hitl_node`/`route_after_hitl` (`agent/nodes.py:961-997`, `1708-1727`) requiring an explicit override flag from the reviewer when `grounding_score < GROUNDING_FLOOR`. Effort: ~1 day including UI/API contract change. Risk: medium — changes the HITL contract, needs explicit product sign-off given the "human is final gate" design principle.

**R4 — MEDIUM — `scripts/smoke_test.py`'s initial state omits `prompt_version`, breaking comparability**
- Evidence: `scripts/smoke_test.py:28-53` (the `initial_state` dict has no `"prompt_version"` key, unlike `main.py:_build_initial_state` line 186 which sets `"prompt_version": PROMPT_VERSION`); `main.py:_write_telemetry` line 49 (`state.get("prompt_version", "unknown")`); confirmed live in `outputs/runs/smoke-7c0d72af.json:6` (`"prompt_version": "unknown"`).
- Impact: every smoke-test telemetry record is permanently unattributable to a prompt version, defeating the project's own re-baselining discipline (any cross-run grounding/SV comparison must first filter on `prompt_version` — smoke runs can't participate in that filter at all, and silently look like a distinct, incomparable bucket).
- Fix: in `scripts/smoke_test.py:28-53`, add `"prompt_version": PROMPT_VERSION` (import from `config`) to the `initial_state` dict, matching `main.py:_build_initial_state`. Effort: 5 minutes. Risk: none — purely additive telemetry field.

**R5 — MEDIUM — Bearer token passed as a URL query parameter on 2 endpoints**
- Evidence: `api/server.py:381` (`async def ui_events(run_id: str, token: str = "")`), `api/server.py:417` (`def ui_preview(run_id: str, token: str = "")`).
- Impact: tokens in URLs are commonly captured in server access logs, browser history, and any intermediate proxy/CDN logs — a real exposure path if this deployment topology changes from "single-user demo behind Caddy" (current, documented, acceptable posture) to anything with shared infrastructure or multiple users.
- Fix: none needed under the current documented single-user-demo posture (already acknowledged in-code, `api/server.py:382-384`). If the deployment posture changes, the SSE/preview auth model would need a short-lived signed-URL token instead of the long-lived bearer token. Flagging only — no action recommended under current scope.

**R6 — MEDIUM — Out-of-scope rejection rate regressed in the ChromaDB → Qdrant migration, ungated**
- Evidence: `docs/archive/retrieval-baseline-chromadb/retrieval_eval_baseline.json` (`out_of_scope_rejection_rate: 1.0`) vs. `docs/archive/retrieval-eval-qdrant/retrieval_eval_qdrant.json` (`out_of_scope_rejection_rate: 0.8`); `scripts/retrieval_eval.py` has no `sys.exit` (Area 1b) and is not invoked by either CI workflow (Area 3d).
- Impact: a real regression exists in retrieval precision on out-of-scope topics, and nothing in CI would catch a further regression — `retrieval_eval.py` is purely a manually-run report generator.
- Fix: add a `--gate` mode to `scripts/retrieval_eval.py` (mirroring `scripts/benchmark.py`'s pattern) that exits 1 if `out_of_scope_rejection_rate` drops below a chosen floor, and wire it into `eval.yml` (manual dispatch) at minimum. Effort: ~2-3 hours. Risk: low — additive, opt-in gate.

**R7 — LOW — Topic string is not sanitized before LLM prompt interpolation**
- Evidence: `agent/nodes.py:291-303` (`f"""...Topic: {state['topic']}..."""`, raw interpolation).
- Impact: a malicious or malformed topic string (prompt-injection payload) reaches the DeepSeek system/user prompt unfiltered. Limited blast radius in practice — this is a single-operator system with two mandatory HITL gates reviewing all output before publish — but it is a real, unmitigated input-sanitization gap.
- Fix: low priority given the existing HITL gates already mitigate the worst outcome (autonomous publish of injected content is impossible by design — `GIT_PUSH_ENABLED`/HITL). If desired: add a basic length cap + control-character strip on `topic` at the API boundary (`CreateRun` Pydantic model, `api/server.py:244-246`) using a `field_validator`. Effort: ~1 hour. Risk: low.

**R8 — LOW — `docker-compose.yml` (root) and `docker-compose.prod.yml` share a container name**
- Evidence: `docker-compose.yml:16` and `docker-compose.prod.yml:10`, both `container_name: content-agent-qdrant`.
- Impact: running both independently on the same Docker host will conflict; not documented as a caveat anywhere.
- Fix: rename the root `docker-compose.yml`'s container to something distinct (e.g. `content-agent-qdrant-dev`), or add a one-line comment warning against running both simultaneously. Effort: 10 minutes. Risk: none.

---

## STEP 6 — HIGHEST ROI NEXT STEPS

| # | What | File | Function | Impact | Effort | Risk |
|---|---|---|---|---|---|---|
| 1 | Add real dependency checks to `/health` | `api/server.py` | `health()` | Uptime monitoring actually reflects service health (R2) | 1-2h | Low — keep checks cheap to not blow the 5s healthcheck timeout |
| 2 | Add `prompt_version` to smoke-test initial state | `scripts/smoke_test.py` | `smoke_test()` | Restores cross-run comparability for smoke telemetry (R4) | 5min | None |
| 3 | Rehydrate `REGISTRY` from checkpointer at startup | `api/server.py` | `lifespan()` | Eliminates the "orphaned run" 404 after any restart (R1) | ~0.5 day | Low-medium — needs correct SqliteSaver iteration API; test against a real interrupt |
| 4 | Add a `--gate` mode to the retrieval eval harness | `scripts/retrieval_eval.py` | `main()` | Catches future retrieval regressions (e.g. the observed out-of-scope-rejection drop) in CI (R6) | 2-3h | Low — opt-in, mirrors `benchmark.py`'s existing pattern |
| 5 | Write a concrete recovery runbook step for orphaned awaiting_review runs (commands, not just "avoid this") | `docs/deploy/RECOVERY.md` | n/a (docs) | Closes the gap between "documented avoidance" and "documented remediation" | 1-2h | None — docs only |
| 6 | Decide and implement (or explicitly decline) a hard grounding floor at the HITL gate | `agent/nodes.py` | `hitl_node` / `route_after_hitl` | Closes the visibility-vs-enforcement gap on sub-floor approvals (R3) | ~1 day | Medium — changes the HITL contract; needs explicit product sign-off given "human is final gate" is a locked design principle |
| 7 | Resolve the root `docker-compose.yml` container-name collision | `docker-compose.yml` | n/a (compose) | Removes a latent local-dev footgun (R8) | 10min | None |

---

## OUTPUT VERIFICATION
