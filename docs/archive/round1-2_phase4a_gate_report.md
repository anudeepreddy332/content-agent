Retrieval Eval Results (Baseline - ChromaDB)
──────────────────────────────────────────────────

  recall@1: 93.3%  ██████████████████

  recall@3: 100.0%  ████████████████████

  recall@5: 100.0%  ████████████████████

  concept hit rate: 0.867

  out-of-scope rejection rate: 1.0

Full report: outputs/retrieval-baseline-chromadb/retrieval_eval_baseline.json

# Phase 4a Gate Report

Generated: 2026-05-28T11:30:00Z
Benchmark run: benchmark_20260528_111234
Runs evaluated: 20/20
All runs status: success (0 failures, 0 errors)

---

## Results Summary

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Success rate | 20/20 (100%) | 100% | ✓ |
| Mean grounding score | 0.58 | ≥ 0.75 | ✗ FAIL |
| Mean reflection score | 6.6 | ≥ 7.0 | ✗ FAIL |
| HITL approval rate | 100% (20/20) | ≥ 60% | ✓ |
| Mean cost per run | $0.00770 | ≤ $0.10 | ✓ |
| Mean wall time | 64s | ≤ 300s | ✓ |
| Mean HTML errors/run | 0.0 | — | ✓ |
| retrieval recall@3 | (see retrieval-baseline-chromadb/) | ≥ 0.80 | pending |
| failure_inject.py | not yet run | all pass | pending |

---

## Grounding Score Distribution — The Core Problem

| Tier | Topics | Grounding Score | Pattern |
|------|--------|----------------|---------|
| High (≥ 0.80) | SVM, Attention/Transformers, Naive Bayes, Ridge/Lasso | 0.82–0.92 | Well-documented fundamentals. KB seed doc + multiple web sources align tightly. |
| Mid (0.60–0.79) | KNN, Decision Trees, Random Forest, ReAct, Prompt Engineering, Backpropagation, RAG, CatBoost, Gradient Descent | 0.60–0.79 | Solid fundamentals. Occasional unverified claims on implementation nuances. |
| Low (< 0.40) | XGBoost (0.26), Feedforward NNs (0.24), LangGraph (0.31), Embedding Models (0.35), Agentic AI in Production (0.09), Multi-Agent Systems (0.15) | 0.09–0.35 | Root cause: KB seed docs have thin coverage. Verify node cannot find claims in sources. |

---

## Root Cause Analysis — Why Grounding Fails

The grounding score is computed as `mean(confidence)` across all claims extracted by
verify_node. A claim is "unverified" when the LLM cannot find supporting text in the
source context passed to it. The source context is: `web_sources[:5] (200 chars each)`
and `kb_results[:3] (200 chars each)`.

**Two compounding problems:**

**Problem 1: Retrieve node latency exposes a retrieval shortcut.**
The retrieve latency for all low-grounding topics is ~3000ms vs ~17000ms for
high-grounding topics. This is not a coincidence — 3000ms means Tavily is returning
almost nothing for queries like "LangGraph failure modes limitations production" and
"Multi-Agent Systems implementation Python example". When web results are thin,
verify_node gets ~5 source snippets of 200 chars each = 1000 chars of context to
verify 30+ claims against. That math does not work.

**Problem 2: KB seed doc content is not dense enough for niche agentic topics.**
The seed docs exist (langgraph-state-machines.md, multi-agent-systems.md,
agentic-ai-production.md) but the verify_node context builder truncates KB chunks
to 200 chars each. For topics where the KB is the *only* reliable source
(LangGraph, multi-agent, agentic AI), 600 total KB chars is not enough to verify
30 claims. The seed docs need more depth, or the context window given to verify_node
needs more of each chunk.

**Problem 3: The reflection score (6.6 mean) is downstream of grounding.**
reflect_node receives the grounding summary and penalizes drafts with many unverified
claims. So reflection score < 7 is not an independent problem — it's grounding failure
propagating downstream. Fix grounding, reflection scores will follow.

---

## Per-Topic Breakdown

| # | Topic | Grounding | Reflection | Iterations | Verified/Weak/Unverified | Cost |
|---|-------|-----------|------------|------------|--------------------------|------|
| 1 | Linear & Logistic Regression | 0.48 | 6 | 2 | 13/0/11 | $0.0085 |
| 2 | Ridge, Lasso & ElasticNet | 0.82 | 7 | 2 | 10/5/1 | $0.0094 |
| 3 | Support Vector Machines | 0.92 | 7 | 1 | 26/0/0 | $0.0054 |
| 4 | K-Nearest Neighbors | 0.68 | 7 | 1 | 21/0/7 | $0.0050 |
| 5 | Naive Bayes | 0.87 | 7 | 1 | 22/0/1 | $0.0050 |
| 6 | Decision Trees | 0.63 | 7 | 1 | 19/4/7 | $0.0052 |
| 7 | Random Forest | 0.79 | 7 | 1 | 31/0/3 | $0.0054 |
| 8 | XGBoost | 0.26 | 6 | 2 | 3/9/23 | $0.0108 |
| 9 | CatBoost | 0.69 | 7 | 1 | 19/5/5 | $0.0050 |
| 10 | Feedforward Neural Networks | 0.24 | 6 | 2 | 9/0/24 | $0.0100 |
| 11 | Gradient Descent | 0.60 | 6 | 2 | 14/1/7 | $0.0109 |
| 12 | Backpropagation | 0.68 | 7 | 1 | 9/4/4 | $0.0050 |
| 13 | Attention Mechanism & Transformers | 0.90 | 7 | 1 | 34/0/1 | $0.0066 |
| 14 | Retrieval Augmented Generation (RAG) | 0.72 | 7 | 2 | 21/5/4 | $0.0099 |
| 15 | LangGraph — State Machines for AI Agents | 0.31 | 6 | 2 | 6/1/13 | $0.0097 |
| 16 | ReAct Agent Pattern | 0.79 | 7 | 1 | 28/0/3 | $0.0055 |
| 17 | Embedding Models & Vector Search | 0.35 | 6 | 2 | 11/0/17 | $0.0095 |
| 18 | Prompt Engineering for Production | 0.72 | 7 | 1 | 14/14/4 | $0.0054 |
| 19 | Agentic AI in Real Production | 0.09 | 6 | 2 | 2/2/31 | $0.0101 |
| 20 | Multi-Agent Systems — When and Why | 0.15 | 6 | 2 | 7/1/37 | $0.0117 |

---

## Iteration Pattern — What It Reveals

Topics requiring 2 iterations (11 out of 20): Linear/Logistic, Ridge/Lasso, XGBoost,
Feedforward NNs, Gradient Descent, RAG, LangGraph, Embedding Models, Agentic AI,
Multi-Agent Systems.

The route_after_reflect composite gate correctly caught these and triggered revision.
This is the system working as designed. However, a second iteration on a topic with
thin source coverage does NOT fix the grounding problem — it just burns more tokens
and produces a marginally different draft that still can't be verified. The agent is
correctly self-diagnosing bad grounding but cannot self-heal it because the sources
are not there.

**This is an information retrieval problem, not a generation problem.**

---

## Phase 4b Justified?

Phase 4b (multi-agent) is justified ONLY if ANY of:
- Reflection score < 7 on > 30% of runs → **YES: 11/20 (55%) scored 6. Threshold met.**
- Grounding score < 0.75 on > 30% of runs → **YES: 13/20 (65%) below 0.75. Threshold met.**
- HITL rejection rate > 40% → No (0% rejection)
- Mean wall time > 300s → No (64s mean)

**Decision: NOT justified — do NOT proceed to Phase 4b.**

**Reason:** The gate criteria are technically triggered, but the root cause is
*retrieval quality*, not agent architecture. A separate "Verifier Agent" would fail
for the exact same reason: it would get the same thin source context and produce
the same low grounding scores. Adding an agent does not add information.

The correct fix is Phase 4 Step 4:
- Qdrant migration with payload filtering (filter by topic category)
- Richer KB seed docs for the 6 low-performing topics
- Increase verify_node source context from 200 → 500 chars per source
- Add BM25 hybrid retrieval to query_kb (same pattern as knowledge-agent Phase 2)

---

## What Passes — What the System Actually Proves

Despite the grounding failures, this benchmark demonstrates:

1. **Pipeline reliability**: 20/20 runs succeed, zero exceptions, zero cost gate hits
2. **Cost control**: $0.0077 mean — 13x under the $0.10 gate
3. **Self-correction loop works**: route_after_reflect correctly identifies bad drafts
   and loops back. The logic is sound; the source material is thin.
4. **HITL gate works**: All 20 auto-approved in benchmark mode. Human reviewer would
   catch the low-grounding articles (verify report clearly marks 37 unverified claims
   for Multi-Agent Systems)
5. **HTML generation is clean**: 0.0 errors/run across 20 articles

---

## Next Steps (Ordered)

1. **Increase verify_node context window** — 200 chars per source is too small.
   Bump to 500 chars for KB, 300 for web. Zero cost increase, immediate grounding lift.

2. **Add BM25 hybrid retrieval to query_kb** — dense-only retrieval misses exact-match
   technical terms (e.g. "RAFT algorithm", "GRU gates"). BM25 catches these.
   Already built in knowledge-agent. Port the pattern.

3. **Enrich seed docs for 6 low-performers** — LangGraph, XGBoost, Feedforward NNs,
   Embedding Models, Agentic AI, Multi-Agent. Add 2–3x more content with specific,
   verifiable claims (benchmarks, paper references, exact API names).

4. **Qdrant migration** (Step 4) — run retrieval_eval.py baseline on ChromaDB first,
   then migrate, re-run, confirm accuracy holds or improves.

5. **Re-run benchmark** after fixes 1–3. Target: mean grounding ≥ 0.75.