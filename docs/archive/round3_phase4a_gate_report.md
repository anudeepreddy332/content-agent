# Phase 4a Gate Report
Generated: 2026-05-30
Benchmark runs: 3 (Round 1: 2026-05-28, Round 2: 2026-05-28, Round 3: 2026-05-29)

## Hard Gate Results

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Pipeline failure rate | 0% (0/60 runs) | ≤ 10% | ✅ |
| Mean cost per run | $0.0089 | ≤ $0.10 | ✅ |
| Mean wall time | 65.8s | ≤ 300s | ✅ |
| HTML validation errors | 0.0/run | 0 | ✅ |

All hard gates pass. Pipeline is operationally solid.

## Quality Gate Results

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Mean grounding (Round 3) | 0.536 | ≥ 0.75 | ❌ |
| Grounding < 0.75 across runs | 15/20 topics | ≤ 30% | ❌ |
| Grounding < 0.60 consistently | 7/20 topics | ≤ 30% | ❌ |
| Mean reflection score | 6.5 | ≥ 7.0 | ❌ |
| Reflection < 7 rate | 12/20 (60%) | ≤ 30% | ❌ |
| Retrieval recall@3 | 100% | ≥ 80% | ✅ |

## Root Cause Analysis

### Two distinct failure modes — not one

**Failure Mode A — Retrieval variance (8 topics):** KNN, Decision Trees,
Random Forest, Feedforward NNs, Backpropagation, RAG, ReAct, XGBoost.
These topics show high grounding variance across benchmark runs (±0.30 or more)
because grounding is driven by which Tavily results happen to appear that day.
These topics are NOT structurally broken. They need retrieve_node stabilization:
broader query diversity + Tavily result caching.

**Failure Mode B — Seed doc insufficiency (7 topics):** Gradient Descent,
LangGraph, Embedding Models, Agentic AI, Multi-Agent Systems, Linear/Logistic
Regression, CatBoost. These topics score consistently low (<0.60) regardless of
Tavily variance. The seed docs lack claim-dense, verifiable facts. Enrichment
will help.

## Phase 4b Decision

Phase 4b (multi-agent) is technically triggered by the gate criteria.
However, the root cause is NOT single-agent architectural limitations — it is
seed doc quality and retrieval variance. Multi-agent architecture cannot fix
either of these. The recommendation is:

1. Enrich 7 consistently-weak seed docs (Failure Mode B)
2. Stabilize retrieve_node with Tavily caching (Failure Mode A)
3. Re-run benchmark — if mean grounding improves to ≥ 0.70, Phase 4b is not needed
4. If grounding remains below target after enrichment, then Phase 4b (verify specialist) is justified

Phase 4b decision: DEFERRED pending enrichment benchmark.