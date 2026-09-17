# Phase 5 — RAG Evaluation Contract

_Canonical Phase-5 evaluation design. Last updated: 2026-09-17 (5A0 baseline measured)._

Phase 5A0's measured control, per-query report, exact-span exposure analysis,
and promotion/rollback rules are frozen in
`docs/PHASE5_RAG_ARCHITECTURE_CONTRACT.md`,
`evals/fixtures/phase5a0_abc_contract.json`, and
`reports/phase5/phase5a0/`. This file remains the authoritative dataset and
metric-layer contract.

## V2 dataset status

`evals/fixtures/retrieval_golden_v2.json` is a **DEVELOPMENT / DIAGNOSTIC NUCLEUS** (35 queries).

It is **not**:

- the final production golden set;
- an untouched holdout;
- sufficient for absence evaluation;
- sufficient alone for final Phase-5 qualification.

All 35 cases are historically exposed as development/diagnostic. None may be relabeled as a true holdout.

Current adjudication aggregate: **ANSWERABLE=30**, **PARTIAL=5** (Q31–Q35), **ABSENT=0**. Q25/Q26 are **ANSWERABLE but AMBIGUOUS** and **non-gating** — ambiguity is not partial answerability.

## Evaluation layers (do not collapse)

Every Phase-5 retrieval experiment must report metrics in three separate layers. A single composite score is forbidden as the sole gate.

### A. RETRIEVAL

Source-level retrieval quality against v2 graded relevance (grades 1/2):

- Recall / Hit@K
- **Precision@K**
- MRR
- **nDCG@K** (enabled by v2 graded relevance, not binary hit alone)
- source recall
- evidence-span recall
- tail-evidence recall

### B. EVIDENCE EXPOSURE

Downstream exposure of retrieved evidence (distinct from retrieval rank):

- retrieved evidence recall
- drafter-exposed recall
- verifier-exposed recall
- lost-after-retrieval count

### C. END-TO-END

Existing Phase-4 groundedness / semantic verification plus:

- citation correctness
- required-content / publication safety
- later business / task metrics (when authorized)

## Per-query regression authority

Every retrieval experiment must emit:

1. aggregate metrics (per layer above);
2. per-query result delta vs baseline;
3. newly fixed queries;
4. newly regressed queries;
5. critical evidence misses.

**An aggregate improvement must never hide a critical query regression.**

Required dimensions where not already present on each case: `failure_tags`, `query_bucket`, coverage facets.

## Final golden-set roadmap

**Current:** 35-query v2 development/diagnostic nucleus.

**Future target:** grow toward a representative **100–300-case** evaluation corpus over time, mixing:

- head / common queries;
- long-tail;
- lexical / identifier;
- semantic paraphrase;
- ambiguous;
- adversarial;
- partial-evidence;
- genuinely ABSENT / OOS;
- hard negatives;
- boundary / tail evidence;
- multi-source;
- metadata-constrained;
- later table / code / PDF cases.

Do not create all categories in one step; grow incrementally with independent adjudication.

## True holdout contract

Because all 35 v2 cases are historically exposed, **none** qualify as a true holdout.

A future holdout must:

- contain previously unseen queries;
- be independently adjudicated;
- remain hidden from implementation / tuning agents;
- be evaluated only after a candidate architecture / configuration is frozen.

Do not reuse v2 queries as holdout.

## Absence-set requirement

V2 contains **zero genuine ABSENT** queries. Therefore absence / no-hit / refusal quality is **currently not measurable**.

Before final Phase-5 qualification, an independently adjudicated **ABSENT slice** is mandatory.

Do **not** reuse historical `min_distance_threshold=0.5` as an absence oracle.

## Living regression corpus lifecycle

```
production / pilot failure
  → adjudication
  → failure tag / root-cause category
  → versioned eval case
  → permanent regression test
```

Failure tags, query buckets, and coverage dimensions are first-class in the evaluation contract.

Do not invent live production cases without adjudication.

## Phase-5 production RAG principles

### Non-negotiable

- parsing before retrieval quality;
- structure-aware canonical representation;
- actual embedding-tokenizer safety;
- exact provenance;
- deterministic IDs / idempotent re-ingestion;
- versioned parser / chunker / index manifests;
- dense / BM25 separate rankers over same canonical child IDs;
- RRF by ID, not raw text;
- retrieval failure distinct from zero relevant results;
- retrieved / drafter / verifier exposure traced separately.

### Conditional, eval-driven

- cross-encoder reranker;
- contextual / generated chunk descriptions;
- BM25 tokenizer redesign;
- query rewriting / HyDE / decomposition;
- embedding-model replacement;
- semantic chunking;
- HNSW tuning;
- GraphRAG / late interaction.

Each conditional item requires pre-registered experiment arms and pass/fail criteria before adoption.
