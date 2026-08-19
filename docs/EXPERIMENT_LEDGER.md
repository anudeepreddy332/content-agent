# Experiment and Evidence Ledger

_Canonical compact index. Last synchronized: 2026-08-19._

This ledger indexes meaningful experiments and release-relevant validation. It does not duplicate
full reports. Detailed artifacts remain under `docs/archive/`, in the named commit/branch, or in the
referenced canonical decision.

## Update and interpretation rules

- Add an entry only after a material experiment or validation completes.
- Record the predeclared gate and its result; do not move thresholds after seeing results.
- Do not retry until lucky.
- A pass at one gate is not integration, merge, or cutover approval.
- An aggregate pass cannot hide a critical per-case regression.
- `INVALID` means the candidate never reached the quality gate; it is not a quality failure.
- Historical metrics with superseded evaluator semantics remain visible but are not compared to
  corrected metrics.
- Every accepted release claim must bind to an exact code SHA, fixture/corpus identity, and artifact.

## Enterprise validation sequence

For every applicable change:

1. isolate the variable;
2. perform isolated validation;
3. integrate dependencies;
4. run focused deterministic tests;
5. run the full regression suite;
6. run subsystem evaluation;
7. run causal validation where applicable;
8. run full end-to-end validation;
9. run performance/scale validation where applicable;
10. run security/failure-path validation where applicable;
11. merge or cut over only after every applicable frozen gate passes.

## Evidence index

### EXP-2026-05-30-01 — Legacy ChromaDB/Qdrant retrieval comparison

| Field | Record |
| --- | --- |
| Question | Did Qdrant hybrid retrieval preserve the earlier ChromaDB any-source retrieval result? |
| Isolated variable | Retrieval backend/fusion implementation in the historical experiment. |
| Control / candidate | ChromaDB baseline / Qdrant plus BM25/RRF. |
| Frozen gate | Historical `recall@k`, concept-hit, and OOS rules. |
| Base / environment | May 2026 Phase 4A environment; not the current exact-73 fixture. |
| Validation | 35-query archived reports; 30 in-domain and 5 labeled OOS. |
| Focused / full regression | Historical report only; no current-SHA full regression claim. |
| Subsystem / causal / E2E | Retrieval subsystem only; no causal product-quality or E2E gate. |
| Result | Historical reports printed near-ceiling any-source hit metrics. |
| Classification | `HISTORICAL — SUPERSEDED EVALUATOR SEMANTICS`. The old recall, nDCG and OOS values are non-gating and not directly comparable to corrected metrics. |
| Evidence | `docs/archive/retrieval-baseline-chromadb/`, `docs/archive/retrieval-eval-qdrant/`, `DECISIONS.md` 2026-08-11. |
| Decision enabled | Preserve as provenance; do not use as current release evidence. |
| Confidence | 0.99 |

### EXP-2026-08-11-01 — Tokenizer-safe 224/32 chunk candidate

| Field | Record |
| --- | --- |
| Question | Does eliminating MiniLM input truncation improve retrieval quality under the corrected evaluator? |
| Isolated variable | Chunk contract only: serving 400 `cl100k`/50 versus MiniLM-tokenizer 224/32. |
| Control / candidate | 73-chunk serving baseline / 139-chunk tokenizer-safe candidate. |
| Frozen gate | Corrected source recall, source nDCG, concept coverage/pass, rank diagnostics, and no material regression. |
| Base / branch / environment | Logical base `61de06d`; detached isolated collections `eval_correctness_baseline_61de06d` and `eval_correctness_candidate_224_61de06d`. |
| Isolated validation | Baseline: 64/73 chunks at truncation risk; candidate: 0/139. |
| Dependency integration | Not integrated into serving. |
| Focused / full regression | Corrected evaluator and deterministic metric tests; no production cutover regression suite. |
| Subsystem evaluation | Source recall and concept metrics at multiple depths. |
| Causal / E2E | Causal variable isolation was valid; no article-quality E2E acceptance. |
| Result | Candidate removed measured truncation but regressed top-1 ranking and top-3/top-5 concept coverage on material queries; higher-depth recovery increased sibling occupancy. |
| Classification | `CLASS C — MIXED`; truncation defect remains open and candidate is rejected for serving cutover. |
| Evidence | `DECISIONS.md` 2026-08-11 and branch `fix/embedding-chunk-alignment` history. |
| Decision enabled | Keep serving chunk/retrieval baseline; separate tokenizer safety from product quality. |
| Confidence | 0.97 |

### EXP-2026-08-12-01 — Exact-73 seven-arm channel ablation

| Field | Record |
| --- | --- |
| Question | Which dense/BM25 channels explain current fused performance, and is chunk-identity misalignment the dominant GTE failure mechanism? |
| Isolated variable | Retrieval channel/model across seven fixed arms; corpus, queries, depths, BM25 and RRF frozen. |
| Control / candidate | BM25; MiniLM, GTE and Jina dense; each dense channel fused with the same BM25/RRF where applicable. |
| Frozen gate | Exact-73 metrics and repeatability; no threshold changes; diagnostics do not authorize cutover. |
| Base / branch / environment | Logical base `794851d`; branch `experiment/exact73-retrieval-channel-ablation`; fixture `4a1d5d1d67b56867c71497cb58ed4964d356a122a14d47ef822c227dba5924e4`; 73 chunks, 20 sources, 30 queries. |
| Isolated validation | Seven arms, stored ranking fingerprints, two consecutive identical ranking/metric runs. |
| Dependency integration | Experimental only; no serving collection/code integration. |
| Focused / full regression | Focused diagnostic tests passed on the evidence branch; no serving full regression or E2E cutover gate. |
| Subsystem evaluation | MiniLM+BM25+RRF remained strongest fused configuration at @3 recall `.950`, nDCG `.948`, concept coverage `.822`, concept pass `.833`. |
| Causal / E2E | Chunk-alignment explanation weakened; no article E2E. |
| Result | GTE dense was competitive but its fusion was weaker; Jina was infeasible under the frozen memory gate and unsuitable; early-prefix causality unsupported. |
| Classification | `DIAGNOSTIC MIXED — NO CUTOVER`; MiniLM serving baseline retained. |
| Evidence | Commit `b9f9d1d`; `docs/archive/exact73_retrieval_channel_ablation_hardened.md` on the evidence branch; result SHA `5c5a9cf1bf6ccafef3f028b01f216e434079a89751eb2cdffa4ef7ece78e4207`. |
| Decision enabled | Test rank/membership mechanisms without redesigning fusion. |
| Confidence | 0.98 |

### EXP-2026-08-12-02 — Common-identity rank-position diagnostic

| Field | Record |
| --- | --- |
| Question | Does rank position among identities shared by MiniLM and GTE explain MiniLM's historical fused advantage? |
| Isolated variable | Original dense rank positions with dense membership equalized to the exact MiniLM/GTE intersection. |
| Control / candidate | MiniLM-rank replay / GTE-rank replay on shared membership. |
| Frozen gate | Exact RRF reconstruction first; compare fixed @3 metrics; diagnostic only. |
| Base / branch / environment | Parent evidence `b9f9d1d`; logical base `794851d`; same exact-73 fixture/depths. |
| Isolated validation | Historical RRF top-5 identities reproduced for 30/30 queries in both arms. |
| Dependency integration | None. |
| Focused / full regression | Focused deterministic diagnostic tests; no serving regression or E2E gate. |
| Subsystem evaluation | Equalized-membership @3 recall tied at `.933`; nDCG differed by only `-0.0027` MiniLM minus GTE. |
| Causal / E2E | Controlled rank-position replay; no product E2E. |
| Result | The historical MiniLM advantage did not survive membership equalization. |
| Classification | `RANK_POSITION_COMPLEMENTARITY_NOT_SUFFICIENT`. |
| Evidence | Commit `4f6637e`; `docs/archive/exact73_rank_position_complementarity.md` on the evidence branch; result SHA `6d6a73dac1c01a0101d4f85dcce67ff2b9884d3ee4b44f0b12981d59d71f76d4`. |
| Decision enabled | Advance only to the predeclared membership/composition diagnostic. |
| Confidence | 0.98 |

### EXP-2026-08-12-03 — Unique dense-membership crossover

| Field | Record |
| --- | --- |
| Question | Do model-unique dense identities, rather than shared-identity rank positions, explain the fused gap? |
| Isolated variable | MiniLM versus GTE unique dense membership crossed with fixed shared-rank anchors. |
| Control / candidate | Four crossover combinations: two anchors by two unique-membership sets. |
| Frozen gate | Six predeclared support gates; exact historical reconstruction; diagnostic only. |
| Base / branch / environment | Branch `experiment/exact73-retrieval-channel-ablation`; parent `4f6637e`; same exact-73 fixture. |
| Isolated validation | Exact 30/30 historical reconstruction; two consecutive identical ranking/metric runs. |
| Dependency integration | None. |
| Focused / full regression | Focused deterministic tests; no serving integration, full regression, or E2E cutover gate. |
| Subsystem evaluation | MiniLM membership exceeded GTE membership under both anchors; cross-anchor delta explained the historical recall gap and about 86% of nDCG gap, concentrated in two queries. |
| Causal / E2E | Crossover provides causal support inside the frozen diagnostic; no article E2E. |
| Result | Unique membership is a supported mechanism, but the effect is localized and does not establish a replacement architecture. |
| Classification | `UNIQUE_MEMBERSHIP_SUPPORTED — DIAGNOSTIC ONLY`. |
| Evidence | Commit `de60706`; `docs/archive/exact73_unique_membership_crossover.md` on the evidence branch; result SHA `3cabc9ba558e76161d0213cec382a37932c32c7e524d9384f46e8bd0c13c93d2`. |
| Decision enabled | Preserve serving baseline; future retrieval research must target query-local membership/composition with repaired evaluation. |
| Confidence | 0.97 |

### VAL-2026-08-13-01 — P0-1 active-content/credential boundary audit

| Field | Record |
| --- | --- |
| Question | Can model/web-influenced content execute with reviewer-app authority or expose its bearer credential at the accepted baseline? |
| Isolated variable | Read-only trace of the current rendering, preview, auth, citation, archive and publish paths. |
| Control / candidate | Current baseline only; target architecture evaluated separately. |
| Frozen rule | Treat web/model content as hostile; no raw active content or credentials in URLs; exact reviewed artifact must bind to exact published commit. |
| Base / environment | `794851dded770ce87d111e73735d000e23597eb1`; repository source plus local artifact compatibility inventory. |
| Validation | Static source trace across `static/index.html`, `api/server.py`, `agent/nodes.py`, prompts and Caddy; 450 historical HTML artifacts parsed for compatibility vocabulary. |
| Integration / regression / subsystem / E2E | No implementation was performed. Therefore no implementation regression, exploit-browser, or E2E pass exists. |
| Result | Same-origin active-content execution paths, query bearer exposure, untrusted citation URLs, pre-approval persistence, and mutable-main publication binding were verified. |
| Classification | `P0 VALIDATED BLOCKER`; architecture approved, implementation not validated. |
| Evidence | Exact source at `794851d`; `architecture.md`; decision `D-2026-08-13-02`. |
| Decision enabled | Authorize the frozen P0-1 implementation mission after this documentation branch is independently reviewed. |
| Confidence | 0.96 |

### VAL-2026-08-19-01 — P0-1 exact-SHA implementation and integration validation

| Field | Record |
| --- | --- |
| Question | Did the frozen P0-1 boundary pass exact-SHA implementation, integration, deterministic, browser-security and public-CI validation without unrelated runtime change? |
| Isolated variable | P0-1 implementation only; no retrieval, verifier rubric, tenant model or deployment target change. |
| Control / candidate | Frozen base `7a606e895fe0a4bc9092659f130881bc7b52bd28` / final implementation `20eb17f2737010dbf72eea0f0e271bf47d5af3de`. |
| Frozen gate | Trusted sanitization/rendering; inert Gate 1; empty-sandbox Gate 2; CSP/header consistency; header-only SSE bearer; no token in URL/storage; canonical citation authority and malformed URL rejection; ordered revision content; approval/archive/Git byte equality; exact-SHA race-safe publication; fixed tests with no threshold changes or retry-until-green. |
| Integration identity | `d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3`; parents exact canonical `904f3efe87e6771329b5088bb3afeb6cd16c90dc` and exact implementation `20eb17f2737010dbf72eea0f0e271bf47d5af3de`. Canonical documentation and reviewed implementation blob identity were preserved; no extra runtime change appeared. |
| Deterministic validation | Final implementation: Ruff fatal tier passed; `215 passed`. Exact integration public CI: runs `32143196853` and `32230388649` succeeded; the latter recorded Ruff success and `215 passed, 3 warnings`. |
| Browser/security validation | `4 passed` with pinned Chromium on exact final implementation; implementation blobs are identical in exact integration. Public integration CI installed pinned Chromium and included the browser tests in its `215`-test suite. |
| Paid/live providers | None used for this validation. The PR-only paid evaluation job was skipped on the integration push and is not claimed. |
| Result | Every frozen P0-1 gate passed and exact integration preserved the reviewed implementation. |
| Classification | `P0-1 VALIDATED AND INTEGRATED — CLOSED`. Not deployment proof or overall production readiness. |
| Evidence | Exact SHAs above; [GitHub Actions run `32230388649`](https://github.com/anudeepreddy332/content-agent/actions/runs/32230388649); decision `D-2026-08-19-01`. |
| Decision enabled | Close P0-1; retain its contract as a regression boundary; reassess remaining enterprise blockers separately. |
| Confidence | 0.99 |

### AUD-2026-08-19-01 — Post-P0-1 enterprise-blocker reassessment

| Field | Record |
| --- | --- |
| Question | After P0-1 closure, which remaining failures are independently reproducible and which are capabilities, limitations, debt or unresolved experiments? |
| Isolated variable | Read-only/static traces plus zero-paid deterministic probes at exact canonical main `d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3`; no runtime or threshold change. |
| Deterministic control | Focused evaluation/verifier/failure suite: `45 collected, 45 passed`. |
| P0-2a reproduction | `scripts/benchmark.py --id 999 --gate` selected `0` topics, reported `Valid: 0/0`, printed a CI pass, and exited `0`. Classification: `PROVEN DEFECT` in evaluation scope/cardinality acceptance. |
| P0-2b reproduction | A single `unverified` verdict with confidence `.99` produced grounding score `.99` and routed to HITL. An empty verifier array was recorded as `completed`; at maximum iterations it reached HITL and an approved HITL decision routed to HTML generation. Classification: `PROVEN DEFECT`, mitigated by mandatory human gates. Correct completeness semantics for a production unknown-sized extracted claim set remain `UNKNOWN REQUIRING EXPERIMENT`. |
| Other classifications | Identity/tenant ownership: `MISSING ENTERPRISE CAPABILITY`; volatile run registry around durable checkpoints: `ARCHITECTURAL LIMITATION`; dependency-blind health, single-worker execution and operational observability: `TECHNICAL DEBT / MISSING ENTERPRISE CAPABILITY`; serving MiniLM truncation: `PROVEN DEFECT`, with replacement `UNKNOWN REQUIRING EXPERIMENT`; hosting deployment identity: `MISSING ENTERPRISE CAPABILITY`. P0-1 artifact/push binding remains closed. |
| Ordering decision | Evidence integrity P0-2a precedes verifier-behavior P0-2b. Per the accepted program sequence, the immediate engineering mission is the separate Stage 2 MCP checkpoint; P0-2a remains queued until MCP is independently validated. |
| P0-2a isolated target | Empty/incoherent selection must fail before calls; smoke slices must be explicitly non-release; release mode must bind exact ordered manifest/cardinality, run identities/statuses, code/config/prompt hashes and report digest. No prompts, retrieval, verifier rubric, corpus or quality threshold changes. |
| Kill/block conditions | Block if canonical full scope/cardinality cannot be frozen, any unscorable/incomplete run can earn release pass, scope expands into model/retrieval behavior, or passing needs a moved threshold or retry-until-green. |
| Decision enabled | `NEXT-P0-INVESTIGATION-REQUIRED`; do not authorize Cursor for P0-2 before Stage 2 MCP validation and independent review of the P0-2a frozen contract. |
| Confidence | 0.97 |
