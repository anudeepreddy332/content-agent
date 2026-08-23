# Experiment and Evidence Ledger

_Canonical compact index. Last synchronized: 2026-08-23._

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

### VAL-2026-08-19-02 — P0-2a exact-SHA implementation validation

| Field | Record |
| --- | --- |
| Question | Does exact P0-2a implementation `0b707e4e431ea7662eec86aec5d4ed18a3c060dd` satisfy the frozen release-evaluator integrity contract, including cleanup of the previously surviving PASS artifact? |
| Isolated variable | Benchmark scope/cardinality, release-evidence identity, trusted finalization, and failure cleanup only; no verifier, prompt/model, retrieval, threshold, corpus, HITL, publication, or P0-1 change. |
| Architecture / base / candidate | Frozen architecture `c8b75c3ab069df29e2201c0540b69bfca86e9cf1`; exact product base `ca29d32b4869269daa47142615d298580a577a77`; final candidate `0b707e4e431ea7662eec86aec5d4ed18a3c060dd`. |
| Scope and ancestry | Linear reviewed history through `d204deb` to `0b707e4`; cumulative changes are exactly `scripts/benchmark.py`, `tests/test_evaluation_integrity.py`, `.github/workflows/eval.yml`, and `evals/benchmark_release_contract.json`. |
| Frozen gate | Exact immutable 20-topic V1; complete ordered valid/scorable units; UVR `<=0.15`; exact prompt/config/GitHub/code/clean-state/contract/manifest identity; coherent aggregates; secret-safe closed evidence; atomic write/reread; external trusted finalization; no failed finalization may leave PASS evidence. |
| Deterministic validation | Frozen sync succeeded; focused `205 passed`; verifier/failure `31 passed`; full `406 passed, 3 warnings`; Ruff fatal tier passed; range diff check passed. |
| Real post-reread attacks | After trusted context construction, PASS write, reread, evidence validation, and trusted persisted-payload comparison succeeded, actual contract and manifest files were independently mutated. Each real loader raised `SystemExit(2)`, which became controlled finalization failure: exit `1`, no PASS print, no PASS or temporary artifact, and one validated sanitized FAIL report. |
| Pre-write and control | Actual contract and manifest loader failures during trusted-context construction both produced controlled sanitized FAIL with no PASS write. With roots unchanged, one trusted PASS survived and PASS printed exactly once. |
| Prior-attack regression / causality | Representative scope/cardinality, V1 mutation/type, result/telemetry, identity, secret, aggregate, coordinated-rewrite, atomic-reread, and runtime-root attacks remained rejected at the intended validator layers. `TEST CAUSALITY: PASS`. |
| Paid/live providers | None. No paid 20-topic release benchmark was run; no provider credential or call was required. |
| Result | Every frozen local implementation and adversarial gate passed at exact `0b707e4`. |
| Classification | `P0-2A IMPLEMENTATION VALIDATED — INTEGRATION PENDING`. Not integration, public integration-SHA CI, deployment proof, post-workflow cryptographic authenticity, P0-2b validation, or overall production readiness. |
| Evidence | Exact SHAs and commands above; decision `D-2026-08-19-04`. |
| Decision enabled | Prepare the exact controlled integration mission after separate human authorization; retain canonical main at `ca29d32` until then. |
| Confidence | 0.99 |

### VAL-2026-08-19-03 — P0-2a canonical integration closeout

| Field | Record |
| --- | --- |
| Question | Did the exact reviewed P0-2a implementation become the canonical runtime/product state through the expected merge parents and exact-integration-SHA public deterministic CI, without provider execution or scope drift? |
| Architecture / implementation / prior closeout | Frozen architecture `c8b75c3ab069df29e2201c0540b69bfca86e9cf1`; validated implementation `0b707e4e431ea7662eec86aec5d4ed18a3c060dd`; prior documentation closeout `174d8c924a35fc5151a4549725db9a01e96f119b`. |
| Canonical integration identity | At integration-closeout verification, direct remote main and fetched `origin/main` resolved `74523ffcfa8906573a72415f1d868dc02996b561`; parents are exact `174d8c9` then exact `0b707e4`. This commit remains the P0-2a runtime/product integration anchor; documentation-only descendants may advance `refs/heads/main` without changing the validated implementation blobs. |
| Deterministic validation | Exact implementation evidence: focused `205 passed`; verifier/failure `31 passed`; full `406 passed, 3 warnings`; Ruff fatal tier PASS; range diff PASS. Exact integration public CI run `32285001516`: event `push`, branch `main`, head SHA exact `74523ff`, overall success, lint success, test success. |
| Paid/live providers | None during integration. The provider-backed `eval-gate` job was skipped. Provider calls/spend: zero. |
| Real release baseline | The paid release-mode frozen-V1 20-topic benchmark has **not** run. No PASS/FAIL baseline, per-topic UVR, mean UVR, grounding, reflection, latency, provider cost, failed/unscorable-unit record, or release evidence digest is claimed. |
| Result | The reviewed P0-2a implementation is the canonical runtime/product state anchored at `74523ff`, and every integration authority matched. `refs/heads/main` must contain that anchor. |
| Classification | `P0-2A VALIDATED AND INTEGRATED — CLOSED`. Evaluator scope/cardinality/release-evidence integrity only; not permanent JSON authenticity, deployment proof, P0-2b validation, or overall production readiness. |
| Evidence | Exact SHAs above; [GitHub Actions run `32285001516`](https://github.com/anudeepreddy332/content-agent/actions/runs/32285001516); decision `D-2026-08-19-05`. |
| Decision enabled | After this docs-only closeout is separately reviewed and integrated, capture the first trusted real frozen-V1 20-topic baseline without code/config/threshold changes or retry-until-green; then begin separate P0-2b architecture. |
| Confidence | 0.99 |

### VAL-2026-08-21-01 — P0-2b slice 1 canonical integration closeout

| Field | Record |
| --- | --- |
| Question | Did exact P0-2b slice 1 become canonical main with matching main-SHA deterministic CI, without retrieval/prompt/model/evaluator/threshold change, without provider execution, and without mutating the immutable BEFORE baseline? |
| Isolated variable | Documentation/status closeout only. No product/runtime change in this record. |
| Canonical integration identity | Direct remote main and fetched `origin/main` resolved exact `230314f7f774ed4b112c377269b190fa1279a004`. Linear ancestry: `eea98c367b0f82fcc844dcca73b3935542adeef6` → `aa90bfc1c4a7d430f0abeb07c84fa0c5416fce70` → `230314f7f774ed4b112c377269b190fa1279a004`. |
| Cumulative slice-1 scope | Exactly `agent/nodes.py`, `config.py`, `tests/test_uvr_fail_closed_routing.py`, `tests/test_failure_injection.py`, `DECISIONS.md`, `PROJECT_STATUS.md`. |
| Integrated contract | UVR-aware fail-closed routing; bounded revision and reverification; ordinary HITL approve cannot grant HTML generation when semantic verification is not accepted. |
| Deterministic validation | Public GitHub Actions run [`32491216346`](https://github.com/anudeepreddy332/content-agent/actions/runs/32491216346): event `push`, branch `main`, head SHA exact `230314f`, lint PASS, deterministic tests PASS, provider-backed `eval-gate` skipped. |
| Immutable BEFORE baseline | GitHub Actions run [`32480353168`](https://github.com/anudeepreddy332/content-agent/actions/runs/32480353168) preserved unchanged: 20/20 complete and scorable; 19/20 at or below UVR 0.15; topic 10 UVR = 5/21 = 0.238095...; overall FAIL; zero benchmark retries. |
| AFTER paid benchmark | Not run. |
| Paid/live providers | None during this closeout. Provider calls/spend: zero. |
| Result | Slice 1 is canonical at `230314f`. Retrieval, prompts, models, evaluator, and quality thresholds unchanged. |
| Classification | `P0-2B SLICE 1 — VALIDATED AND INTEGRATED`. Overall P0-2b remains OPEN. Claim completeness remains unresolved. Not an AFTER benchmark, remaining-P0-2b close, deployment proof, or overall production readiness. |
| Evidence | Exact SHAs above; CI run `32491216346`; BEFORE run `32480353168`; decision `D-2026-08-21-03`. |
| Decision enabled | `ARCHITECTURE FREEZE FOR REMAINING P0-2B SEMANTICS`. Do not start remaining P0-2b implementation from this closeout. |
| Confidence | 0.99 |

### VAL-2026-08-23-01 — P0-2b slice 2A canonical integration closeout

| Field | Record |
| --- | --- |
| Question | Did exact P0-2b slice 2A become canonical main with matching main-SHA deterministic CI, without retrieval/prompt/model/production-verifier/threshold change, without provider execution, and without mutating the immutable BEFORE baseline? |
| Isolated variable | Documentation/status closeout only. No product/runtime change in this record. |
| Canonical integration identity | Direct remote main and fetched `origin/main` resolved exact `bc96009d39394039ca019ec0f4da6358cf14be1d`. Linear ancestry: `a659fe9303626f82b1ca83fedfb5410a436b95d0` → `0a1363f4bd328cd94fd662531780b7e9fa920376` → `bc96009d39394039ca019ec0f4da6358cf14be1d`. |
| Cumulative slice-2A scope | Exactly `docs/P0_2B_CLAIM_SEMANTICS_V1.md`, `evals/claim_semantics_v1.schema.json`, `evals/fixtures/claim_semantics_v1.json`, `scripts/evaluate_claim_semantics.py`, and `tests/test_claim_semantics_evaluator.py`. No production module imported or changed. |
| Integrated contract | Deterministic claim-semantics oracle; 14 frozen fixtures; 18 canonical gold factual atoms; material/full claim recall use independent gold denominators; duplicate/compound/fragment/qualifier-loss gaming detected deterministically; JSON schema authority and executable semantic validation enforced on every official pack load. |
| Deterministic validation | Public GitHub Actions run [`32638253105`](https://github.com/anudeepreddy332/content-agent/actions/runs/32638253105): event `push`, branch `main`, head SHA exact `bc96009`, lint PASS, deterministic tests `467 passed, 4 warnings`, provider-backed `eval-gate` skipped. |
| Metric mathematics / schema authority / test causality | `METRIC MATHEMATICS: PASS`; `SCHEMA AUTHORITY: PASS`; `TEST CAUSALITY: PASS`; status `P0-2B-SLICE2A-INTEGRATION-READY`. |
| Immutable BEFORE baseline | GitHub Actions run [`32480353168`](https://github.com/anudeepreddy332/content-agent/actions/runs/32480353168) preserved unchanged: 20/20 complete and scorable; 19/20 at or below UVR 0.15; topic 10 UVR = 5/21 = 0.238095...; overall FAIL; zero benchmark retries. |
| AFTER paid benchmark | Not run. |
| Paid/live providers | None during integration or this closeout. Provider calls/spend: zero. |
| Result | Slice 2A is canonical at `bc96009`. Production runtime behavior unchanged. Slice 2A is evaluation infrastructure only, not a production completeness guarantee. |
| Classification | `P0-2B SLICE 2A — VALIDATED AND INTEGRATED`. Overall P0-2b remains OPEN. Claim completeness remains unresolved. Not an AFTER benchmark, remaining-P0-2b close, deployment proof, or overall production readiness. |
| Evidence | Exact SHAs above; CI run `32638253105`; BEFORE run `32480353168`; decision `D-2026-08-23-01`. |
| Decision enabled | `P0-2B SLICE 2B — SEMANTIC TRACE PRESERVATION`. Do not start Slice 2B implementation from this closeout. |
| Confidence | 0.99 |

### VAL-2026-08-23-02 — P0-2b slice 2B canonical integration closeout

| Field | Record |
| --- | --- |
| Question | Did exact P0-2b slice 2B become canonical main with matching main-SHA deterministic CI, without retrieval/prompt/model/threshold/benchmark change, without provider execution, and without mutating the immutable BEFORE baseline? |
| Isolated variable | Documentation/status closeout only. No product/runtime change in this record. |
| Canonical integration identity | Direct remote main and fetched `origin/main` resolved exact `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`. Linear ancestry: `e82672936fbb61ee7b1bde7dd3a1ced34f094fa8` → `d3422e4252d6e127603109dd1cb0d6bfaa35a5c0` → `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`. |
| Cumulative slice-2B scope | Exactly `agent/semantic_trace.py`, `agent/nodes.py`, `agent/state.py`, `main.py`, and `tests/test_semantic_trace_v1.py`. |
| Integrated contract | `semantic_trace_v1` preserves exact drafts per iteration; exact verifier-consumed input; raw verifier response; pre/post dedup and attribution state; revision feedback/linkage; final UVR_v1 decision evidence; deterministic hashes; reread/tamper validation; failure-state evidence; and secret-safety boundaries. Production semantic/routing behavior did not change. |
| Deterministic validation | Public GitHub Actions run [`32644879371`](https://github.com/anudeepreddy332/content-agent/actions/runs/32644879371): event `push`, branch `main`, head SHA exact `f6cc5a9`, lint PASS, deterministic tests `497 passed`, provider-backed `eval-gate` skipped. |
| Independent validation | `TRACE RECONSTRUCTABILITY: PASS`; `TRACE INTEGRITY: PASS`; `FAILURE TRACE SEMANTICS: PASS`; `REVISION TRACE CAUSALITY: PASS`; `ROUTING NON-INTERFERENCE: PASS`; `TRACE SECRET SAFETY: PASS`; `LEAN SCOPE: PASS`; status `P0-2B-SLICE2B-INTEGRATION-READY`. |
| Known limitations recorded, not fixed | Runtime trace does not yet bind a verified Git/code SHA; crash telemetry may lack complete mid-run evidence; verifier source_context is already truncated before trace capture; claim completeness remains unknown; Slice-2A claim-semantics oracle is not yet production-connected. Inputs to the principal audit, not a design mandate. |
| Immutable BEFORE baseline | GitHub Actions run [`32480353168`](https://github.com/anudeepreddy332/content-agent/actions/runs/32480353168) preserved unchanged: 20/20 complete and scorable; 19/20 at or below UVR 0.15; topic 10 UVR = 5/21 = 0.238095...; overall FAIL; zero benchmark retries. |
| AFTER paid benchmark | Not run. |
| Paid/live providers | None during integration or this closeout. Provider calls/spend: zero. |
| Result | Slice 2B is canonical at `f6cc5a9`. Production semantic/routing behavior unchanged. Slice 2B is reconstructable trace persistence only, not a production completeness guarantee. No automatic Slice 2C. |
| Classification | `P0-2B SLICE 2B — VALIDATED AND INTEGRATED`. Overall P0-2b remains OPEN. Claim completeness remains unresolved. Not an AFTER benchmark, remaining-P0-2b close, Slice 2C authorization, deployment proof, or overall production readiness. |
| Evidence | Exact SHAs above; CI run `32644879371`; BEFORE run `32480353168`; decision `D-2026-08-23-02`. |
| Decision enabled | `PRINCIPAL TECHNICAL + BUSINESS PROJECT AUDIT`. Decision outcomes before further major implementation: GO, NARROW, PORTFOLIO-CLOSE, PIVOT. Do not start Slice 2C from this closeout. |
| Confidence | 0.99 |
