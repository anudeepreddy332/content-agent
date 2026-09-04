# PROJECT STATUS

_Canonical current-state snapshot. Last synchronized: 2026-09-04._

This file answers what is true now. It is not a history log. Material history remains in
`DECISIONS.md`; experiment detail is indexed in `docs/EXPERIMENT_LEDGER.md`; the old v5 freeze
remains preserved in `FREEZE.md`.

## Authoritative SHA roles

- **Current canonical repository state:** the HEAD of `refs/heads/main`. Before
  client-demo-fixes integration, exact `e4b39caa37497c1743e33843c464ea4381f166d4`
  (demo publish-target guard). Client-demo fixes closeout merges from
  `feature/client-demo-rehearsal-fixes`.
- **P0-2b slice 2B canonical integration commit:**
  `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`. Linear ancestry is exact
  `e82672936fbb61ee7b1bde7dd3a1ced34f094fa8` →
  `d3422e4252d6e127603109dd1cb0d6bfaa35a5c0` →
  `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`.
- **P0-2b slice 2A canonical integration commit:**
  `bc96009d39394039ca019ec0f4da6358cf14be1d`. Linear ancestry is exact
  `a659fe9303626f82b1ca83fedfb5410a436b95d0` →
  `0a1363f4bd328cd94fd662531780b7e9fa920376` →
  `bc96009d39394039ca019ec0f4da6358cf14be1d`.
- **P0-2b slice 1 canonical integration commit:**
  `230314f7f774ed4b112c377269b190fa1279a004`. Linear ancestry is exact
  `eea98c367b0f82fcc844dcca73b3935542adeef6` →
  `aa90bfc1c4a7d430f0abeb07c84fa0c5416fce70` →
  `230314f7f774ed4b112c377269b190fa1279a004`.
- **P0-2a canonical integration commit:** `74523ffcfa8906573a72415f1d868dc02996b561`.
  Its parents are exact prior documentation closeout
  `174d8c924a35fc5151a4549725db9a01e96f119b` and exact validated implementation
  `0b707e4e431ea7662eec86aec5d4ed18a3c060dd`.
- **Final validated P0-1 implementation:** `20eb17f2737010dbf72eea0f0e271bf47d5af3de`,
  developed from frozen implementation base `7a606e895fe0a4bc9092659f130881bc7b52bd28`.
  No later descendant is implicitly approved.
- **Frozen P0-2a architecture:** `c8b75c3ab069df29e2201c0540b69bfca86e9cf1`.
- **Semantic P0 F-01 callable validation boundary — CLOSED:** exact
  `8530b078837b1a8669433777c1f8d9a1add25a8a`. Strict schema enforcement at every
  callable boundary; malformed materiality fails closed.
- **Semantic P0 Slice 1 — VALIDATED AND INTEGRATED — CLOSED:** exact canonical
  integration merge `5f4163f2aa53155216342d20e627abd88fb60a1e` (PR #7). Validated
  implementation contained by that merge: `c866de29ecc11156d28dffddd069a97b3008ca28`.
  Merged PR head: `51d88091da354bedb612f19147adc88b72525541`. Linear ancestry
  includes F-01 correction `8530b078837b1a8669433777c1f8d9a1add25a8a`, F-02
  implementation `650e52f5ecd4554c1c435a5cc99920bf42276ad8`, and manifest-boundary
  correction `c866de29ecc11156d28dffddd069a97b3008ca28`. F-01 CLOSED; F-02
  required-vs-final semantic safety validated; approved qualification
  identity/callable boundary validated; no remaining P0/P1 finding in the bounded
  final qualification; merged-main smoke validation passed. Production runtime
  semantic behavior remains unchanged. Do not reopen without new material P0/P1
  evidence.
- **P0-2b slice 2B — VALIDATED AND INTEGRATED:** exact canonical main
  `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`. Reconstructable semantic-trace persistence only;
  production semantic/routing behavior unchanged. Overall P0-2b remains OPEN. No later
  descendant is implicitly validated for remaining P0-2b semantics.
- **P0-2b slice 2A — VALIDATED AND INTEGRATED:** exact canonical main
  `bc96009d39394039ca019ec0f4da6358cf14be1d`. Evaluation infrastructure only; production runtime
  unchanged. Overall P0-2b remains OPEN. No later descendant is implicitly validated for remaining
  P0-2b semantics.
- **P0-2b slice 1 — VALIDATED AND INTEGRATED:** exact integration commit
  `230314f7f774ed4b112c377269b190fa1279a004`. Overall P0-2b remains OPEN. No later descendant
  is implicitly validated for remaining P0-2b semantics.
- **P0-2a — VALIDATED AND INTEGRATED — CLOSED:** exact implementation
  `0b707e4e431ea7662eec86aec5d4ed18a3c060dd` is integrated by exact merge
  `74523ffcfa8906573a72415f1d868dc02996b561`. No later descendant is implicitly validated.
- **Audited runtime reference:** `794851dded770ce87d111e73735d000e23597eb1`. This remains the
  historical snapshot against which the original P0-1 threat boundary was established.
- System shape at current main: supervised single-operator LangGraph pipeline with Tavily plus
  Qdrant/BM25/RRF retrieval, DeepSeek drafting/verification/reflection, two human review gates,
  local Git integration, and a separate human-triggered publish endpoint.
- Serving retrieval baseline: the existing `all-MiniLM-L6-v2` dense channel plus BM25, fused with
  RRF `k=60`, over the current serving corpus/chunk contract. No replacement has passed every
  required product-quality and integration gate.

## Enterprise-production decision

**BLOCKED.** P0-1 is closed, but the repository remains a supervised single-operator prototype and
is not approved for enterprise production, shared public operation, unattended autonomy, or
multi-customer onboarding.

The prior `docs/PRODUCTION_READINESS.md` conclusion was scoped to a June 2026 supervised demo and is
historical evidence, not a current enterprise-release decision.

## Accepted priority order

The sequence below supersedes the earlier P0-1/P0-2/P0-3 milestone ordering for
**current engineering work**. Historical P0-1/P0-2a/P0-2b slice boundaries
remain preserved below as evidence; they are not reopened here.

### High-impact engineering priority order

1. **Semantic P0 Slice 1 — VALIDATED AND INTEGRATED — CLOSED** at exact canonical
   integration merge `5f4163f2aa53155216342d20e627abd88fb60a1e` (PR #7).
2. **Minimal exact evidence-exposure completion + small real provider 2C/2D
   qualification using the corrected semantic ruler** — **current authorized
   engineering mission.**
3. **Retrieval redesign — major product-quality program:**
   - resolve MiniLM truncation / embedding-input mismatch;
   - production-shaped chunking/embedding strategy;
   - exact evidence recall/exposure;
   - deterministic fusion/ranking;
   - evaluate real query transformation only if it causally improves final
     verifier-visible evidence.
4. **Revision safety qualification:**
   - targeted bad claims resolved;
   - previously verified material claims retained;
   - no new unresolved material claims introduced.
5. **Wire qualified semantic policy into production runtime routing.**
6. **Enterprise capabilities:**
   - identity / ACL / tenancy;
   - durability / recovery / observability;
   - production/cloud deployment hardening.

Retrieval is intentionally prioritized ahead of revision/runtime/enterprise
infrastructure **after** the minimum semantic/provider measurement foundation
is trustworthy. This ordering is impact-driven and intended to avoid
over-engineering. P2/P3/non-blocking improvements are deferred unless they
become prerequisites for a P0/P1 item. AWS/cloud work is not the current
priority.

### Historical milestone record (preserved)

1. **P0-1 — Active-content execution and credential exposure boundary: VALIDATED AND INTEGRATED**
   - Closed at exact canonical main `d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3`.
   - The implemented boundary includes one trusted renderer/sanitizer, inert Gate-1 Markdown,
     empty-sandbox Gate-2 review, restrictive FastAPI/Caddy/article CSP, Authorization-header SSE,
     no bearer in URLs or browser storage, canonical `source_ref` citation authority with public
     HTTPS URL validation, ordered revision-content preservation, approval/hash/archive/Git byte
     equivalence, and exact-SHA publication with remote-parent race protection.
   - Exact integration-SHA public CI passed Ruff and `215` tests. The pinned Chromium security suite
     passed `4` tests on the exact final implementation; those implementation blobs are unchanged in
     the integration commit.
   - This closes the frozen P0-1 implementation contract only. It is not proof of hosting-provider
     deployment identity or overall production readiness.

2. **P0-2 — Evaluation acceptance integrity, then verifier acceptance semantics**
   - **P0-2a — VALIDATED AND INTEGRATED — CLOSED:** exact implementation
     `0b707e4e431ea7662eec86aec5d4ed18a3c060dd` passed the frozen deterministic and adversarial
     acceptance gates. Zero, partial, duplicate, incomplete, unscorable, identity-drifted,
     secret-bearing, or failed-finalization evidence cannot earn or retain a release PASS in the
     validated implementation.
   - Exact reviewed merge `74523ffcfa8906573a72415f1d868dc02996b561` is the canonical P0-2a
     runtime/product integration anchor and must be contained by `refs/heads/main`. Public push CI
     run [`32285001516`](https://github.com/anudeepreddy332/content-agent/actions/runs/32285001516)
     is bound to that SHA and passed its deterministic lint and test jobs; the provider-backed
     evaluation job was skipped. No provider call or spend occurred during P0-2a integration.
     The first trusted real 20-topic BEFORE baseline was later captured as immutable GitHub
     Actions run [`32480353168`](https://github.com/anudeepreddy332/content-agent/actions/runs/32480353168)
     and is preserved unchanged. No AFTER paid benchmark has run after P0-2b slice 2B.
   - **P0-2b slice 2B — VALIDATED AND INTEGRATED:** `semantic_trace_v1` reconstructable
     evidence persistence integrated at exact `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`.
     Exact drafts per iteration, exact verifier-consumed input, raw verifier response,
     pre/post dedup and attribution state, revision feedback/linkage, final UVR_v1 decision
     evidence, deterministic hashes, reread/tamper validation, failure-state evidence, and
     secret-safety boundaries are preserved. Production semantic/routing behavior is
     unchanged. Public main-push CI run
     [`32644879371`](https://github.com/anudeepreddy332/content-agent/actions/runs/32644879371)
     passed deterministic lint and `497` tests; the provider-backed evaluation job was skipped.
   - **P0-2b slice 2A — VALIDATED AND INTEGRATED:** deterministic claim-semantics oracle
     integrated at exact `bc96009d39394039ca019ec0f4da6358cf14be1d`. Fourteen frozen fixtures;
     18 canonical gold factual atoms; material/full claim recall use independent gold
     denominators; duplicate/compound/fragment/qualifier-loss gaming is detected
     deterministically; JSON schema authority and executable semantic validation are enforced.
     Production runtime behavior is unchanged. Public main-push CI run
     [`32638253105`](https://github.com/anudeepreddy332/content-agent/actions/runs/32638253105)
     passed deterministic lint and `467` tests; the provider-backed evaluation job was skipped.
   - **P0-2b slice 1 — VALIDATED AND INTEGRATED:** UVR-aware fail-closed routing, bounded
     revision and reverification, and the Gate-1 HTML-eligibility guard are integrated at exact
     `230314f7f774ed4b112c377269b190fa1279a004`. Ordinary HITL approve cannot grant HTML
     generation when semantic verification is not accepted. Public main-push CI run
     [`32491216346`](https://github.com/anudeepreddy332/content-agent/actions/runs/32491216346)
     passed deterministic lint and tests; the provider-backed evaluation job was skipped.
   - **P0-2b overall — OPEN.** Claim completeness for a production unknown-sized extracted
     claim set remains unresolved. Slice 2A is evaluation infrastructure only, not a production
     completeness guarantee. Slice 2B is reconstructable trace persistence only, not a
     completeness guarantee. Slice 1, Slice 2A, and Slice 2B do not close remaining P0-2b
     semantics. No automatic Slice 2C.
   - **UNKNOWN REQUIRING EXPERIMENT:** the correct production completeness contract for a
     model-extracted, unknown-sized claim set. Known-cardinality fixtures already enforce exact
     counts and must not be confused with production completeness.
   - P0-2a closes evaluator scope, cardinality, and release-evidence integrity only. Slice 1
     closes only the recorded routing/HITL-eligibility contract. Neither establishes overall
     production readiness. Historical pre-slice-1 routing-defect evidence remains in the ledger.

3. **P0-3 — Identity and tenant boundary**
   - `MISSING ENTERPRISE CAPABILITY / ENTERPRISE MULTI-CUSTOMER RELEASE BLOCKER`.
   - One global bearer and no principal-to-run ownership are consistent with the declared
     single-operator prototype, not an acceptable multi-customer authorization model.

4. **P1 — Durable recovery, readiness and deployment identity**
   - Volatile run discovery around durable checkpoints is an `ARCHITECTURAL LIMITATION`; restart can
     make resumable runs undiscoverable.
   - Single-worker execution, dependency-blind `/health`, HA/scale, and hosting-provider deployment
     identity remain `MISSING ENTERPRISE CAPABILITY` or `TECHNICAL DEBT`, as applicable.
   - P0-1 exact artifact/commit/push binding is closed and must not be reopened as this later
     deployment-identity work.

## Current status

- **Client-demo hardening:** CLOSED / LIVE-DEMO-REHEARSED.
- **Overall P0-2b:** OPEN.
- **Semantic P0 Slice 1 — VALIDATED AND INTEGRATED — CLOSED:** exact canonical
  integration merge `5f4163f2aa53155216342d20e627abd88fb60a1e` (PR #7).
- **F-01 callable validation-boundary defect:** CLOSED at
  `8530b078837b1a8669433777c1f8d9a1add25a8a`.
- **F-02 required-vs-final semantic contract:** validated; approved qualification
  identity/callable boundary validated; no remaining P0/P1 finding in the bounded
  final qualification; merged-main smoke validation passed (focused v2 `73 passed`,
  combined v1/v2 `107 passed`, Ruff PASS).
- **Paid 20-topic benchmark:** BLOCKED.
- **Production runtime semantic behavior:** unchanged.
- **Reopen rule:** Semantic P0 Slice 1 must not be reopened without new material
  P0/P1 evidence.

## Current authorized mission

**Minimal exact evidence-exposure completion + small real provider 2C/2D
qualification using the corrected semantic ruler** is the current authorized
engineering mission. Retrieval redesign remains the next major engineering program
(frozen six-step roadmap priority 3).

Immutable BEFORE baseline remains GitHub Actions run `32480353168` (unchanged).
Claim completeness remains unresolved and is not an acceptance condition.
Overall P0-2b remains OPEN. No paid AFTER benchmark is authorized. No large
re-evaluation is authorized.

MCP and A2A are optional developer-workflow infrastructure. They are out of this
product mission and must not block the frozen sequence. This file does not itself
authorize implementation, merge, publish, deployment, or provider spend.

## Historical metrics remain valid diagnostics

Existing V/W/U counts, UVR_v1, grounding scores, semantic trace, revision
telemetry, and previous provider runs remain legitimate historical diagnostics.
They are **not** discarded and prior evaluation work is **not** implied invalid.

The current semantic program exists because those historical metrics do not
independently detect all of:

- omitted material claims;
- material weak claims that old UVR can false-green;
- unsupported material claims introduced into final content.

## Evidence exposure distinction

- LangGraph state / retrieval state is **not** identical to exact model-visible
  prompt context.
- Semantic trace currently provides strong verifier-consumed-context evidence.
- Exact draft-visible evidence/exposure remains part of the next bounded
  qualification (priority 2 above).
- MiniLM embedding truncation and downstream prompt/source-context truncation
  are separate failure modes.

## Evidence-exposure truncation — OPEN / REQUIRES CAUSAL VALIDATION

Current runtime source-context construction deliberately truncates retrieved
source text before draft/verifier exposure:

- Tavily/web content: first **1500** characters per source
- KB content: first **2000** characters per result

This is distinct from MiniLM embedding-input truncation. `_build_source_context()`
in `agent/nodes.py` applies these Phase-1 experimental limits and feeds the
same builder to draft/verification grounding paths.

**Known architectural risk:** A relevant source may be successfully retrieved
while decisive supporting, qualifying, or contradicting evidence lies outside the
exposed prefix. In that case the draft/verifier can make an incorrect grounding
judgment despite retrieval having succeeded.

**Frozen hypothesis to test later:**

> Fixed prefix truncation materially reduces claim-level evidence sufficiency
> and/or causes false weak, false unverified, or false verified decisions
> compared with a claim-aware evidence-exposure strategy.

Important constraints:

- Removing all truncation is **not** already the accepted solution.
- Sending every entire document to the verifier is **not** preselected.
- The future experiment must compare alternatives causally.
- Candidate solutions may include claim-aware evidence spans, surrounding
  context, batching, or larger complete exposure where context permits.
- Silent prefix truncation must **not** be considered enterprise-qualified until
  tested.

LangGraph state containing a source does **not** prove the draft/verifier
actually saw the relevant part of that source. Semantic trace gives strong
evidence of verifier-consumed context, but the captured context may already
contain upstream truncation.

**Status:** OPEN — scheduled after Semantic P0 closure and the minimum
evidence/provider qualification foundation, as part of the retrieval/evidence
redesign program (priority 3). No runtime change authorized by this record.

## Client-demo fixes closed boundary

- **Status:** `CLIENT-DEMO-FIXES: CLOSED`; `LIVE-DEMO-REHEARSED`.
- **Validated branch:** `feature/client-demo-rehearsal-fixes` at implementation HEAD
  `a44416e46f90159f003f87a62d4c1c33987a9c9f` (docs child `66df1ee526cc2e7096f2b8c04c7221ce5068e27a`).
- **HTML policy:** `p0-1-v2`; historical `p0-1-v1` is not silently redefined.
- **Rehearsal chronology (historical failures preserved):**
  1. Run `fd0cb6a3-781f-422c-ade4-3b1e49c04ed7` — **FAILED** before Gate 2: HTML delimiter
     false-positive on code examples (historical raw-substring policy).
  2. Run `1e68d0ee-546e-4a25-b459-1340a9f22b2a` — HTML and both gates **succeeded**;
     publication **correctly failed closed** because local demo-fork `main` did not equal the
     approved remote parent.
  3. Run `2c91cc9e-766d-4921-a6c2-d37253f76543` — **successful human rehearsal:**
     - Topic: LangGraph State Machines
     - V/W/U/total: `32 / 0 / 0 / 32`; grounding: `0.916`; reflection: `7/10`
     - Reflection provenance: real judge; `provider_called=true`; `parse_status=ok`
     - HTML policy: `p0-1-v2`; `policy_diagnostics=[]`
     - Gate 1: human approved; Gate 2: human approved
     - Artifact SHA256: `87af3a25133257918b68833058db02c2abebc55ed9e1dca01f010ef9687c63d4`
     - Approved remote parent: `7914833b1dea16789ac82f22be1fcc18e8bb965c`
     - Resulting demo-fork main: `7b6de76d0fdb0b78440f4211eb6c98cd8a7a6b73`
     - Final remote-parent race check passed; publication succeeded to approved demo fork
     - Demo URL `https://tmw-demo-site.netlify.app` returned HTTP 200
     - Production repository remained untouched
- **Non-blocking backlog (future hardening, not demo blockers):**
  - Successful run telemetry did not stamp immutable Content Agent code identity
    (`code_identity.available=false`). Branch/SHA is strongly supported by workspace state but
    not independently proven from run telemetry.
  - Final 32/32-verified run did not exercise Weak/Unverified UI filters; this does not reopen
    the demo.

## P0-2a closed implementation boundary

- Isolated variable: benchmark scope/cardinality and evidence-binding semantics only.
- Historical frozen control at `ca29d32`: `--id 999 --gate` selected zero topics and exited `0`
  with a pass.
- Required target: empty or incoherent selection fails before provider calls; smoke slices are
  explicitly non-release; a release gate binds an exact ordered topic manifest and expected count,
  exact run IDs/statuses, prompt/config hashes, code SHA before and after execution, main-branch CI
  ref, and report digest. The exact 20-topic fixture is immutable V1, not a permanent product count;
  future fixture evolution requires a separately reviewed successor contract.
- Deterministic proof precedes any paid run. No prompts, retrieval, verifier rubric, corpus, or
  quality threshold may change in this mission.
- Block if the canonical full manifest/cardinality cannot be frozen, if any unscorable/incomplete
  run can satisfy a release gate, or if passing requires threshold changes or retry-until-green.
- Validation at exact `0b707e4`: `205 passed` focused evaluation-integrity tests; `31 passed`
  verifier/failure regressions; `406 passed, 3 warnings` full suite; Ruff fatal tier and range
  `git diff --check` passed.
- Independent real-file attacks changed the contract and manifest only after PASS write, reread,
  schema/digest/secret validation, and trusted persisted-payload comparison. Loader-originated
  `SystemExit(2)` was converted to controlled failure; each path exited `1`, printed no release
  PASS, retained one validated sanitized FAIL report, and left no PASS or temporary artifact.
  Equivalent pre-write contract and manifest failures were also controlled. Unchanged roots left
  one trusted PASS and printed PASS exactly once.
- Integration at exact `74523ffcfa8906573a72415f1d868dc02996b561` preserves the reviewed
  implementation and passed public main-push CI run `32285001516`. No provider-backed gate ran
  during P0-2a integration.

## P0-2b slice 1 integrated boundary

- Isolated variable: production routing after verify/reflect and Gate-1 HTML eligibility only.
- Exact integration: `230314f7f774ed4b112c377269b190fa1279a004`.
- Cumulative slice-1 scope is exactly `agent/nodes.py`, `config.py`,
  `tests/test_uvr_fail_closed_routing.py`, `tests/test_failure_injection.py`, `DECISIONS.md`,
  and `PROJECT_STATUS.md`.
- Integrated contract: UVR-aware fail-closed routing; bounded revision and reverification;
  ordinary HITL approve cannot grant HTML generation when semantic verification is not accepted.
- Public main-push CI run `32491216346` passed deterministic lint/test and skipped provider eval.
- Retrieval, prompts, models, evaluator, and quality thresholds are unchanged. `UVR_THRESHOLD`
  names the frozen `0.15` gate; it is not a moved threshold.
- Immutable BEFORE baseline `32480353168` is preserved unchanged. No AFTER paid benchmark has
  run. Claim completeness remains unresolved. Overall P0-2b remains OPEN.

## P0-2b slice 2A integrated boundary

- Isolated variable: deterministic claim-semantics evaluation infrastructure only.
- Exact integration: `bc96009d39394039ca019ec0f4da6358cf14be1d`.
- Ancestry: `a659fe9303626f82b1ca83fedfb5410a436b95d0` →
  `0a1363f4bd328cd94fd662531780b7e9fa920376` →
  `bc96009d39394039ca019ec0f4da6358cf14be1d`.
- Cumulative slice-2A scope is exactly `docs/P0_2B_CLAIM_SEMANTICS_V1.md`,
  `evals/claim_semantics_v1.schema.json`, `evals/fixtures/claim_semantics_v1.json`,
  `scripts/evaluate_claim_semantics.py`, and `tests/test_claim_semantics_evaluator.py`.
- Integrated contract: deterministic claim-semantics oracle; 14 frozen fixtures; 18 canonical
  gold factual atoms; material/full claim recall use independent gold denominators;
  duplicate/compound/fragment/qualifier-loss gaming detected deterministically; JSON schema
  authority (`DEFAULT_SCHEMA`) and executable semantic validation enforced on every official
  pack load. No production module is imported or changed.
- Public main-push CI run `32638253105` passed deterministic lint/test (`467 passed`) and
  skipped provider eval. Provider calls/spend during integration: zero.
- Slice 2A is evaluation infrastructure only, not a production completeness guarantee.
  Retrieval, prompts, models, production verifier, evaluator release gate, and quality
  thresholds are unchanged. Immutable BEFORE baseline `32480353168` is preserved unchanged.
  No AFTER paid benchmark has run. Overall P0-2b remains OPEN.

## P0-2b slice 2B integrated boundary

- Isolated variable: reconstructable semantic-trace persistence only. Production
  semantic/routing behavior did not change.
- Exact integration: `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`.
- Ancestry: `e82672936fbb61ee7b1bde7dd3a1ced34f094fa8` →
  `d3422e4252d6e127603109dd1cb0d6bfaa35a5c0` →
  `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`.
- Cumulative slice-2B scope is exactly `agent/semantic_trace.py`, `agent/nodes.py`,
  `agent/state.py`, `main.py`, and `tests/test_semantic_trace_v1.py`.
- Integrated contract: `semantic_trace_v1` preserves exact drafts per iteration; exact
  verifier-consumed input; raw verifier response; pre/post dedup and attribution state;
  revision feedback/linkage; final UVR_v1 decision evidence; deterministic hashes;
  reread/tamper validation; failure-state evidence; and secret-safety boundaries.
- Public main-push CI run `32644879371` passed deterministic lint/test (`497 passed`)
  and skipped provider eval. Provider calls/spend during integration: zero.
- Retrieval, prompts, models, production verifier rubric, evaluator release gate, and
  quality thresholds are unchanged. Immutable BEFORE baseline `32480353168` is
  preserved unchanged. No AFTER paid benchmark has run. Overall P0-2b remains OPEN.
- Recorded, not fixed, as inputs to the principal audit: runtime trace does not yet
  bind a verified Git/code SHA; crash telemetry may lack complete mid-run evidence;
  verifier source_context is already truncated before trace capture; claim completeness
  remains unknown; Slice-2A claim-semantics oracle is not yet production-connected.

## Parallel retrieval research

- Known MiniLM truncation risk remains **OPEN / PROVEN DEFECT**; replacement architecture remains
  **UNKNOWN REQUIRING EXPERIMENT**.
- Tokenizer-aligned `224/32` child chunking removed measured truncation but produced mixed/regressive
  product-quality results; it is not an accepted serving fix.
- Exact-73 diagnostics at `b9f9d1d`, `4f6637e`, and `de60706` clarify channel behavior but authorize
  no production cutover.
- Research remains isolated from serving. Evaluation acceptance integrity precedes further quality
  conclusions. No model, chunking, fusion, threshold, corpus, or collection change is accepted.

## Latest accepted evidence

- Exact P0-1 implementation `20eb17f2737010dbf72eea0f0e271bf47d5af3de`: Ruff fatal tier,
  `215 passed` deterministic tests, and `4 passed` pinned-Chromium security tests.
- Exact integration `d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3`: two public push CI runs
  succeeded (`32143196853`, `32230388649`); [run `32230388649`](https://github.com/anudeepreddy332/content-agent/actions/runs/32230388649)
  recorded Ruff success and `215 passed, 3 warnings`. The PR-only paid evaluation job was correctly
  skipped on push.
- Fresh zero-paid audit on exact `ca29d32` on 2026-08-19: `45/45` focused tests and `215/215` full
  deterministic/browser tests passed. Independent probes reproduced the P0-2a zero/partial/
  duplicate scope passes and P0-2b status/confidence, empty verdict, ceiling, and approved-HITL
  routing defects.
- Exact P0-2a implementation `0b707e4e431ea7662eec86aec5d4ed18a3c060dd`: `205` focused,
  `31` verifier/failure, and `406` full tests passed; Ruff and range diff checks passed. Independent
  unchanged, real post-reread contract/manifest drift, and pre-write loader-failure probes passed
  their frozen causal gates. No provider or paid release benchmark was run.
- Exact P0-2a integration `74523ffcfa8906573a72415f1d868dc02996b561`: parents are exact prior
  closeout `174d8c9` and validated implementation `0b707e4`; public main-push CI run `32285001516`
  passed lint/test and skipped the provider-backed evaluation gate. Provider calls/spend during
  P0-2a integration: zero.
- Immutable BEFORE 20-topic release baseline GitHub Actions run `32480353168`: 20/20 complete
  and scorable; 19/20 at or below UVR 0.15; topic 10 UVR = 5/21 = 0.238095...; overall FAIL;
  zero benchmark retries. Preserved unchanged through P0-2b slice 1. No AFTER paid benchmark.
- Exact P0-2b slice 2B integration `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`: ancestry
  `e826729` → `d3422e4` → `f6cc5a9`; public main-push CI run `32644879371` passed lint/test
  (`497 passed, 4 warnings`) and skipped provider eval. Provider calls/spend during
  integration: zero. Production semantic/routing behavior unchanged.
- Exact P0-2b slice 2A integration `bc96009d39394039ca019ec0f4da6358cf14be1d`: ancestry
  `a659fe9` → `0a1363f` → `bc96009`; public main-push CI run `32638253105` passed lint/test
  (`467 passed, 4 warnings`) and skipped provider eval. Provider calls/spend during integration:
  zero. Production runtime unchanged.
- Exact P0-2b slice 1 integration `230314f7f774ed4b112c377269b190fa1279a004`: ancestry
  `eea98c3` → `aa90bfc` → `230314f`; public main-push CI run `32491216346` passed lint/test
  and skipped provider eval. Provider calls/spend during this documentation closeout: zero.
- Retrieval evidence and classifications remain indexed in `docs/EXPERIMENT_LEDGER.md`.

## Explicitly not accepted

- Enterprise production readiness, multi-tenancy, HA, unattended autonomy, or verified current
  deployment state.
- A smoke/partial/zero-cardinality benchmark pass as full release evidence.
- Verifier confidence as a substitute for verified status or production claim completeness.
- Naive tokenizer-aligned rechunking, GTE, Jina, or a new fusion design as the serving replacement.
- `N=0`, incomplete, stale, incoherent, or non-final-SHA evidence as a release pass.
- P0-2a closure as proof of deployment, permanent cryptographic authenticity of exported JSON,
  remaining P0-2b correctness, or overall product production readiness.
- P0-2b slice 2B integration as a production completeness guarantee, as closure of overall
  P0-2b, as an AFTER paid benchmark, as Slice 2C authorization, or as enterprise production
  readiness.
- P0-2b slice 2A integration as a production completeness guarantee, as closure of overall
  P0-2b, as an AFTER paid benchmark, or as enterprise production readiness.
- P0-2b slice 1 integration as closure of overall P0-2b, as an AFTER paid benchmark, as a
  claim-completeness contract, or as enterprise production readiness.
- The June 2026 live-demo URL, Docker tag, or deployment state as verified current.
- The proposed S3/event-driven ingestion, Postgres metadata truth, Qdrant-derived-index,
  deterministic-identity/idempotent-ingestion, heterogeneous parsing, structure-aware chunking,
  retrieval-scale, tenant/ACL, and model-tier program as accepted architecture. It remains
  `ROADMAP INPUT — NOT ACCEPTED ARCHITECTURE` for a separate Architect mission.

## Operating rule

No threshold lowering after results, no retry-until-lucky, and no aggregate score may hide a
critical individual regression. A pass at one gate is evidence for that gate only; it is not merge
or cutover approval.
