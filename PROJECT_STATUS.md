# PROJECT STATUS

_Canonical current-state snapshot. Last synchronized: 2026-08-19._

This file answers what is true now. It is not a history log. Material history remains in
`DECISIONS.md`; experiment detail is indexed in `docs/EXPERIMENT_LEDGER.md`; the old v5 freeze
remains preserved in `FREEZE.md`.

## Authoritative SHA roles

- **Current canonical repository state:** the HEAD of `refs/heads/main`. It must contain the exact
  P0-2a integration commit below; documentation-only descendants do not alter the validated P0-2a
  runtime/product implementation blobs.
- **P0-2a canonical integration commit:** `74523ffcfa8906573a72415f1d868dc02996b561`.
  Its parents are exact prior documentation closeout
  `174d8c924a35fc5151a4549725db9a01e96f119b` and exact validated implementation
  `0b707e4e431ea7662eec86aec5d4ed18a3c060dd`.
- **Final validated P0-1 implementation:** `20eb17f2737010dbf72eea0f0e271bf47d5af3de`,
  developed from frozen implementation base `7a606e895fe0a4bc9092659f130881bc7b52bd28`.
  No later descendant is implicitly approved.
- **Frozen P0-2a architecture:** `c8b75c3ab069df29e2201c0540b69bfca86e9cf1`.
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
     evaluation job was skipped. No provider call or spend occurred during integration, and the
     first real paid 20-topic release benchmark has **not** run.
   - **P0-2b — PROVEN DEFECT:** production verifier routing computes grounding from confidence
     independently of verdict status, accepts an unknown-sized or empty extracted verdict set, and
     permits cost/iteration ceilings to reach HITL regardless of verification status. Mandatory
     human gates mitigate unattended publication but do not make the machine acceptance semantics
     sound.
   - **UNKNOWN REQUIRING EXPERIMENT:** the correct production completeness contract for a
     model-extracted, unknown-sized claim set. Known-cardinality fixtures already enforce exact
     counts and must not be confused with production completeness.
   - P0-2a closes evaluator scope, cardinality, and release-evidence integrity only. It does not
     establish permanent cryptographic authenticity for exported JSON or overall production
     readiness. After the documentation closeout is separately reviewed and integrated, the first
     trusted real 20-topic release baseline must be captured before P0-2b begins.

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

## Current authorized mission

P0-2a is validated and integrated at exact runtime/product integration anchor
`74523ffcfa8906573a72415f1d868dc02996b561`. After this documentation-only closeout is separately
reviewed and integrated, the next controlled operational mission is the first trusted real
20-topic V1 release baseline on the then-current HEAD of `refs/heads/main`. P0-2b follows that
baseline as a separately architected verifier-semantics mission.

MCP and A2A are optional developer-workflow infrastructure. They are out of this product mission and
must not block the frozen sequence. This file does not itself authorize implementation, merge,
publish, deployment, or provider spend.

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
  implementation and passed public main-push CI run `32285001516`. No provider-backed gate or paid
  release benchmark ran during integration.

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
  integration: zero. Real paid 20-topic release baseline: not yet run.
- Retrieval evidence and classifications remain indexed in `docs/EXPERIMENT_LEDGER.md`.

## Explicitly not accepted

- Enterprise production readiness, multi-tenancy, HA, unattended autonomy, or verified current
  deployment state.
- A smoke/partial/zero-cardinality benchmark pass as full release evidence.
- Verifier confidence as a substitute for verified status or production claim completeness.
- Naive tokenizer-aligned rechunking, GTE, Jina, or a new fusion design as the serving replacement.
- `N=0`, incomplete, stale, incoherent, or non-final-SHA evidence as a release pass.
- P0-2a closure as proof of a paid full release benchmark, deployment, permanent cryptographic
  authenticity of exported JSON, P0-2b correctness, or overall product production readiness.
- The June 2026 live-demo URL, Docker tag, or deployment state as verified current.
- The proposed S3/event-driven ingestion, Postgres metadata truth, Qdrant-derived-index,
  deterministic-identity/idempotent-ingestion, heterogeneous parsing, structure-aware chunking,
  retrieval-scale, tenant/ACL, and model-tier program as accepted architecture. It remains
  `ROADMAP INPUT — NOT ACCEPTED ARCHITECTURE` for a separate Architect mission.

## Operating rule

No threshold lowering after results, no retry-until-lucky, and no aggregate score may hide a
critical individual regression. A pass at one gate is evidence for that gate only; it is not merge
or cutover approval.
