# PROJECT STATUS

_Canonical current-state snapshot. Last synchronized: 2026-08-19._

This file answers what is true now. It is not a history log. Material history remains in
`DECISIONS.md`; experiment detail is indexed in `docs/EXPERIMENT_LEDGER.md`; the old v5 freeze
remains preserved in `FREEZE.md`.

## Authoritative SHA roles

- **Current canonical main:** `d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3`.
  This is the reviewed non-fast-forward integration commit for P0-1. Its parents are exact canonical
  governance state `904f3efe87e6771329b5088bb3afeb6cd16c90dc` and exact final implementation
  `20eb17f2737010dbf72eea0f0e271bf47d5af3de`.
- **Final validated P0-1 implementation:** `20eb17f2737010dbf72eea0f0e271bf47d5af3de`,
  developed from frozen implementation base `7a606e895fe0a4bc9092659f130881bc7b52bd28`.
  No later descendant is implicitly approved.
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
   - **P0-2a — PROVEN DEFECT:** the benchmark gate can select zero topics and emit a passing `0/0`
     result. Release-grade evaluation scope, expected cardinality, and evidence identity therefore
     remain fail-open.
   - **P0-2b — PROVEN DEFECT:** production verifier routing computes grounding from confidence
     independently of verdict status, accepts an unknown-sized or empty extracted verdict set, and
     permits cost/iteration ceilings to reach HITL regardless of verification status. Mandatory
     human gates mitigate unattended publication but do not make the machine acceptance semantics
     sound.
   - **UNKNOWN REQUIRING EXPERIMENT:** the correct production completeness contract for a
     model-extracted, unknown-sized claim set. Known-cardinality fixtures already enforce exact
     counts and must not be confused with production completeness.
   - P0-2a precedes P0-2b because release evidence must fail closed before behavioral experiments
     can earn an acceptance claim.

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

The current engineering mission is the separately governed **Stage 2 MCP architecture and
implementation checkpoint**. P0-1 documentation closeout does not authorize Cursor or another
implementation agent to begin P0-2, and MCP implementation must not silently implement P0-2a.
After MCP is independently validated, P0-2a is intended to be the first substantive Content Agent
engineering mission executed through that validated workflow, unless the Architect proves a
smaller prerequisite mission is required. After P0-2a, the MCP-assisted governance workflow must be
explicitly evaluated based on that real mission. Only after that proof may separate Stage 3 A2A
architecture and implementation begin; A2A does not block starting P0-2a.

The frozen dependency is: **MCP validated → P0-2a executed through MCP → MCP-assisted workflow
evaluated → A2A**.

No P0-2 implementation, merge, publish, or deployment is authorized by this file.

## P0-2a investigation boundary after MCP

- Isolated variable: benchmark scope/cardinality and evidence-binding semantics only.
- Frozen control: current `--id 999 --gate` behavior selects zero topics and exits `0` with a pass.
- Required target: empty or incoherent selection fails before provider calls; smoke slices are
  explicitly non-release; a release gate binds an exact ordered topic manifest and expected count,
  exact run IDs/statuses, prompt/config hashes, code SHA, and report digest.
- Deterministic proof precedes any paid run. No prompts, retrieval, verifier rubric, corpus, or
  quality threshold may change in this mission.
- Block if the canonical full manifest/cardinality cannot be frozen, if any unscorable/incomplete
  run can satisfy a release gate, or if passing requires threshold changes or retry-until-green.

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
- Fresh zero-paid audit on 2026-08-19: `45/45` focused evaluation/verifier/failure tests passed;
  independent probes reproduced the P0-2a zero-topic gate pass and P0-2b status/confidence and empty
  verdict acceptance defects.
- Retrieval evidence and classifications remain indexed in `docs/EXPERIMENT_LEDGER.md`.

## Explicitly not accepted

- Enterprise production readiness, multi-tenancy, HA, unattended autonomy, or verified current
  deployment state.
- A smoke/partial/zero-cardinality benchmark pass as full release evidence.
- Verifier confidence as a substitute for verified status or production claim completeness.
- Naive tokenizer-aligned rechunking, GTE, Jina, or a new fusion design as the serving replacement.
- `N=0`, incomplete, stale, incoherent, or non-final-SHA evidence as a release pass.
- The June 2026 live-demo URL, Docker tag, or deployment state as verified current.
- The proposed S3/event-driven ingestion, Postgres metadata truth, Qdrant-derived-index,
  deterministic-identity/idempotent-ingestion, heterogeneous parsing, structure-aware chunking,
  retrieval-scale, tenant/ACL, and model-tier program as accepted architecture. It remains
  `ROADMAP INPUT — NOT ACCEPTED ARCHITECTURE` for a separate Architect mission.

## Operating rule

No threshold lowering after results, no retry-until-lucky, and no aggregate score may hide a
critical individual regression. A pass at one gate is evidence for that gate only; it is not merge
or cutover approval.
