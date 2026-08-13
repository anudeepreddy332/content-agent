# PROJECT STATUS

_Canonical current-state snapshot. Last synchronized: 2026-08-13._

This file answers what is true now. It is not a history log. Material history remains in
`DECISIONS.md`; experiment detail is indexed in `docs/EXPERIMENT_LEDGER.md`; the old v5 freeze
remains preserved in `FREEZE.md`.

## Authoritative baseline

- Accepted production-code baseline: `794851dded770ce87d111e73735d000e23597eb1` (`origin/main`).
- System shape at that baseline: single-operator LangGraph pipeline with Tavily plus Qdrant/BM25/RRF
  retrieval, DeepSeek drafting/verification/reflection, two human review gates, local Git integration,
  and a separate human-triggered publish endpoint.
- Serving retrieval baseline: the existing `all-MiniLM-L6-v2` dense channel plus BM25, fused with
  RRF `k=60`, over the current serving corpus/chunk contract. This is retained because no replacement
  has passed every required product-quality and integration gate.
- Current branch authorized by this synchronization: `chore/canonical-engineering-state`,
  documentation only, review required, no merge authorized by this file.

## Enterprise-production decision

**BLOCKED.** The repository is a useful supervised single-operator prototype, but the accepted
baseline is not approved for enterprise production, shared public operation, or multi-customer
onboarding.

The prior `docs/PRODUCTION_READINESS.md` conclusion was scoped to a June 2026 supervised demo and is
historical evidence, not a current enterprise-release decision.

## Accepted priority order

1. **P0-1 — Active-content execution and credential exposure boundary**
   - Architecture: **APPROVED and frozen**.
   - Implementation: **not yet validated, not merged, not deployed**.
   - Current code still renders model/web-influenced active content in the same-origin reviewer UI,
     exposes bearer credentials in SSE/preview URLs, trusts model-provided citation URLs, and does
     not bind publication to the exact reviewed artifact and commit.
   - Authoritative implementation contract: `architecture.md`, section "P0-1 accepted target".

2. **P0-2 — Verifier semantics and evaluation integrity**
   - The exact-run telemetry and explicit verification-status correction is merged at the baseline.
   - Still open: verifier failure/empty-verdict states are not consistently enforced before HITL or
     publication; benchmark and telemetry coherence can still false-pass; required evaluation
     provenance is not yet tied to one final integration SHA.

3. **P0-3 — Identity and tenant boundary**
   - `MISSING ENTERPRISE CAPABILITY / ENTERPRISE MULTI-CUSTOMER RELEASE BLOCKER`.
   - This is not classified as a malfunction of the declared single-operator prototype.

4. **P1 — Durable execution and immutable approval/publish binding beyond P0-1**
   - Includes durable run discovery/recovery, concurrency/queue ownership, high availability, and
     deployment identity beyond the exact-artifact/exact-commit binding included in P0-1.

## Current authorized mission

After independent review of this documentation branch, the next product mission is implementation
of the already-approved P0-1 contract. The implementation agent may implement that specification
only; it may not redesign the architecture, relax frozen gates, merge, publish, or deploy without a
new authorization.

## Parallel retrieval research

- Known MiniLM truncation risk remains **OPEN**.
- Tokenizer-aligned `224/32` child chunking removed measured truncation but produced mixed/regressive
  product-quality results; it is **not** an accepted serving fix.
- Exact-73 diagnostics at `b9f9d1d`, `4f6637e`, and `de60706` clarify channel behavior but authorize
  no production cutover.
- Research must remain isolated from serving. Evaluator/fixture repair precedes further quality
  conclusions. No model, chunking, fusion, threshold, corpus, or collection change is accepted.

## Latest accepted evidence

- Independent architecture/audit review on 2026-08-13: P0-1 architecture accepted; current release
  remains blocked.
- Main integration baseline `794851d`: deterministic test/lint evidence exists, while final-SHA live
  evaluation provenance and several fail-closed semantics remain incomplete.
- Retrieval evidence and classifications are indexed in `docs/EXPERIMENT_LEDGER.md`; detailed
  historical reports remain under `docs/archive/` and on their evidence branches.

## Explicitly not accepted

- Current reviewer rendering as safe.
- Current bearer transport as safe.
- Current citation attribution as authoritative.
- Enterprise production readiness, multi-tenancy, HA, or unattended autonomy.
- Naive tokenizer-aligned rechunking, GTE, Jina, or a new fusion design as the serving replacement.
- `N=0`, incomplete, stale, incoherent, or non-final-SHA evaluation evidence as a release pass.
- The June 2026 live-demo URL, Docker tag, or deployment state as verified current.

## Operating rule

No threshold lowering after results, no retry-until-lucky, and no aggregate score may hide a
critical individual regression. A pass at one gate is evidence for that gate only; it is not merge
or cutover approval.
