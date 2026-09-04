# DECISIONS LOG

Purpose: Prevent future chats from re-litigating solved problems.
Rules: Never delete old decisions. Rejected ideas stay. Consult before proposing architecture changes.

New material decisions use stable IDs and, where applicable, record the question, evidence, exact
SHA/branch/artifact, alternatives, reasoning, status, confidence, and supersession relationship.
Older entries are preserved in their original format; later evidence supersedes their current
conclusion without rewriting their history.

---
Decision ID: D-2026-09-04-01
Date: 2026-09-04

Decision: Adopt impact-driven post-Semantic-P0 priority sequence:
semantic closure → minimal exposure/provider qualification → retrieval redesign →
revision safety → runtime semantic wiring → enterprise infrastructure/cloud.

Reason: A trustworthy semantic ruler is required before comparing systems, but
after that minimum foundation, retrieval is the largest remaining product-quality
risk due to known embedding truncation/exposure issues. Revision/runtime/cloud
work must not delay retrieval without new P0/P1 evidence.

Priority order (impact-driven; P2/P3 deferred unless prerequisite):

1. Finish Semantic P0 Slice 1 adversarial qualification; integrate only if clean.
2. Minimal exact evidence-exposure completion + small real provider 2C/2D
   qualification using the corrected semantic ruler.
3. Retrieval redesign (MiniLM truncation, chunking/embedding, exact evidence
   recall/exposure, fusion/ranking; query transformation only if causally
   improves verifier-visible evidence).
4. Revision safety qualification (bad claims resolved; verified material retained;
   no new unresolved material claims).
5. Wire qualified semantic policy into production runtime routing.
6. Enterprise capabilities (identity/ACL/tenancy; durability/recovery/observability;
   production/cloud hardening).

Also recorded:

- Existing historical metrics (V/W/U, UVR_v1, grounding, semantic trace, revision
  telemetry, prior provider runs) remain valid diagnostics; not discarded.
- No large re-evaluation is authorized.
- No paid benchmark is authorized.
- Low-impact P2/P3 work is deferred unless it becomes a prerequisite for P0/P1.
- AWS/cloud work is not the current priority.
- F-02 checkpoint `650e52f5ecd4554c1c435a5cc99920bf42276ad8` is implemented but
  **NOT** adversarially closed; do not claim F-02 closed until adversarial
  validation is complete.

Question: What is the authorized engineering priority order after the Semantic P0
metric-registry / claim-semantics v2 program establishes a trustworthy semantic
ruler?

Evidence: Principal Review of F-02 checkpoint `650e52f5` on
`feature/semantic-p0-metric-registry-v2`; F-01 closed at `8530b078`; client-demo
CLOSED/LIVE-DEMO-REHEARSED; overall P0-2b OPEN; paid 20-topic benchmark BLOCKED;
known MiniLM truncation OPEN; semantic trace provides verifier-consumed context but
exact draft-visible exposure remains unqualified.

Status: **ACCEPTED** as current roadmap priority sequence. Does not authorize merge,
paid benchmark, large re-evaluation, or production semantic wiring.

Confidence: 0.93

Supersedes / superseded by: Supersedes only the "Next authorized engineering
program" and active-priority clauses in D-2026-08-27-02 and the corresponding
PROJECT_STATUS mission text. Historical P0-1/P0-2a/P0-2b slice decisions,
evidence, and non-claims remain preserved. Does not close F-02, overall P0-2b,
or enterprise production readiness.

---
Decision ID: D-2026-09-04-02
Date: 2026-09-04

Decision: **Current fixed-prefix source-context truncation is NOT accepted as
enterprise-qualified evidence exposure. Preserve current runtime behavior until
a bounded causal experiment is run; do not change limits by intuition alone.**

Reason/evidence:

- `_build_source_context()` in `agent/nodes.py` caps web sources at 1500 chars
  and KB results at 2000 chars.
- Existing code comments identify these as Phase-1 experimental limits.
- The same builder feeds grounding context to draft/verification paths.
- Successfully retrieving a source therefore does not guarantee that the model
  received the decisive evidence within it.
- This risk is separate from the known MiniLM embedding truncation issue.

Frozen hypothesis (not yet tested):

> Fixed prefix truncation materially reduces claim-level evidence sufficiency
> and/or causes false weak, false unverified, or false verified decisions
> compared with a claim-aware evidence-exposure strategy.

Non-decision:

- Do NOT preselect "send every full document" as the solution.
- Candidate alternatives (claim-aware spans, surrounding context, batching,
  larger complete exposure where context permits) must be compared causally.
- No runtime changes are authorized by this decision.

Priority: High-impact product-quality item, scheduled after Semantic P0 closure
and the minimum evidence/provider qualification foundation (D-2026-09-04-01
priority 2), as part of the retrieval/evidence redesign program (priority 3).

Status: **OPEN — EXPERIMENT REQUIRED**

Confidence: 0.92

Supersedes / superseded by: Complements D-2026-09-04-01; does not supersede
MiniLM truncation decisions (D-2026-08-13-03) or any historical retrieval
evidence. Does not authorize runtime limit changes, merge, or paid benchmark.

---
Decision ID: D-2026-08-27-02
Date: 2026-08-27

Decision: **CLIENT-DEMO-FIXES: CLOSED.** Successful human client-demo rehearsal
validated the delimiter-policy correction (`p0-1-v2`), publish-target guards, and
remote-parent race protection. Status: **LIVE-DEMO-REHEARSED.**

Rehearsal chronology (historical; failures preserved):
1. Run `fd0cb6a3-781f-422c-ade4-3b1e49c04ed7` — **FAILED** before Gate 2: HTML
   delimiter false-positive on code examples (historical raw-substring policy).
2. Run `1e68d0ee-546e-4a25-b459-1340a9f22b2a` — HTML and both gates succeeded;
   publication correctly failed closed because local demo-fork `main` did not equal
   the approved remote parent.
3. Run `2c91cc9e-766d-4921-a6c2-d37253f76543` — **successful human rehearsal:**
   - Topic: LangGraph State Machines
   - V/W/U/total: `32 / 0 / 0 / 32`; grounding: `0.916`; reflection: `7/10`
   - Reflection provenance: real judge; `provider_called=true`; `parse_status=ok`
   - HTML policy: `p0-1-v2`; `policy_diagnostics=[]`
   - Gate 1: human approved; Gate 2: human approved
   - Artifact SHA256: `87af3a25133257918b68833058db02c2abebc55ed9e1dca01f010ef9687c63d4`
   - Approved remote parent: `7914833b1dea16789ac82f22be1fcc18e8bb965c`
   - Resulting demo-fork main: `7b6de76d0fdb0b78440f4211eb6c98cd8a7a6b73`
   - Final remote-parent race check passed; publication to demo fork succeeded
   - Demo URL `https://tmw-demo-site.netlify.app` returned HTTP 200
   - Production repository remained untouched

Non-blocking backlog (future hardening, not demo blockers): successful run telemetry
did not stamp immutable Content Agent code identity (`code_identity.available=false`).
Branch/SHA is strongly supported by workspace state but not independently proven from
run telemetry. The final 32/32-verified run did not exercise Weak/Unverified UI
filters; this does not reopen the demo.

Question: Did the client-demo correction project achieve a supervised human rehearsal
with approved artifact publication to the demo fork only, without production impact?

Evidence: Principal decision `CLIENT-DEMO-FIXES: APPROVED FOR FORMAL CLOSEOUT`.
Validated branch `feature/client-demo-rehearsal-fixes` at `66df1ee526cc2e7096f2b8c04c7221ce5068e27a`.
Pre-merge canonical main `e4b39caa37497c1743e33843c464ea4381f166d4`.

Status: **CLOSED.** Docs closeout on `feature/client-demo-rehearsal-fixes`. Next
engineering program: Semantic P0 Metric Registry.

Confidence: 0.95

Supersedes / superseded by: Closes the in-flight client-demo correction tracked from
`b4bcef2` / `f992ebf` / `a44416e` / `66df1ee`. Does not close overall P0-2b or
enterprise production readiness.

---
Decision ID: D-2026-08-27-01
Date: 2026-08-27

Decision: HTML delimiter policy versioned to `p0-1-v2`. `p0-1-v1` remains the
historical raw-substring policy: article body rejected `{{` or `}}`; full document
and reassembly rejected `{{` only (no structural HTML context), so literal code
examples could false-positive. `p0-1-v2` permits template delimiters only inside
approved `pre`/`code` (and immutable shell `style`) and records a reconstructable
`scope` on scanner diagnostics (`article_body` | `document` | `revised_body`).
`html_revise_node` persists the same safe `policy_diagnostics` on scanner
failure while still discarding the revision. Encoded braces are not a bypass:
`nh3` canonicalizes `&#123;&#123;` to literal `{{` before the scanner runs.

Question: Can the HTML-aware delimiter scanner ship under the historical
`p0-1-v1` identifier, without coordinate-scope or html_revise diagnostics?

Evidence: Principal review of `f992ebf` required a child correction. Boundary
tests on `f992ebf` passed 16 behavioral cases and failed only version, `scope`,
and html_revise persistence. No encoded-brace bypass. No sanitizer/CSP/URI
allowlist change.

Status: **VALIDATED AND INTEGRATED** as part of client-demo fixes closeout on
`feature/client-demo-rehearsal-fixes`. Human rehearsal run
`2c91cc9e-766d-4921-a6c2-d37253f76543` confirmed `p0-1-v2` with
`policy_diagnostics=[]`. See D-2026-08-27-02.

Confidence: 0.90

Supersedes / superseded by: Versions the acceptance semantics introduced in
`f992ebf`; does not redefine historical `p0-1-v1`. Does not change sanitizer
allowlist, CSP, or publish-target/Git race checks.

---
Decision ID: D-2026-08-26-02
Date: 2026-08-26

Decision: Demo `PUBLISH_TARGET` validation MUST inspect each remote's fetch URLs
**and** effective push URLs (`git remote get-url --all` and
`git remote get-url --push --all`). GitPython `remote.urls` is fetch-only and is
not sufficient. A clone with fetch=`themachinist-website-fork` and
pushurl=`themachinist-website` is denied. Any remote whose fetch or push identity
is `anudeepreddy332/themachinist-website` fails closed. The selected
`PUBLISH_REMOTE` effective push destination(s) must all be
`anudeepreddy332/themachinist-website-fork`; unparseable push URLs deny. Production
confirm-gate and SHA/parent/non-force checks are unchanged.

Question: Can `PUBLISH_TARGET=demo` allow a local merge or `/publish` push when
the clone's fetch URL is the approved fork but `git push` would follow a
production `pushurl`?

Evidence: Independent validation at `e630b75` reproduced allowed=True for
fetch=fork/pushurl=production; `git push` wrote to the pushurl repo. Correction
uses authoritative git get-url fetch+push listing. Regression tests cover the
split-identity hole at `evaluate_publish_target`, `git_node` (no website file),
and `/publish` (409, no ls-remote/push).

Status: **VALIDATED AND INTEGRATED** at exact canonical main
`e4b39caa37497c1743e33843c464ea4381f166d4`.

Confidence: 0.93

Supersedes / superseded by: Supersedes the fetch-only remote-URL inspection in
D-2026-08-26-01. Does not change demo allowlist members, production confirm
semantics, or retrieval/verifier/auth.

---
Decision ID: D-2026-08-26-01
Date: 2026-08-26

Decision: Fail-closed `PUBLISH_TARGET` policy ADOPTED for client-demo publish safety.
Unset/unknown target disables publication. `PUBLISH_TARGET=demo` is valid only when the
configured Git remote URL is `anudeepreddy332/themachinist-website-fork` and
`NETLIFY_BASE_URL` canonicalizes to `https://tmw-demo-site.netlify.app`. Production
requires `PUBLISH_TARGET=production` **and** `CONFIRM_PRODUCTION_PUBLISH=I_UNDERSTAND`.
The same validator gates `git_node` (local merge) and `POST /ui/runs/{id}/publish`
(remote push). Existing exact-SHA / remote-parent / non-force Git checks are unchanged.

Question: Can a supervised localhost demo accidentally publish to the production
website because `GIT_PUSH_ENABLED=true` while path/remote/site still point at
themachinist-website / themachinist.org?

Evidence: Pre-change, those four env vars were independent; `config.py` defaulted
`THEMACHINIST_REPO_PATH` at the production clone. Post-change, `agent/publish_target.py`
inspects actual git remote URLs (not path strings) and deny-by-default. Tests in
`tests/test_publish_target.py` cover unset target, demo allowlist, production repo/remote/site
mismatch, `GIT_PUSH_ENABLED=false`, production-without-confirm, SHA/parent preservation,
git_node enforcement, API enforcement, `GET /`, and no browser token persistence.

Status: **VALIDATED AND INTEGRATED** at exact canonical main
`e4b39caa37497c1743e33843c464ea4381f166d4`. Client demos must set `PUBLISH_TARGET=demo`.
No production publish was performed.

Confidence: 0.95

Supersedes / superseded by: Fetch-only remote-URL inspection superseded in part
by D-2026-08-26-02 (effective push URLs / pushurl must also be checked). Does not
supersede the local-merge-no-autonomous-push safeguard
(DECISIONS 2026-06-14/2026-06-16/2026-06-19). Adds a target-identity gate in front
of that path. Does not change retrieval, verifier, or auth.

---
Decision ID: D-2026-08-23-02
Date: 2026-08-23

Decision: Record P0-2b slice 2B as **P0-2B SLICE 2B — VALIDATED AND INTEGRATED** at exact
canonical main `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`. This is a status closeout of
already-accepted slice-2B semantic-trace persistence. It does not add production
architecture, change production semantic or routing behavior, or close overall P0-2b.

Question: Did exact slice-2B ancestry become canonical main with matching main-SHA
deterministic CI, without retrieval/prompt/model/threshold/benchmark change, without
provider execution, and without mutating the immutable BEFORE baseline?

Integration evidence: Direct remote `refs/heads/main` and fetched `origin/main` both
resolved exact `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`. Linear ancestry is exact
`e82672936fbb61ee7b1bde7dd3a1ced34f094fa8` →
`d3422e4252d6e127603109dd1cb0d6bfaa35a5c0` →
`f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`. Cumulative slice-2B scope is exactly
`agent/semantic_trace.py`, `agent/nodes.py`, `agent/state.py`, `main.py`, and
`tests/test_semantic_trace_v1.py`.

Validation evidence: Public GitHub Actions run
[`32644879371`](https://github.com/anudeepreddy332/content-agent/actions/runs/32644879371)
is a successful `push` run on `main` bound to exact `f6cc5a9`; lint PASS, deterministic
tests `497 passed`, provider-backed `eval-gate` skipped. Independent validation recorded
TRACE RECONSTRUCTABILITY, TRACE INTEGRITY, FAILURE TRACE SEMANTICS, REVISION TRACE
CAUSALITY, ROUTING NON-INTERFERENCE, TRACE SECRET SAFETY, and LEAN SCOPE as PASS.

Integrated contract: `semantic_trace_v1` now preserves exact drafts per iteration; exact
verifier-consumed input; raw verifier response; pre/post dedup and attribution state;
revision feedback/linkage; final UVR_v1 decision evidence; deterministic hashes;
reread/tamper validation; failure-state evidence; and secret-safety boundaries.
Production semantic/routing behavior did not change.

Known limitations recorded, not fixed: (1) runtime trace does not yet bind a verified
Git/code SHA; (2) crash telemetry may lack complete mid-run evidence; (3) verifier
source_context is already truncated before trace capture; (4) claim completeness remains
unknown; (5) Slice-2A claim-semantics oracle is not yet production-connected. These are
inputs to the upcoming principal audit, not a design mandate.

Bounded claim: Slice 2B is integrated as reconstructable semantic-trace persistence only.
Overall P0-2b remains **OPEN**. Immutable BEFORE baseline GitHub Actions run `32480353168`
is preserved unchanged. No AFTER paid benchmark has run. No provider call or spend
occurred during integration or this closeout. No new production architectural decision
is made here. No automatic Slice 2C.

Next mission: `PRINCIPAL TECHNICAL + BUSINESS PROJECT AUDIT`. Decision outcomes before
further major implementation: GO, NARROW, PORTFOLIO-CLOSE, PIVOT.

Status: `P0-2B SLICE 2B — VALIDATED AND INTEGRATED`. Overall P0-2b remains OPEN.

Confidence: 0.99

Supersedes / superseded by: Supersedes only the "Next mission" clause in D-2026-08-23-01.
Slice-2A evaluation-infrastructure contract, slice-1 routing contract, non-claims, and
historical evidence remain preserved. Does not supersede P0-2a evaluation-integrity
decisions, does not close remaining P0-2b completeness semantics, and does not authorize
Slice 2C.

---
Decision ID: D-2026-08-23-01
Date: 2026-08-23

Decision: Record P0-2b slice 2A as **P0-2B SLICE 2A — VALIDATED AND INTEGRATED** at exact
canonical main `bc96009d39394039ca019ec0f4da6358cf14be1d`. This is a status closeout of
already-accepted slice-2A evaluation infrastructure. It does not add production architecture,
change production runtime behavior, or close overall P0-2b.

Question: Did exact slice-2A ancestry become canonical main with matching main-SHA
deterministic CI, without retrieval/prompt/model/production-verifier/threshold change,
without provider execution, and without mutating the immutable BEFORE baseline?

Integration evidence: Direct remote `refs/heads/main` and fetched `origin/main` both
resolved exact `bc96009d39394039ca019ec0f4da6358cf14be1d`. Linear ancestry is exact
`a659fe9303626f82b1ca83fedfb5410a436b95d0` →
`0a1363f4bd328cd94fd662531780b7e9fa920376` →
`bc96009d39394039ca019ec0f4da6358cf14be1d`. Cumulative slice-2A scope is exactly
`docs/P0_2B_CLAIM_SEMANTICS_V1.md`, `evals/claim_semantics_v1.schema.json`,
`evals/fixtures/claim_semantics_v1.json`, `scripts/evaluate_claim_semantics.py`, and
`tests/test_claim_semantics_evaluator.py`. No production module is imported or changed.

Validation evidence: Public GitHub Actions run
[`32638253105`](https://github.com/anudeepreddy332/content-agent/actions/runs/32638253105)
is a successful `push` run on `main` bound to exact `bc96009`; lint PASS, deterministic
tests `467 passed`, provider-backed `eval-gate` skipped. The integrated slice-2A contract
is a deterministic claim-semantics oracle with 14 frozen fixtures and 18 canonical gold
factual atoms; material/full claim recall use independent gold denominators; duplicate,
compound, fragment, and qualifier-loss gaming is detected deterministically; JSON schema
authority (`DEFAULT_SCHEMA`) and executable semantic validation are enforced on every
official pack load. Production runtime behavior is unchanged.

Bounded claim: Slice 2A is integrated as evaluation infrastructure only. It is not a
production completeness guarantee for an unknown-sized model-extracted claim set. Overall
P0-2b remains **OPEN**. Immutable BEFORE baseline GitHub Actions run `32480353168` is
preserved unchanged. No AFTER paid benchmark has run. No provider call or spend occurred
during integration or this closeout. No new production architectural decision is made here.

Next mission: `P0-2B SLICE 2B — SEMANTIC TRACE PRESERVATION`. Do not start Slice 2B
implementation from this closeout without separate authorization.

Status: `P0-2B SLICE 2A — VALIDATED AND INTEGRATED`. Overall P0-2b remains OPEN.

Confidence: 0.99

Supersedes / superseded by: Supersedes only the "Next mission" clause in D-2026-08-21-03.
Slice-1 routing contract, non-claims, and historical evidence remain preserved. Does not
supersede P0-2a evaluation-integrity decisions, does not close remaining P0-2b completeness
semantics, and does not authorize Slice 2B implementation.

---
Decision ID: D-2026-08-21-03
Date: 2026-08-21

Decision: Record P0-2b slice 1 as **P0-2B SLICE 1 — VALIDATED AND INTEGRATED** at exact
canonical main `230314f7f774ed4b112c377269b190fa1279a004`. This is a status closeout of
already-accepted slice-1 routing. It does not add architecture, change product code, or
close overall P0-2b.

Question: Did exact slice-1 ancestry become canonical main with matching main-SHA
deterministic CI, without retrieval/prompt/model/evaluator/threshold change, without
provider execution, and without mutating the immutable BEFORE baseline?

Integration evidence: Direct remote `refs/heads/main` and fetched `origin/main` both
resolved exact `230314f7f774ed4b112c377269b190fa1279a004`. Linear ancestry is exact
`eea98c367b0f82fcc844dcca73b3935542adeef6` →
`aa90bfc1c4a7d430f0abeb07c84fa0c5416fce70` →
`230314f7f774ed4b112c377269b190fa1279a004`. Cumulative slice-1 files remain exactly
`agent/nodes.py`, `config.py`, `tests/test_uvr_fail_closed_routing.py`,
`tests/test_failure_injection.py`, `DECISIONS.md`, and `PROJECT_STATUS.md`.

Validation evidence: Public GitHub Actions run
[`32491216346`](https://github.com/anudeepreddy332/content-agent/actions/runs/32491216346)
is a successful `push` run on `main` bound to exact `230314f`; lint PASS, deterministic
tests PASS, provider-backed `eval-gate` skipped. UVR-aware fail-closed routing, bounded
revision and reverification, and the Gate-1 HTML-eligibility guard are the integrated
slice-1 contract: ordinary HITL approve cannot grant HTML generation when semantic
verification is not accepted.

Bounded claim: Slice 1 is integrated. Overall P0-2b remains **OPEN**. Claim completeness
for a production unknown-sized extracted claim set remains unresolved. Immutable BEFORE
baseline GitHub Actions run `32480353168` is preserved unchanged (20/20 complete and
scorable; 19/20 at or below UVR 0.15; topic 10 UVR = 5/21 = 0.238095...; overall FAIL;
zero benchmark retries). No AFTER paid benchmark has run. No new architectural decision
is made here.

Next mission: `ARCHITECTURE FREEZE FOR REMAINING P0-2B SEMANTICS`. Do not start remaining
P0-2b implementation from this closeout.

Status: `P0-2B SLICE 1 — VALIDATED AND INTEGRATED`. Overall P0-2b remains OPEN.

Confidence: 0.99

Supersedes / superseded by: Supersedes only the "Not merged" status clauses in
D-2026-08-21-01 and D-2026-08-21-02. Their routing contract, non-claims, and historical
evidence remain preserved. Does not supersede P0-2a evaluation-integrity decisions and
does not close remaining P0-2b completeness semantics.

---
Decision ID: D-2026-08-21-02
Date: 2026-08-21

Decision: P0-2b slice 1 HITL fail-closed correction. Ordinary HITL approve
(auto, interactive, or API) grants HTML eligibility only when
`semantic_verification_accepted(state)` is true. If semantic verification has
failed or is incomplete, `route_after_hitl` must not return `"html_gen"`.
Allowed existing destinations remain: explicit reject → END; explicit
feedback → draft; blocked ordinary approve → END (same destination as
reject/unknown). No new HITL status, graph edge, or workflow.

Status: Implemented as a follow-up commit on
`feature/p0-2b-slice1-uvr-fail-closed-routing`. Not merged.

Confidence: 0.94

Supersedes / superseded by: Narrows D-2026-08-21-01's clause that interactive/API
HITL "remains an explicit human decision" with respect to HTML eligibility.
Human intervention is preserved (reject, feedback, review). Ordinary approve
cannot launder a remaining failed/incomplete semantic state into HTML.

---
Decision ID: D-2026-08-21-01
Date: 2026-08-21

Decision: P0-2b slice 1 — semantic verification fail-closed routing. A verification
result proceeds past `route_after_reflect` only when `verification_status ==
"completed"`, the verdict set is nonempty, UVR is deterministically computable,
and UVR <= 0.15. UVR above 0.15 with remaining iteration capacity reuses the
existing draft revision path (prior unverified claims injected; verify runs again).
Parse failure, skipped verification, empty verdicts, upstream failure, and
unknown/incomplete status fail closed. Grounding/reflection/confidence cannot
override UVR or status failure. Claim-completeness remains explicitly `unknown`.
At MAX_ITERATIONS=2 or COST_GATE_USD=$0.10, routing still enters HITL; automatic
HITL approval cannot launder a semantic failure (reject → END). Interactive/API
HITL remains an explicit human decision. Thresholds, retrieval, prompts, and
models are unchanged.

Status: Implemented on `feature/p0-2b-slice1-uvr-fail-closed-routing` from canonical
main `eea98c367b0f82fcc844dcca73b3935542adeef6`. Not merged.

Confidence: 0.93

Supersedes / superseded by: Does not supersede P0-2a evaluation-integrity decisions.
Narrows production routing for P0-2b slice 1 only. Completeness contract remains
an open experiment (PROJECT_STATUS P0-2b UNKNOWN).

---
Decision ID: D-2026-08-19-05
Date: 2026-08-19

Decision: Close P0-2a as **VALIDATED AND INTEGRATED — CLOSED** at exact canonical runtime/product
integration anchor `74523ffcfa8906573a72415f1d868dc02996b561`.

Question: Did the exact reviewed P0-2a implementation become the canonical runtime/product state
through the expected merge identity and main-SHA public deterministic CI, without provider
execution or unrelated runtime change?

Integration evidence: At integration-closeout verification, direct remote main and fetched
`origin/main` both resolved exact
`74523ffcfa8906573a72415f1d868dc02996b561`. The merge has first parent exact prior documentation
closeout `174d8c924a35fc5151a4549725db9a01e96f119b` and second parent exact validated implementation
`0b707e4e431ea7662eec86aec5d4ed18a3c060dd`. Frozen architecture remains exact
`c8b75c3ab069df29e2201c0540b69bfca86e9cf1`.

Validation evidence: Exact implementation `0b707e4` passed `205` focused evaluation-integrity tests,
`31` verifier/failure regressions, and `406` full tests with three dependency deprecation warnings;
Ruff's fatal tier and range diff check passed. Public GitHub Actions run
[`32285001516`](https://github.com/anudeepreddy332/content-agent/actions/runs/32285001516) is a
successful `push` run on `main` bound to exact `74523ff`; its deterministic lint and test jobs
passed and its provider-backed evaluation job was skipped. No provider call or spend occurred
during integration, and no real paid 20-topic release benchmark was executed.

Bounded claim: P0-2a closes evaluator scope/cardinality/release-evidence integrity only. It does
not establish product-wide production readiness, deployment identity, P0-2b verifier correctness,
or permanent cryptographic authenticity of exported JSON after workflow finalization.

Next mission: After this documentation-only closeout is separately reviewed and integrated, run
the first trusted real release-mode benchmark against the exact HEAD of `refs/heads/main` using the
frozen ordered 20-topic V1 contract and approved GitHub Actions release workflow. Preserve complete
evidence, telemetry, exact identities, digest, and actual cost. Do not change code, configuration, prompts,
models, retrieval, thresholds, or corpus; do not retry to obtain PASS. A legitimate FAIL is accepted
baseline evidence. Only after that baseline is captured does P0-2b status/confidence/completeness/
routing semantics become the next architecture mission.

Status: `P0-2A VALIDATED AND INTEGRATED — CLOSED`. The real paid 20-topic release baseline remains
`NOT YET RUN`; the Content Agent remains blocked for overall enterprise production readiness.

Confidence: 0.99

Supersedes / superseded by: Supersedes only the integration-pending status in D-2026-08-19-04 and
the corresponding open integration clauses in D-2026-08-19-02 and D-2026-08-19-03. Their frozen
scope, historical evidence, and non-claims remain preserved.

---
Decision ID: D-2026-08-19-04
Date: 2026-08-19

Decision: Accept exact P0-2a implementation `0b707e4e431ea7662eec86aec5d4ed18a3c060dd`
as **IMPLEMENTATION VALIDATED — INTEGRATION PENDING** against frozen architecture
`c8b75c3ab069df29e2201c0540b69bfca86e9cf1`.

Question: Does the cumulative four-file implementation now make release PASS fail closed over exact
V1 scope, valid/scorable telemetry, frozen UVR and prompt/config identity, GitHub/code/clean-state
identity, aggregate/result coherence, secret-safe evidence, atomic write/reread, external trusted
runtime provenance, and every tested finalization failure?

SHA and scope evidence: Remote main remained exact
`ca29d32b4869269daa47142615d298580a577a77`; the architecture branch remained exact `c8b75c3`; and
the implementation branch resolved exact `0b707e4`, a linear descendant through `d204deb`. The
cumulative range changes only `scripts/benchmark.py`, `tests/test_evaluation_integrity.py`,
`.github/workflows/eval.yml`, and new `evals/benchmark_release_contract.json`. P0-2b runtime,
prompts/models, retrieval, thresholds, HITL, publication, P0-1, and `FREEZE.md` are unchanged.

Deterministic evidence: frozen dependency sync succeeded; the focused evaluation-integrity suite
passed `205`; verifier/failure regressions passed `31`; the full suite passed `406` with three
dependency deprecation warnings; Ruff's fatal tier and `git diff --check ca29d32..0b707e4` passed.
No retry-until-green or threshold change occurred.

Adversarial evidence: Independent exact-20 synthetic controls proved unchanged roots retain one
trusted PASS and print PASS once. Real on-disk contract and manifest mutations introduced only after
PASS write, reread, evidence validation, and trusted persisted-payload comparison caused the normal
loaders' `SystemExit(2)` paths; each was converted to controlled finalization failure, exited `1`,
printed no PASS, left no PASS or temporary artifact, and retained one validated sanitized FAIL
report. Real contract and manifest loader failures before PASS write were also controlled. The
representative prior scope, V1, telemetry/type, identity, secret, aggregate, coordinated-rewrite,
atomic-reread, and final-root-drift attacks failed at their intended validator layers.

Bounded claim: P0-2a provides trusted release-evaluator finalization at workflow time. It does not
provide signed or permanently tamper-proof JSON after the workflow, validate P0-2b semantics, run a
paid full release benchmark, or establish product-wide production readiness.

Alternatives considered: accept `d204deb` despite a surviving PASS artifact (rejected by the frozen
fail-closed gate); require post-workflow cryptographic authenticity in P0-2a (rejected as an explicit
scope expansion); integrate now (rejected because integration and main cut-in require separate human
authorization and exact integration-SHA validation).

Status: `P0-2A IMPLEMENTATION VALIDATED — INTEGRATION PENDING`. Canonical main remains exact
`ca29d32b4869269daa47142615d298580a577a77`. No provider call, paid release benchmark, merge,
publish, or deployment occurred. After P0-2a integration, P0-2b verifier
status/confidence/completeness/routing semantics is the next separate product architecture mission.

Confidence: 0.99

Supersedes / superseded by: Supersedes only the open implementation-status clauses in
D-2026-08-19-02 and D-2026-08-19-03. Their historical defect evidence, frozen architecture, scope,
and integration authorization boundary remain preserved.

---
Decision ID: D-2026-08-19-03
Date: 2026-08-19

Decision: Tighten the frozen P0-2a evidence contract with post-run code attestation, immutable
versioned fixture identity, main-only release qualification, and exact-object specification access.

Reviewer-required clarifications: A long benchmark must re-attest HEAD and tracked staged/unstaged
cleanliness after all units and before PASS; `content-agent-release-topics-v1` must remain exactly
20 ordered topics with its frozen digest rather than absorb future fixture changes; release-qualified
paid evidence must run under GitHub Actions on `refs/heads/main` with `github.sha` equal to the
expected and checked-out SHA; implementation must read the unmerged specification from an exact Git
object while branching product work only from `ca29d32`.

Frozen consequence: Mid-run HEAD or tracked-state drift produces release FAIL. Count, ID, manifest,
required-field, or semantic evolution requires an explicitly reviewed successor contract such as
V2. Feature/architecture refs are smoke/non-release only and are rejected before provider-bearing
workflow steps when release mode is requested. The independently authorized final architecture SHA
must be read with `git show <SHA>:docs/P0_2_VERIFIER_EVALUATION_INTEGRITY_ARCHITECTURE.md`; it must be
the one child of reviewed architecture commit `5451b53`, while implementation still starts from
exact canonical product main `ca29d32`.

Scope: The four implementation files remain `scripts/benchmark.py`,
`tests/test_evaluation_integrity.py`, `.github/workflows/eval.yml`, and new
`evals/benchmark_release_contract.json`. P0-2b, UVR `0.15`, verifier, prompts/models, retrieval,
corpus, HITL, publication, P0-1, tenant roadmap, and MCP/A2A status are unchanged.

Status: Final architecture tightening only. Independent review and explicit human implementation
authorization remain required. No runtime implementation, provider call, merge, publish, or deploy.

Confidence: 0.99

---
Decision ID: D-2026-08-19-02
Date: 2026-08-19

Decision: Split P0-2 into sequential P0-2a evaluation-integrity and P0-2b verifier-semantics
missions; freeze P0-2a as the current smallest implementation mission. MCP and A2A are optional,
out of scope, and do not block the product-hardening sequence.

Problem / question: Current `scripts/benchmark.py` can select no or partial work and still emit a
passing gate, while verifier confidence/status/completeness and ceiling routing have separate runtime
defects. Combining them would mix deterministic evidence-substrate repair with model/runtime
behavior and would ask later verifier experiments to trust a fail-open evaluator.

Exact evidence: current `origin/main` and independent remote query both resolved
`ca29d32b4869269daa47142615d298580a577a77`. Zero-paid probes reproduced an unknown-ID `0/0` gate
pass, partial and duplicate-unit passes, unsafe zero/negative/out-of-range limit behavior,
`unverified + 0.99` scoring/routing, valid empty output marked completed, parse/empty states reaching
HITL at the iteration ceiling, cost-ceiling HITL routing, and approved HITL routing to HTML
generation. Existing exact-run telemetry and verifier-schema protections remain correct. Focused
tests passed `45/45`; the full suite passed `215/215` with pinned Chromium.

Reasoning: Evaluation selection/cardinality/evidence identity can be corrected with synthetic tests
without changing the verifier. That correction is a prerequisite measurement substrate for any
causal verifier-behavior experiment. Production verifier completeness remains `UNKNOWN REQUIRING
EXPERIMENT`; a global verdict count is not justified for an unknown-sized extracted claim set.

Frozen contract: `docs/P0_2_VERIFIER_EVALUATION_INTEGRITY_ARCHITECTURE.md`. P0-2a is limited to
`scripts/benchmark.py`, `tests/test_evaluation_integrity.py`, `.github/workflows/eval.yml`, and new
`evals/benchmark_release_contract.json`. It changes no model, prompt, threshold, verifier, retrieval,
corpus, HITL, publication, deployment, or provider-call behavior.

Status: `P0-2-CURSOR-OR-CODEX-READY` for architecture completeness only. Independent review and
explicit human authorization remain required before implementation. No implementation, merge,
publish, deployment, or provider spend occurred.

Confidence: 0.98

Supersedes / superseded by: Supersedes the current applicability of the documentation-only MCP-first
dependency recorded at `ca29d32`; it does not rewrite that historical state or alter P0-1 closure.

---
Decision ID: D-2026-08-19-01
Date: 2026-08-19

Decision: Close P0-1 as `VALIDATED AND INTEGRATED` at exact canonical main
`d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3` while retaining the overall enterprise-production
block.

Problem / question: Did the frozen P0-1 active-content, credential, citation, approval/artifact and
publication boundary survive exact-SHA implementation, independent review, isolated integration,
deterministic/browser validation, and public integration-SHA CI?

SHA and ancestry evidence: final Reviewer-approved implementation
`20eb17f2737010dbf72eea0f0e271bf47d5af3de` descends from frozen implementation base
`7a606e895fe0a4bc9092659f130881bc7b52bd28`. Canonical integration
`d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3` has parents exact canonical governance state
`904f3efe87e6771329b5088bb3afeb6cd16c90dc` and exact final implementation `20eb17f`. Integration
inspection found the canonical documentation blobs preserved from `904f3ef`, every reviewed P0-1
implementation blob preserved from `20eb17f`, and no additional runtime change.

Implemented controls: one server-side trusted renderer/sanitizer; inert Gate-1 Markdown; empty
sandbox and trusted `srcdoc` at Gate 2; restrictive and consistent FastAPI, Caddy and generated-
article CSP/headers; Authorization-header SSE; no bearer in URLs or browser storage; canonical
retrieved `source_ref` citation authority with DNS/IP/WHATWG and malformed-authority rejection;
ordered text/inline/block revision-content preservation; server-owned approval hash; archive/Git
byte equivalence; and exact-commit publication with expected-remote-parent race protection.

Validation evidence: exact final implementation passed Ruff's fatal tier, `215` deterministic
tests, and `4` pinned-Chromium browser-security tests. Exact integration has two successful public
push CI runs, `32143196853` and `32230388649`. The latter recorded Ruff success and `215 passed,
3 warnings`; its PR-only paid evaluation job was correctly skipped. No paid provider result is
claimed.

Alternatives considered: leave P0-1 open because unrelated enterprise blockers remain (rejected:
misstates the validated boundary); call the system production-ready (rejected: verifier/evaluation,
identity/tenant, durable recovery, readiness/HA and deployment-identity gaps remain); treat a later
descendant as implicitly approved (rejected: violates exact-SHA governance).

Status: `P0-1 VALIDATED AND INTEGRATED — CLOSED`. This is neither deployment proof nor overall
production-readiness approval. No later descendant is implicitly validated.

Confidence: 0.99

Supersedes / superseded by: Supersedes only the open/unimplemented P0-1 status clauses in
D-2026-08-13-02 and D-2026-08-18-01. Their threat analysis, frozen contract, historical governance
and exact-base record remain preserved.

---
Decision ID: D-2026-08-18-01
Date: 2026-08-18

Decision: Separate the audited P0-1 runtime reference from the exact authorized implementation
base under one specific independently reviewed exception.

Audited runtime reference: `794851dded770ce87d111e73735d000e23597eb1`. This remains the exact
runtime snapshot against which the P0-1 threat boundary and architecture were audited.

Authorized P0-1 implementation base: `7a606e895fe0a4bc9092659f130881bc7b52bd28`.
Independent Git inspection verified that `7a606e8` is a direct descendant of `794851d` containing
documentation changes only and therefore does not alter the audited runtime/product implementation.

Problem / question: The canonical architecture originally required implementation to start from
`794851d`, while the reviewed canonical-state synchronization at `7a606e8` contained the accepted
implementation contract but no runtime changes.

Reasoning: Authorizing `7a606e8` preserves the audited runtime implementation exactly while making
the reviewed canonical documentation available at the implementation base. This is a specific
exception, not a rule that documentation-only or arbitrary later descendants inherit authorization.

Evidence: Parent/ancestry and changed-file inspection for `794851d..7a606e8`; independent Reviewer
governance decision dated 2026-08-18.

Alternatives considered: Require Cursor to start at `794851d` and separately import the contract
(rejected: creates avoidable state divergence); authorize the next branch tip or all documentation
descendants (rejected: weakens exact-SHA governance).

Status: Accepted governance correction. Any implementation base other than exact SHA `7a606e8`
remains `ARCHITECTURE-BLOCKED` unless independently reviewed. This decision does not itself authorize
Cursor to begin; the independent Reviewer must inspect the correction commit.

Confidence: 0.99

Supersedes / superseded by: Narrows the implementation-start wording in `architecture.md`; does not
change the audited runtime reference or any other frozen P0-1 gate.

---
Decision ID: D-2026-08-13-04
Date: 2026-08-13

Decision: Adopt the enterprise validation sequence and independent ownership model as repository
governance.

Problem / question: Multiple architecture, experiment and implementation agents need one rule for
what evidence earns a state transition and who may approve it.

Decision and reasoning: Applicable work proceeds through variable isolation, isolated validation,
dependency integration, focused deterministic tests, full regression, subsystem evaluation,
causal validation where applicable, full E2E, performance/scale where applicable, and
security/failure-path validation where applicable. Merge/cutover occurs only after every applicable
frozen gate passes. No threshold lowering after results, retry-until-lucky, or aggregate score that
hides a critical individual regression is allowed.

The Architect/Auditor investigates, prioritizes, isolates variables, designs architecture and freezes
gates. The independent Reviewer challenges the design/evidence, approves or tightens specifications,
and reviews implementation; agents do not self-approve. The Implementation Agent implements only the
approved contract, runs its gates, reports evidence, and stops rather than redesigning or moving
thresholds. Repository evidence is authoritative; conversation memory is not.

Evidence: Independent Reviewer directive dated 2026-08-13; canonical rules in
`docs/EXPERIMENT_LEDGER.md` and `PROJECT_STATUS.md`.

Alternatives considered: Per-agent memory files (rejected: divergent truth); updating canonical docs
after every conversation turn (rejected: churn without validated state transition); allowing an
implementation agent to approve its own design (rejected: no independent control).

Status: Accepted governance. Update canonical state only after a material validated transition.

Confidence: 0.99

Supersedes / superseded by: Adds a repository-wide governance contract; does not erase earlier
experiment-specific rules.

---
Decision ID: D-2026-08-13-03
Date: 2026-08-13

Decision: Keep the existing MiniLM+BM25+RRF retrieval baseline serving while the known truncation
defect and replacement research remain open and isolated.

Problem / question: MiniLM's serving chunks exceed its effective input limit, but removing measured
truncation did not automatically improve retrieval/product-quality metrics. Subsequent exact-73
diagnostics explain mechanisms without completing a replacement cutover gate.

Evidence: Corrected baseline/candidate comparison in the 2026-08-11 decision; hardened seven-arm
ablation `b9f9d1d`; common-identity rank diagnostic `4f6637e`; unique-membership crossover
`de60706`; compact records in `docs/EXPERIMENT_LEDGER.md`.

Relevant baseline / branches: Production baseline `794851d`; evidence branch
`experiment/exact73-retrieval-channel-ablation`; exact-73 fixture fingerprint
`4a1d5d1d67b56867c71497cb58ed4964d356a122a14d47ef822c227dba5924e4`.

Alternatives considered: Adopt tokenizer-aligned `224/32` chunking because it removes truncation
(rejected: mixed/regressive quality); cut over to GTE (rejected: fused baseline remains stronger);
cut over to Jina (rejected: invalid/infeasible under the frozen local gate); redesign fusion from
diagnostics alone (rejected: not evaluated end-to-end).

Reasoning: Tokenizer safety and retrieval acceptance are separate gates. The current baseline is
imperfect but remains the strongest fully evidenced serving option. Evaluator/fixture repair must
precede further release conclusions.

Status: Accepted serving/research separation. Known truncation defect remains OPEN. No retrieval
cutover authorized.

Confidence: 0.97

Supersedes / superseded by: Supersedes the unqualified current applicability of the 2026-06-07
"retrieval is healthy and not the bottleneck" conclusion. That earlier conclusion remains historical.

---
Decision ID: D-2026-08-13-02
Date: 2026-08-13

Decision: Approve and freeze the P0-1 Active-Content Execution / Credential-Exposure Boundary
architecture in `architecture.md`.

Problem / question: At baseline `794851d`, model/web-influenced Markdown, HTML and citation URLs can
reach same-origin active rendering; Gate-2 preview is unsandboxed; bearer credentials appear in
event/preview URLs; raw generated HTML is persisted before final approval; and publish is not bound
to the exact reviewed artifact and commit.

Evidence: Exact-baseline source trace through `static/index.html`, `api/server.py`, `agent/nodes.py`,
`prompts/html_template.md`, `prompts/html_revise_system.md`, and `Caddyfile`; compatibility inventory
over 450 historical HTML artifacts; independent Reviewer approval dated 2026-08-13.

Alternatives considered: Browser-only sanitization (rejected: duplicates policy and leaves server
persistence/publication unsafe); keep same-origin new-tab preview with a capability token (rejected:
active origin and credential boundary remain); treat HITL as an XSS control (rejected: approval does
not neutralize active content); redesign retrieval in the same mission (rejected: independent scope).

Reasoning: One server-side trusted rendering boundary, inert sandboxed review, header-authenticated
streaming, canonical server-built citations, and exact hash/commit binding form the smallest coherent
boundary. Deterministic exploit/browser gates are frozen before implementation.

Status: `ARCHITECTURE APPROVED`; implementation not validated, not merged, not deployed. This
decision authorizes implementation only after the canonical-state documentation branch is reviewed.

Confidence: 0.96

Supersedes / superseded by: Supersedes the current applicability of the 2026-06-17 demo UI/preview
security tradeoff and the 2026-06-20 audit's supervised-demo release conclusion. Historical entries
remain preserved.

---
Decision ID: D-2026-08-13-01
Date: 2026-08-13

Decision: Establish one canonical engineering-state system in Git.

Problem / question: `PROJECT_STATUS.md` and `architecture.md` were stale; `FREEZE.md` is historical;
`DECISIONS.md` preserves material history but is not a concise current-state view; experiment evidence
was scattered across archived files and evidence branches. Fresh agents could reach contradictory
conclusions.

Decision and roles:

- `PROJECT_STATUS.md`: concise current state only.
- `architecture.md`: accepted architecture contract only.
- `DECISIONS.md`: append-only material decision history.
- `docs/EXPERIMENT_LEDGER.md`: one compact evidence/experiment index.
- `FREEZE.md`: preserved historical v5 freeze record, not a live status file.
- `README.md`: navigation to the canonical surfaces, not an independent status authority.

Evidence: Read of every tracked root state document and every file under `docs/`/`docs/archive/` at
`794851d`; branch/ref inspection including current retrieval evidence branches.

Alternatives considered: Turn `FREEZE.md` into live state (rejected: destroys historical evidence);
use `docs/PRODUCTION_READINESS.md` as current status (rejected: fixed June 2026 audit scope and
superseded conclusions); create separate Architect/Reviewer/Cursor memory files (rejected: divergent
truth); copy detailed reports into a new ledger (rejected: duplication).

Status: Accepted. Documentation synchronization is review-required and does not itself approve a
product merge or deployment.

Confidence: 0.99

Supersedes / superseded by: Supersedes the prior README label that grouped all root documents as
undifferentiated "canonical living docs"; preserves every older decision and freeze record.

---
Date: 2026-08-12

Decision: Evaluation gates must bind telemetry to the exact successful CLI invocation and
must distinguish verifier completion from the unverified-claim rate (UVR).

Problem: `check_telemetry_fields.py` ignored a nonzero subprocess result and selected the newest
existing JSON by mtime, so stale valid telemetry could print PASS. The benchmark also represented
zero verdicts as UVR 0.0 in run display, aggregate, and gate logic. A verifier parse failure could
therefore look like a successful run with zero unverified claims.

Action: Require CLI exit status 0, parse its emitted `RUN_ID`, load only
`outputs/runs/<RUN_ID>.json`, verify that record's `run_id` and requested topic, then validate
required fields. The mtime fallback is removed. New records carry `verification_status`:
`completed`, `parse_failed`, `skipped_cost_gate`, or `upstream_failed`. Completed runs with
verdicts compute normal UVR. Completed zero-verdict runs report UVR N/A: they are valid only when
the benchmark topic explicitly sets `allow_zero_claims`; otherwise they are unscorable/incomplete.
All non-completed statuses fail benchmark validation. Historical files lacking the new field remain
readable but are treated as `unknown`, never inferred completed.

Trade-off: Benchmark gates now reject incomplete evidence rather than presenting a misleading
zero rate. This does not change the UVR threshold (still <=0.15), verifier prompt/rubric, topics,
retrieval, or production telemetry history.

Evidence: Deterministic regressions cover nonzero CLI plus stale telemetry, missing exact telemetry,
run/topic mismatches, completed zero-claim opt-in and rejection paths, parse failure, aggregate N/A,
and UVR values 0.00, 0.15, and above 0.15.

Status: Accepted evaluation-integrity correction. No paid benchmark run was used to validate it.

---
Date: 2026-08-11

Decision: Retrieval evaluation semantics corrected; no production chunking or retrieval-architecture
change accepted. All pre-2026-08-11 `recall@k`, nDCG, concept-hit, and OOS-rejection figures remain
historical / legacy evaluator semantics, not directly comparable to the corrected source-level
metrics below.

Discovery: The old `recall@k` asked whether *any* expected source appeared in the top-k, so it
was hit/success@k rather than true multi-source recall. Its chunk-level DCG assigned a gain to
every returned chunk from a relevant source while IDCG counted only unique expected sources;
therefore repeated sibling chunks could yield nDCG > 1. Its OOS gate read a fused result's
`distance`, even though BM25-only records carry synthetic `0.0` before RRF fusion; that value is
not a calibrated dense confidence.

Action: `scripts/retrieval_eval.py` now exposes `hit@k`, `source_recall@k`, source-level nDCG
(only the first relevant occurrence earns gain while retaining its original rank),
`concept_coverage@k`, the retained >=67% `concept_pass@k`, and source-diversity diagnostics.
`tools/query_kb.py` adds a diagnostic-only raw-component path returning dense top-1
distance/similarity, BM25 top score, and hybrid RRF score without changing public `query_kb()`
results. `out_of_scope_rejection_rate` is deprecated/non-gating until a calibrated hybrid policy
exists.

Evidence: New isolated baseline `eval_correctness_baseline_61de06d` (unchanged 400
cl100k/50 chunker, 73 chunks, 64/73 = 87.7% MiniLM truncation risk) measured hit@1/3/5/8/10
= 0.967/1.000/1.000/1.000/1.000; source_recall =
0.883/0.950/0.967/0.983/0.983; MRR = 0.978; source nDCG =
0.967/0.946/0.954/0.961/0.961; concept_coverage =
0.636/0.822/0.867/0.892/0.892; and mean duplicate-source slots =
0.000/1.314/2.000/3.486/4.457. Detached experimental candidate
`eval_correctness_candidate_224_61de06d` (224 MiniLM tokens/32 overlap only, 139 chunks,
0% truncation) measured source_recall = 0.850/0.950/0.967/0.983/0.983, MRR = 0.961,
source nDCG = 0.933/0.934/0.943/0.951/0.951, concept_coverage =
0.511/0.706/0.828/0.872/0.892, and duplicate-source slots =
0.000/1.343/2.543/4.429/5.543.

Classification: CLASS C — MIXED. Ten in-domain queries retained source ranking but lost
top-3/top-5 evidence coverage; their coverage commonly recovered by top-8/top-10 while sibling
occupancy grew. Two queries showed source-ranking loss: `vanishing gradient problem in deep
networks` lost one expected source by top-3, and the XGBoost gradient/Hessian query moved its
first relevant source from rank 1 to rank 3. The required per-query source/chunk/concept evidence
is in the evaluator output comparison generated for this run.

Next experiment (not implemented): MiniLM-safe child chunks + hybrid retrieval + post-retrieval
neighboring-context expansion, assessed with the corrected evaluator and a predeclared check for
the two ranking-regression queries.

Status: Accepted evaluator repair only. Do not merge the experimental 224/32 chunking code or
change the production collection, top-k, model, RRF constant, reranking, or retrieval topology.

---
Date: 2026-06-20

Decision: LangSmith tracing made DEFAULT-ON when configured (feature/langsmith-fixes),
superseding the 2026-06-19 entry's three-var AND-gate below. `setup_langsmith_tracing()` now
enables iff `LANGCHAIN_API_KEY` + `LANGCHAIN_PROJECT` are both set; `LANGSMITH_TRACING="0"` is
a force-off override (any other value, or unset, has no effect — it is no longer required to
opt in). Also: `agent/nodes.py::_get_client()` wraps the DeepSeek client with LangSmith's
`wrap_openai()` when tracing is on (DeepSeek's API is OpenAI-compatible), so every `_llm_call()`
becomes a traced "llm" run with token usage attached; `_llm_call()` then layers this pipeline's
own DeepSeek per-token pricing (`DEEPSEEK_INPUT_COST_PER_M`/`DEEPSEEK_OUTPUT_COST_PER_M`) onto
that run's `usage_metadata`, since LangSmith's built-in price table has no `deepseek-chat`
entry — without it, tokens would show but cost would read as zero.

Reason: parity with the code-agent project, where tracing activates automatically whenever
LangSmith credentials are present (no separate enable flag), and token usage/cost are visible
per-trace. The original `LANGSMITH_TRACING=1` AND-gate required an extra flag with no safety
benefit over the key+project check already being the real gate; cost was previously invisible
in the LangSmith UI for every traced run because the LLM call path used the raw `openai`
client (not a LangChain chat model), so no LLM-type run — and no usage_metadata — was ever
created for it.

Evidence: tests/test_tracing.py (8 tests, $0) — disabled-by-default, disabled-on-partial-config,
default-on-with-key+project-and-no-flag, force-off-override, `is_tracing_enabled()` state check,
and two fresh-process (subprocess) import checks. Full suite green (62 passed). Manually verified
(mocked OpenAI response + patched LangSmith client, $0, no network) that `_attach_usage_to_current_run`
populates `run.metadata["usage_metadata"]` with `input_tokens`/`output_tokens`/`total_tokens`/
`total_cost` matching `_cost()`'s own arithmetic. `scripts/smoke_test.py` re-run with tracing OFF
(the production default) to confirm the unwrapped client path is unaffected: SMOKE PASS,
cost=$0.0128, grounding=0.75, reflection=7 — consistent with pre-change baselines. No prompt
file touched; `prompt_version` unchanged (sha-6687240c8cd8). structlog untouched — additive only.

Status: Accepted (locked). Off by default wherever `LANGCHAIN_API_KEY`/`LANGCHAIN_PROJECT`
aren't set (no change to docker-compose.prod.yml/.demo.yml or .env.example) — auto-on per-operator
the moment they add credentials, no flag required.

---
Date: 2026-06-19

Decision: LangSmith tracing IMPLEMENTED (feature/langsmith), closing the "IN PROGRESS / not
started" entry directly below. Opt-in only: `observability/tracing.py::setup_langsmith_tracing()`
gates on ALL THREE of `LANGSMITH_TRACING=1`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` being set;
missing any one is a true no-op (no langsmith import, no env mutation, no latency). When all
three are set, it normalizes `LANGSMITH_TRACING` to the literal `"true"` that
`langchain_core`/`langsmith` actually check for, then returns — LangGraph's native
LangChain-callback tracing takes over process-wide from there, with zero per-node code changes.

Reason: same as the entry below — closes the Phase 5 / PRODUCTION_READINESS.md Monitoring gap
cheaply, without building a self-hosted LangFuse stack.

Evidence: tests/test_tracing.py (8 tests, $0) — off-by-default, off-when-any-single-var-missing,
the "1"-vs-"true" normalization, and two fresh-process (subprocess) import checks proving
`api.server` boots cleanly both with the flag unset and with dummy (non-functional)
credentials set. No prompt file touched; `config.PROMPT_VERSION` confirmed unchanged
(sha-6687240c8cd8). structlog untouched — this is additive, not a replacement.

Status: Accepted (locked). Off by default in every existing deployment (no env var changes
made to docker-compose.prod.yml/.demo.yml or .env.example) — opt-in per-operator.

---
Date: 2026-06-19

Decision: LangSmith / cross-node tracing is the next planned workstream, status IN PROGRESS
(not started — recorded now so it isn't lost between sessions). Scope: instrument all seven
graph nodes with LangSmith tracing, closing the Phase 5 observability gap that structlog-to-
stdout (M5) does not cover — M5 makes a single run reconstructable from telemetry alone, but
does not give cross-run/cross-node trace visualization or latency-breakdown tooling.

Reason: agent.md Phase 5 names LangFuse/LangSmith monitoring as unstarted; PRODUCTION_READINESS.md
independently flags Monitoring as MISSING (no dashboards/alerting anywhere in the repo). Tracing
is the cheapest of the two to add given structlog already emits structured per-node events.

Alternatives Considered: Self-hosted LangFuse (heavier ops burden, a service to deploy/maintain
on the same EC2 box already running the demo) vs. LangSmith (hosted, lower setup cost) — leaning
LangSmith for that reason, not yet decided.

Status: IN PROGRESS / not started. No code changes yet. Revisit this entry when work begins.

---
Date: 2026-06-19

Decision: P-demo cloud deploy LIVE. EC2 (Ubuntu, Graviton/arm64) + Caddy (automatic TLS reverse
proxy, no manual cert management) + Docker Hub (`anudeepreddy332/content-agent:demo`) + sslip.io
(wildcard DNS-as-a-service, avoids needing a real domain for a demo). Live URL:
https://54-221-24-43.sslip.io.

Reason: the local-server-only demo (2026-06-18 rehearsal) required a human to keep a laptop
process running for anyone to use it. A persistent public URL needed: (a) TLS without manual
cert hassle -> Caddy's automatic HTTPS; (b) a domain without buying one -> sslip.io's
IP-embedded wildcard resolution; (c) a deploy that doesn't rebuild the embedding-model-baking
Docker image on a small EC2 box every time -> build once on a fast machine, push to Docker Hub,
EC2 only ever pulls (see the next entry).

Evidence: `docker compose -f docker-compose.prod.yml -f docker-compose.demo.yml config` confirms
the merged config removes `app`'s host port entirely (Caddy is the only public surface) and
adds `caddy` bound to 80/443, reverse-proxying to `app:8000` over the internal Docker network.
GET https://54-221-24-43.sslip.io/health returns 200 from the live box.

Tradeoffs: sslip.io ties the URL to the EC2 instance's IP — a new instance (e.g. after a
stop/start that reassigns the public IP) needs the DNS-equivalent step re-run (see
docs/deploy/DEPLOY_DEMO.md step 4) and the demo URL changes. Acceptable for a demo; would need
a real domain + Caddy's same automatic-HTTPS for anything longer-lived.

Status: Accepted (locked). Documented in docs/deploy/DEPLOY_DEMO.md, PROJECT_STATUS.md.

---
Date: 2026-06-19

Decision: Cloud publish endpoint (POST /ui/runs/{id}/publish) + Caddy/EC2 deploy topology
ADOPTED. This decision should have been recorded when the code shipped (feature/demo-ui,
merged to main) and is being backfilled now rather than left as an undocumented gap.

The endpoint is human-triggered (a separate explicit POST, not auto-fired after gate 2) and
gated on the run's git_status already being "merged"/"tagged_and_merged" — i.e. git_node has
already done its local merge and already refused to push. The endpoint then runs
`git push $PUBLISH_REMOTE main` inside the bind-mounted fork clone. git_node itself was NOT
modified: grep confirms `push` does not appear anywhere in git_node's body, only in this new,
separate, human-gated endpoint.

Reason: the local-merge-no-autonomous-push property (DECISIONS 2026-06-14/2026-06-16) is the
project's primary publish safeguard. Adding a real "go live" button to the demo had to extend
the publish surface without weakening that property — preserved by keeping the push in its own
endpoint that only succeeds after both HITL gates already passed and git_node already ran.

Evidence: tests/test_api_publish.py (11 tests: 404/409/500/200 paths, mocked subprocess, $0).
docker-compose.demo.yml verified via `docker compose config` to remove the app's host port
binding and add Caddy as the sole public entry point (see the deploy-topology entry above).

Status: Accepted (locked, in production on the live demo).

---
Date: 2026-06-18

Decision: DEMO (P-demo) live rehearsal PASSED end to end against the fork. Netlify site
(tmw-demo-site.netlify.app) deployed from themachinist-website-fork; server launched with
GIT_PUSH_ENABLED=true + THEMACHINIST_REPO_PATH=~/tmp/tmw-fork; topic "Why batch normalization
speeds up training" driven through both gates via the SPA's /ui endpoints.

Evidence: retrieve 10 web sources -> grounding 0.773 (floor 0.60), 28 claims, reflection 7/10;
html_gen produced 0 validation warnings; gate 1 + gate 2 both approved through the SSE flow;
git_node returned git_status=merged (local --no-ff merge only, feature branch auto-deleted,
no push attempted by the agent — confirms local-merge-no-push held under the demo's streaming
path exactly as under the CLI/poll path). Human ran `git push origin main` on the fork
(c2407ca..9cd3102); Netlify redeployed; article URL returned 200 and the homepage Learning Log
card rendered the new title. Total cost $0.0066.

Status: Accepted (rehearsal complete). Remaining for P-demo: AWS deploy of the validated
container for a persistent public URL (currently demo requires a human-run local server).

---
Date: 2026-06-17

Decision: DEMO (P-demo) — interactive live front-end ADOPTED. A self-contained SPA
(static/index.html, vanilla JS + EventSource, served at GET /) over a NEW additive streaming
surface on the FastAPI app: POST /ui/runs, POST /ui/runs/{id}/{approve|reject|feedback}, and an
SSE drain GET /ui/runs/{id}/events. The SPA shows each node completing in real time, both HITL
gates inline (gate 1 = draft + grounding table; gate 2 = rendered HTML in an iframe), and a
published-link panel. Publish target for the demo is a Netlify-deployed copy of the FORK
(~/tmp/tmw-fork), never production — launched with GIT_PUSH_ENABLED=true +
THEMACHINIST_REPO_PATH=<fork>; the human still runs `git push` (local-merge-no-push unchanged).
API host: AWS (post-rehearsal).

Architecture constraints honored: (1) the frozen poll API (/runs*) and ALL 40 prior tests are
untouched — the /ui surface is purely additive. (2) Streaming uses GRAPH.stream(stream_mode=
["updates","values"]) but ALL graph compute still runs on the single EXECUTOR (max_workers=1)
via a per-run queue.Queue that the SSE endpoint drains, so the check_same_thread=False SQLite
invariant holds. (3) interrupt()-based HITL is unchanged — graph.stream surfaces interrupts the
same way invoke does; it is a swap of the observation channel, not a re-architecture of the gate
model. (4) HITL stays MANDATORY; no auto-approve path added. SSE auth is the same fail-closed
compare_digest check, token via query param (EventSource cannot set headers) — acceptable for a
single-user demo, noted as a minor log-exposure smell.

Rejected: Streamlit/Gradio (rerun/generator models fight live per-node streaming AND the two
paused gates; still need the SSE endpoint underneath; read as internal tooling). LangGraph
Studio + LangSmith (shows graph internals, not the topic->gates->live-site narrative; no website
payoff; LangSmith was cut at freeze). asciinema (CLI recording, ruled out by the goal).

Evidence: tests/test_api_stream.py (3 tests: full two-gate publish with ordered node events +
both gate payloads + terminal summary; SSE token auth; resume-after-terminal 409). Full suite
43 passed, $0. Local smoke: server boots, serves SPA at /, fail-closed 401 on /ui/runs and SSE.
No prompt files touched -> grounding baseline sha-6687240c8cd8 unchanged (orthogonal to metrics).

Status: Accepted (code complete + tested). Remaining = operator steps: Netlify fork deploy,
one live local rehearsal (real DeepSeek/Tavily spend), AWS deploy. Suggest its own feature
branch off main + a v6-demo tag.

---
Date: 2026-06-17

Decision: P2.2 (index.html Learning Log auto-update) ACCEPTED, no reopen. git_node now patches
a Learning Log card into index.html on every successful publish, in place on re-publish of the
same slug, excluded from the tag-trigger diff so the merged/tagged_and_merged distinction from
B7 still holds.

Status: Accepted (locked, merged to main as v6-p2.2-index-auto-update).

---
Date: 2026-06-17

Decision: P2.3 (raise MAX_ITERATIONS above 2 to chase grounding) REJECTED. Reverted to
MAX_ITERATIONS=2; experiment toggle removed (commit d301ed2, this branch).

Reason: A higher ceiling cannot help drafts that never revise in the first place (the gate
that matters is route_after_reflect ever choosing to revise, not how many times it's allowed
to). On drafts that do revise, the extra iterations traded SV (substantive verified claims)
for vagueness rather than improving grounding — UVR-only improvement is explicitly disallowed
as a success criterion per the SV-no-loss co-condition.

Alternatives Considered: Raising the ceiling to 3-4 (rejected on the above); pairing a higher
ceiling with stronger per-iteration grounding feedback (not attempted — M4's unconditional
grounding-report feedback was already in place and didn't change the verdict).

Evidence: scripts/archive/p2_3_analyze.py, scripts/archive/p2_3_reach.py (kept for the record,
not part of the production pipeline; relocated to scripts/archive/ in the feature/cleanup
reorg). SV noise band +/-7 at n<=3 observed and accounted for.

Tradeoffs: None — code reverted cleanly, no prompt or baseline change (sha-6687240c8cd8
unaffected).

Status: Rejected (locked). MAX_ITERATIONS stays 2 per CLAUDE.md locked decisions.

---
Date: 2026-06-17

Decision: P2 second HITL gate LOCKED as a CONTENT-FROZEN layout gate. Two gates, no production
bypass: gate 1 (hitl_node) reviews CONTENT; once approved, content is frozen. gate 2
(hitl_html_node) reviews the RENDERED HTML for design/structure/formatting/positioning ONLY.
request_changes routes to a new html_revise_node — a single temperature-0 LLM pass that edits
markup/layout on the existing html_output and loops back to hitl_html (NOT to draft). The
freeze is mechanically enforced: a visible-word-multiset guard discards any revision that
alters text/claims/code/sources (drift > 2 words) and keeps the original with a warning.
hitl_html feedback is stored in a dedicated html_feedback field, never hitl_feedback, so it
can never reach draft_node.

Correction recorded: the first implementation routed request_changes -> draft, which re-opened
content, re-fired gate 1, and could not change layout anyway (layout lives in html_template.md +
deterministic renderers, not draft_node). Caught before validation; superseded.

Reason: the reviewer must be able to fix how the article LOOKS without any risk of changing
what it SAYS, after content has been approved.

Tradeoffs: introduces one new LLM call per layout-revision request (human-triggered, t=0, cost
accounted in total_cost_usd/latency_ms.html_revise). New prompt prompts/html_revise_system.md
is orthogonal to the grounding baseline (sha-6687240c8cd8) — no metric re-baseline. Recurring
or systematic layout issues should be fixed once in html_template.md (permanent, free) rather
than via per-article LLM revision.

Status: Accepted (locked). API: existing /feedback endpoint resumes both gates; no new endpoint.

---
Date: 2026-06-16

Decision: B6-B8 LOCKED. git_node validated against a fork (both paths: new-file→"merged" no
tag; changed-file→"tagged_and_merged" + v-YYYYMMDD-<slug>). html_gen_node filename fix applied
(published name always canonical <slug>.html; local-archive suffix decoupled — see
2026-06-14). One supervised live publish executed (HITL approve → local merge → human push →
curl 200). Rollback script corrected after a faulty first pass: (a) `--merges` path-filter
returns nothing due to git TREESAME history simplification, so the publish merge is now found
by grepping the branch name in the merge message (with a file-commit fallback); (b) `-m 1` is
applied ONLY when the target is a merge (2nd-parent check), else plain revert; (c) the manual
`rm -f` was REMOVED — it was a content-loss bug that deleted the article on a MODIFICATION
rollback, where the correct outcome is restoring the previous version (file must remain). The
"it passed once" was a coincidence of the test article (a double-publish whose modify-rollback
correctly left the file in place, misread as a leftover).

Reason: a rollback that silently deletes an edited article instead of restoring its prior
version is worse than no script. Correctness re-derived from git semantics, not from one
manual pass — same discipline as the M5 telemetry and B3 always-exit-0 findings.

Evidence: fork re-test — initial-publish rollback removes the file; modification rollback
keeps it with restored prior content; neither needs manual deletion. RECOVERY.md present.

Status: Accepted (locked).

---
Date: 2026-06-16

Decision: PRODUCTION FREEZE. Tags v5-b6b8-complete (last code) and v5-freeze (docs-only).
Phase 4A + Phase 4B (B1-B8) complete; B6 staging-branch literal mechanism, B9 autonomy,
tracing, M5b, Docling, OpenClaw all DEFERRED as new phases. Four standing limitations
documented in FREEZE.md. HITL mandatory; no autonomous publish.

Reason: 10-day deadline met — a deployed, authenticated, container/cloud system with measured
grounding, enforced CI gates, failure tests, durable async HITL, validated publish + rollback.

Status: Accepted (frozen). Reopen only with new evidence per the locked-decisions list.

---
Date: 2026-06-14

Decision: html_gen_node filename fix (B6 blocker). The PUBLISHED filename is now always the
canonical <slug>.html, decoupled from local-archive collision-avoidance. Previously the
run-id uniqueness suffix (meant only to avoid clobbering outputs/articles/ debug copies)
leaked into state["html_filename"], which git_node writes into the website repo — so every
republish of the same topic produced a NEW filename and git_node always took the "new file
-> merged" path; the "changed file -> tagged_and_merged" path was unreachable. Fix: published
filename = f"{slug}.html" unconditionally; the local archive keeps its own per-run-unique
name (write-only, never reaches git_node). Live URLs are now stable across republishes.

Reason: republishing an article must update <slug>.html in place so git sees a modification
(the tag-then-merge rollback-protection path) and the public URL stays constant.

Evidence: fork test now yields "merged" on first publish and "tagged_and_merged" + a
v-YYYYMMDD-<slug> tag on republish. Regression test added (filename always slug-based).
Class: a local convenience (archive collision-avoidance) leaking into production publishing
semantics — same family as the M5 retrieve_kb telemetry leak.

Status: Accepted (locked).

---
Date: 2026-06-14

Decision: B6-B8 LOCKED (publish validation + supervised publish + rollback). git_node's
diff strategy validated against a FORK of themachinist-website (never production): new-file
-> "merged" (no tag); changed-file -> pre-merge tag v-YYYYMMDD-<slug> then "tagged_and_merged";
feature branch auto-deleted; tag prune keeps newest 5. ONE supervised real publish executed
with GIT_PUSH_ENABLED=true + HITL approval, then the operator's manual `git push origin main`
(Netlify deploy verified by curl 200 + body check).

KEY FINDING (load-bearing, NOT a bug): git_node performs a LOCAL merge only — it has no
remote/push. The automated agent therefore CANNOT publish to production autonomously; every
live publish requires a deliberate human `git push origin main`. This is the freeze's primary
publish safeguard and is left intentionally unchanged. GIT_PUSH_ENABLED gates the local git
operations, not the push.

Rollback: scripts/rollback_publish.sh reverts the article's --no-ff merge commit
(git revert -m 1) — safe on a pushed branch, uniform across both publish paths, independent
of the pre-merge tag. Single command + supervised push. RECOVERY.md created. Tag-based
rollback window = newest 5 v- tags (prune limit); revert-based rollback is unbounded.

Evidence: fork test both paths correct (git_status, tag presence, branch deletion, empty
error_log); one live publish reachable at the site URL; rollback dry-run on the fork removed
the article cleanly. .gitignore cleaned (outputs/ and tmp/ consolidated, milestone cruft and
blanket *.html removed).

Status: Accepted (locked).

---

Date: 2026-06-13

Decision: B5 LOCKED. [as previously specified — non-root image, model baked, uv --frozen
--no-dev, docker-compose.prod.yml separate from dev, Qdrant network-isolated (no host port),
secrets via runtime env_file, API loopback 127.0.0.1:8000 + SSH tunnel, GIT_PUSH_ENABLED=
false, CMD=main.py serve, DEPLOY.md with B4 durability limitation, .env.example created.]
Evidence: clean --frozen build; both services healthy; ingest 20 files/73 chunks via isolated
network; full approve cycle grounding 0.94, git_status dry_run, prompt_version sha-6687240c8cd8
unchanged; Qdrant isolation confirmed. Merged to main, tag v5-b5-complete.
Status: Accepted (locked).

---
Date: 2026-06-13

Decision: B4 LOCKED (with one documented limitation). FastAPI front end with durable async
HITL: graph compiled once with a SqliteSaver checkpointer; hitl_node pauses via interrupt()
when HITL_MODE=api and resumes on Command(resume={action,...}). Single-worker executor —
interrupt() makes invoke() RETURN at the gate, so the worker is idle during human-wait and
runs queue only during compute; max_workers=1 also makes check_same_thread=False safe.
Endpoints: POST /runs, GET /runs/{id}, POST /runs/{id}/approve|reject|feedback; bearer auth,
fail-closed (hmac.compare_digest). HITL MANDATORY on every API run (no auto-approve path —
B9 territory, deferred). CLI path unchanged: build_graph(None) == the old bare compile();
verified by the --auto regression. State shape now defined once in main._build_initial_state,
imported by both CLI and API (kills the silent-merge drift hazard).

Limitation (accepted for freeze): the SqliteSaver checkpoint is durable across restart, but
the in-memory run REGISTRY is not — a paused run returns 404 after an app restart though its
state survives in checkpoints.sqlite. Mitigation: drain awaiting_review runs before any
restart/deploy. Post-freeze fix: rehydrate REGISTRY from the checkpointer at startup.

Evidence: 32/32 tests ($0); live approve/feedback/reject cycles correct, git_status dry_run,
prompt_version sha-6687240c8cd8 unchanged; CLI --auto regression passed; durability test
FAILED exactly as predicted (registry wiped on restart). Merged to main, tag v5-b4-complete.

Status: Accepted (locked, limitation documented).

---
Date: 2026-06-13

Decision: B5 LOCKED. Containerized for single-VM deploy. Non-root image (uid 10001), uv
--frozen --no-dev (build fails on stale lock; dev deps excluded), embedding model baked at
build time so M5 warmup is offline and the first run is fast. New docker-compose.prod.yml
(app + Qdrant) kept SEPARATE from the dev docker-compose.yml (which stays as the local
host-dev file). Production posture: Qdrant network-isolated (NO host port — reachable only as
qdrant:6333 inside the compose network); secrets injected at runtime via env_file, never
baked (.env dockerignored; verified via docker history/env); API bound to 127.0.0.1:8000
(loopback) + SSH tunnel, NOT internet-facing; GIT_PUSH_ENABLED=false (git_node dry-run);
CMD=main.py serve. checkpoints.sqlite + telemetry persist on the ./outputs bind mount.
DEPLOY.md runbook created, carrying the B4 durability limitation forward. .env.example
created (was referenced by run() but absent from the repo).

Reason: freeze requires a deployed, usable, cloud-hosted system with no secret leakage and an
isolated vector store.

Alternatives Considered: overwriting docker-compose.yml (rejected — breaks local host-dev
Qdrant access); named volume for outputs (rejected for the freeze — bind mount is easier to
inspect run JSONs; documented chown step); publishing Qdrant on 127.0.0.1:6333 for host
debugging (rejected — full isolation is the stronger reading of the constraint).

Evidence: clean --frozen build; both services healthy under prod compose; ingest via isolated
network; full approve cycle complete/dry_run; isolation + no-secrets checks passed.

Status: Accepted (locked for the freeze window).

---

Date: 2026-06-13

Decision: B3 LOCKED. CI in two tiers: per-push free (.github/workflows/ci.yml — ruff
fatal-tier lint E9/F63/F7/F82 + 23 B2/B1 tests, uv sync --dev in both jobs); manual
secret-gated eval (.github/workflows/eval.yml, ~$0.07/dispatch — golden fixture with
ENFORCED exit code >=11/12 grounding & >=10/12 specificity, telemetry-correctness gate,
benchmark.py --gate slice requiring zero failures + per-run UVR <= 0.15). SV reported but
NOT gated in CI (±7 noise band at n=3 would flake a threshold). Two latent always-exit-0
bugs fixed: the golden test and benchmark.py both previously printed advisory gates
without setting an exit code — a CI step running either would have passed on garbage.
chromadb removed (zero imports since the Qdrant migration). docker-compose.yml un-ignored
(was gitignored — a CI checkout / git-based deploy would not have contained it; B5 blocker).

Reason: regression protection + deployment prerequisites for the freeze.

B3 bug lessons (encoded as process, not just fixed):
  1. Branch a feature off its dependency branch, or merge the dependency to main first.
     feature/b3-ci was cut from main before B2 merged -> B3 had no test files.
  2. Every dependency must be declared in pyproject.toml in the same commit it is used.
     A local `uv add --dev` masked a missing declaration; `uv remove chromadb` then
     cascaded and silently uninstalled ruff and pytest. A local venv hides missing
     declarations — CI is the truth. B4 follows this: deps added + grep-verified up front.
  3. CI uses `uv sync --dev` in lint and test jobs (both need dev deps).

Evidence: per-push CI green; one eval dispatch green end-to-end (all three gates, artifacts
uploaded). Merged to main --no-ff, tag v5-b3-complete.

Status: Accepted (locked).

---
Date: 2026-06-13

Decision: B2 LOCKED — failure-injection suite (tests/, 17 tests) covering the five
documented fault modes: DeepSeek auth (must propagate to the crash handler, never
swallowed — M4 fail-loud lesson encoded as a test), DeepSeek timeout/rate-limit (exactly
3 tenacity attempts, reraise), Tavily empty/error (per-query capture to error_log,
force-refresh pass exercised, retrieve == retrieve_web + retrieve_kb latency identity
guarded), Qdrant down (_collection_exists converts connection failure to empty kb_results
before any encoder load), malformed model JSON (_extract_json_array both layers;
verify -> empty report + score 0 + iteration_metrics entry; draft -> [PARSE ERROR]
sections with raw preserved), cost gate (fires BEFORE the LLM call in verify and reflect;
route_after_reflect cost gate and MAX_ITERATIONS ceiling beat the revise gate).

Reason: B-series production hardening; deploying untested failure paths was rejected in
the freeze-scope decision. The suite doubles as regression armor for M4/M5 fixes.

Evidence: 17/17 passed twice (3.2-4.1s); second run with networking disabled — zero
network dependency confirmed; $0 API cost; no production code changed; smoke test
passed post-suite (run smoke-810eee8f, grounding 0.79, $0.0082).

Tradeoffs: openai exception constructors in tests assume SDK >=1.x signatures (pinned in
uv.lock); reflect cost-gate test asserts only no-LLM-call + dict return (gate return
shape not independently verified). Mock-based — does not exercise real network timeout
behavior; the eval workflow's real runs cover that surface.

Status: Accepted (locked). Suite is the free per-push CI tier (B3).

---
Date: 2026-06-12

Decision: M5 core LOCKED. (1) Claim→source attribution is implemented POST-HOC: the
verifier's free-text source_url is resolved in Python (_resolve_attributions, agent/nodes.py)
against the actual retrieved set, tagging every claim web/kb/none/unresolved, with chunk_index
propagated through tools/query_kb.py and the full kb_results set (text included) persisted in
every run record. Zero verifier-visible change — verify prompt, source context, and
prompt_version (sha-6687240c8cd8) untouched, so no metric re-baseline. (2) The KB-latency
anomaly is CLOSED as a probe artifact: the diagnostic probe pre-loaded the encoder into a
local variable while tools.query_kb's module-level _encoder singleton stayed None, so "cold
query_kb 8.1s" was a second full model load, not Qdrant. Warm queries are 29–33 ms; Qdrant and
BM25 exonerated. A warmup() (encoder + BM25 build) now runs in main.py before graph
invocation, so latency_ms.retrieve_kb measures steady-state only. (3) The retrieve_kb
telemetry field itself was broken at birth (chained assignment `= latency = t_web` recorded
WEB latency under retrieve_kb); fixed to `latency - t_web`. check_telemetry_fields.py now
asserts attribution-count and chunk_index CORRECTNESS, not just field presence.

Reason: M5 exit gate requires runs reconstructable from telemetry alone; KB-verified claims
were previously unreconstructable (only kb_results_count persisted) and the verifier's cited
sources were never validated against the retrieved set (hallucinated citations would pass
silently into the article's sources list).

Alternatives Considered: Exact-chunk attribution via chunk-tagged source context (DEFERRED as
M5b — _build_source_context changes alter verifier behavior WITHOUT changing
prompt_hashes.verify_system, a documented hash blind spot; doing it requires golden-fixture
re-pass and an explicit metric re-baseline). LangSmith tracing (CUT — see freeze-scope entry).

Evidence: probe runs 2026-06-12 (model load 8248 ms ≈ "cold" query_kb 8095 ms; injection test
confirmed); validation run with prompt_version sha-6687240c8cd8 unchanged, attribution sums
matching claim count, unresolved = 0, kb_results populated with chunk_index.

Tradeoffs: KB attribution is file-level + candidate-chunk set (≤5 persisted chunks with full
text), not exact-chunk. attribution.unresolved > 0 in future runs is a verifier-hallucinated
citation signal — investigate, don't suppress. Per-iteration full grounding reports added to
iteration_metrics as the final M5 close-out.

Status: Accepted (locked).

---
Date: 2026-06-12

Decision: PRODUCTION-FREEZE SCOPE adopted — 10-day deadline to a deployed, usable,
cloud-hosted system. Roadmap compressed: Day 1 hardening (M5 residual + M6 remainder + B1 +
LLM timeout + utcnow fixes) → B2 (failure-injection tests) → B3 (CI with enforced thresholds)
→ B4 (FastAPI + durable HITL via LangGraph checkpointer, 3 days, fallback = CLI-in-container
if not working by end of Day 6) → B5 (container + cloud, Qdrant network-isolated,
GIT_PUSH_ENABLED=false) → B6/B7/B8 compressed to one validation day (git automation against a
fork, one supervised publish, rollback script + runbook) → freeze tag. Explicit cuts:
LangSmith/cross-node tracing (structlog-to-stdout + run records meet the M5 reconstructability
gate; stdout IS the container log sink); M5b exact-chunk tagging; B9 autonomy gating (HITL
stays MANDATORY on every run — consistent with B9's own earn-autonomy philosophy, simply not
granted); Docling/multi-format ingest DROPPED (dead code; removing it resolves the Python
3.14 vs Docling 3.11–3.13 contradiction without touching the runtime — this completes M6);
20-topic benchmark runs as a manual CI workflow, not per-push.

Reason: Hard external deadline. The pipeline and its measurement system are complete and
locked; remaining value is operationalization, not experiments.

Alternatives Considered: Full sequential B1–B9 (does not fit 10 days); pinning Python to 3.13
to keep Docling (rejected — downgrading the runtime for dead code); skipping B2/B3 to buy B4
time (rejected — deploying untested failure paths is how production incidents are made).

Evidence: M1–M5 arc closed and locked; 100/100 gate report; recall@3 = 1.0; calibrated SV/UVR
metrics with prompt-hash comparability.

Tradeoffs: Tracing and autonomy remain post-freeze work. The supervised-publish staging step
(Day 9) is the only gate between dry-run and live git pushes.

Status: Accepted (locked for the freeze window).

---

Date: 2026-06-11

Decision: M4 PASSED its pre-registered criterion (b). Grounding-report feedback on revision is permanent and unconditional: every revision pass (iterations >= 1) injects the prior iteration's unverified claims with ground/generalize/cut instructions. All M4 experiment toggles (M4_FORCE_REVISE, M4_GROUNDING_FEEDBACK, M4_FREEZE_CACHE, H1/H2 freeze hardening, debug prints) removed; per-iteration verify metrics (iteration_metrics) retained as permanent instrumentation.

Reason: With sources frozen and revision forced in both arms, feedback-driven revision cut UVR beyond the registered gate on the primary headroom topic with no SV loss, while the blind re-roll revision (previous production behavior) actively regressed UVR on 2 of 3 topics. PASS is via criterion (b) only: all SV Effects (+3.4/+5.7/+5.4) were within the ±7 noise band, so the locked claim is over-claiming reduction, not depth gain.

Alternatives Considered: Keeping the blind re-roll loop (rejected — control data shows it is worse than nothing on healthy topics: Ridge UVR 0.170→0.270, Multi-Agent 0.275→0.341); re-scoping/removing the loop (FAIL-A branch, not reached); bundling reflect-notes feedback into the revision prompt (deferred — one variable at a time).

Evidence: m4_analyze 2026-06-11, 18 runs (3 topics x 2 arms x 3 reps), frozen cache verified per-query via freeze instrumentation, single prompt_version sha-6687240c8cd8. Multi-Agent: UVR2 treat-ctrl = -0.111 (gate <= -0.10), Effect +5.7 (no-loss condition met). CatBoost: -0.097, missed by 0.003, same direction (supporting, not load-bearing). UVR gate (<= +0.05) held on all topics. One voided 18-run round preceded this: a stale installed copy of tools/web_search.py in .venv shadowed the source tree (fixed via uv pip uninstall + uv sync --reinstall); preflight now asserts module identity.

Tradeoffs: Feedback fires only when a revision occurs (route_after_reflect composite gate unchanged), so the lever's reach is bounded by the gate. SV lift remains unproven. Telemetry field m4_feedback_claims and the experiment_flags stamp keep their names for run-record continuity.

Status: Accepted (locked).

---
Date: 2026-06-10

Decision: M6a prompt‑hash versioning IMPLEMENTED. Static PROMPT_VERSION="v1.0" replaced with a content hash over the four runtime prompt files (draft_system.md, verify_system.md, reflect_system.md, html_template.md). Per‑file hashes (PROMPT_HASHES) are stamped into telemetry alongside the composite. Any prompt edit silently re‑baselines all metrics (proven by CatBoost 0.06→0.50 discontinuity across verify‑prompt versions); the hash makes this visible and defines comparability: grounding/SV numbers are comparable across runs iff verify_system hashes match.

Reason: M3 metric‑discontinuity finding made this URGENT. M4 will involve prompt changes; running it without the hash would repeat the exact class of mistake the M1→M3 arc taught.

Alternatives Considered: None — this is a direct fix for a confirmed blind spot. Delaying M4 until after M6a is explicitly a sequencing decision.

Evidence: Verification steps in the M6a package confirm the hash changes on prompt edit and recovers on revert.

Tradeoffs: The remainder of M6 (Python 3.14/Docling pin contradiction) stays scheduled — this package deliberately does NOT touch pyproject.toml, one variable at a time.

Status: Accepted (locked).

---

Date: 2026-06-10

Decision: CatBoost one‑run adjudication CONFIRMED source‑depth ceiling. The flat SV on CatBoost in M3 is NOT over‑claiming, extraction failure, or verifier error. The draft already produces as many specific grounded claims as the sources can support.

Reason: 27/27 substantive claims verified (confidence 0.85‑0.95). The 4 unverified claims were all specificity:generic (editorial framing statements, not technical assertions). grounding_score 0.80, SV 27, UVR 0.12. The cite‑or‑generalize instruction is functioning correctly.

Evidence: outputs/runs/39b31720‑8636‑4665‑9ff8‑f98b239c137f.json.

Tradeoffs: Embedding and ReAct are assumed (not adjudicated) to have the same ceiling. If raising SV on those topics ever matters, the lever is source enrichment, not further draft/verifier changes.

Status: Accepted (locked).

---

Date: 2026-06-09

Decision: M3 LOCKED with a documented judgment call. Source-aware drafting (retrieve-then-draft)
+ cite-or-generalize instruction made PERMANENT (SOURCE_AWARE_DRAFT gate removed). The SV metric
(substantive-verified claims, calibrated 10/12 specificity, 11/12 grounding) is adopted as the
primary quality metric, with UVR retained as the over-claiming gate.

Reason: The pre-registered criterion (SV +25% on >=3/4 failure topics) formally FAILED (1/4).
Locked anyway because: (a) the criterion misjudged headroom — ReAct was already healthy under the
new metric (baseline UVR 0.12); (b) the worst over-claimer was dramatically fixed (Multi-Agent
SV +253%, UVR 0.63 -> 0.22), far beyond the noise band; (c) zero harm — no topic regressed beyond
noise, unlike M2; (d) architecturally correct (draft sees sources). This is an explicit
goalpost-move recorded as such.

Evidence: M3 A/B, frozen cache, calibration gate passed. Controls established the SV noise band
at ~±6-7 SV (Ridge +6.0, SVM -7.5 in opposite directions, n=2); only Multi-Agent (+14.3) clears it.
CatBoost/Embedding/ReAct deltas all within noise.

Tradeoffs / open items:
1. METRIC DISCONTINUITY: the verify prompt changed (lenient rubric + specificity field), so all
   pre-M3 grounding numbers are incomparable with post-M3 numbers. CatBoost blind UVR measured
   0.06 (M2.6) vs 0.50 (M3) — prompt drift and/or topic instability. M6 prompt-hash versioning is
   now URGENT, not scheduled.
2. CatBoost is the least stable measurement in the project (0.057 / 0.124 / 0.344 / 0.50 across
   blind cells). One adjudicated run scheduled before M4 to determine: genuinely unsupported
   claims vs extraction granularity, and whether flat SV is a ceiling.
3. SV deltas < ~7 at n<=3 are non-results by definition of the noise band.

Status: Accepted (locked). Next: one-run CatBoost adjudication, then M4 (reflect-loop validation,
testing the grounding-report feedback lever as the revision mechanism).

---
Date: 2026-06-08

Decision: Root cause of the residual grounding gap = OVER-CLAIMING VIA SUBSTITUTION (confirmed by
claim-count pattern, not yet by direct claim adjudication). The verifier is EXONERATED and retrieval
is CONFIRMED healthy. UVR-alone is voided as an objective because it rewards vagueness. Both "give
the draft more information" fixes (M1 freshness, M2 source-awareness) are VOIDED as fixes. New
direction: a joint grounded-depth metric (SV = substantive verified claims, gated on verified-
fraction >= 0.70) plus draft claim-discipline (cite-or-generalize, inverting the M2 synthesis
instruction).

Reason: Verifier scored 8/8 on known ground truth incl. paraphrase (conf 0.95). M2 ran with live
sources (retr_ms 6-16s, zero Tavily errors) yet grounding worsened on 3/4 topics. Claim-count
decomposition: on CatBoost/Embedding/ReAct, verified DOWN + unverified UP with total claims roughly
flat = substitution of verifiable generic claims with unverifiable specific ones. CatBoost blind
0.057 vs source-aware 0.244 proves sources are adequate and that the synthesis instruction induced
unsupported specifics.

Alternatives Considered: verifier strictness (refuted, 8/8 fixture); source insufficiency (refuted,
blind CatBoost 0.057); retrieval failure (refuted, live-fetch confirmed); claim-density/volume
(refuted for CatBoost, total claims flat).

Evidence: M2.5b Steps 1/3/4; M2.6 Diagnostic 1 claim-count table; verifier unit test 8/8;
M2 results table.

Tradeoffs: The grounding-only era is over; all future grounding work is judged jointly with depth so
optimization cannot win by going vague. M1 freshness still helps a minority of topics (Embedding) and
the score-gate is retained; it is just not the primary lever.

Status: Accepted (locked). M1/M2 voided as fixes. Next: M3 (instrument + calibrate joint metric,
re-baseline), then the claim-discipline experiment, then M4/M5. Direct claim adjudication (Diagnostic 2)
was skipped by decision; the M3 experiment will surface any extraction-granularity residual.
---
Date: 2026-06-08

Decision: Verifier EXONERATED; retrieval CONFIRMED healthy; root cause of the residual grounding
gap is over-claiming amplified by source injection, and UVR-alone is an unsafe objective because it
rewards vagueness. Roadmap redirects from "give the draft more information" to "constrain what the
draft claims" + a joint grounding/depth metric (M3 brought forward).

Reason: Verifier scored 8/8 on known ground truth incl. paraphrase. M2 treatment ran with live
sources yet grounding got worse on 3/4 topics. CatBoost blind 0.057 vs source-aware 0.244 proves
sources are adequate and that source injection introduced unverifiable claims.

Alternatives Considered: verifier strictness (refuted by 8/8 unit test); source insufficiency
(refuted by blind CatBoost 0.057); retrieval failure (refuted by live-fetch confirmation).

Evidence: M2.5b Steps 1,3,4; M2 results table; verifier unit test 8/8.

Status: Accepted (locked). M1/M2 voided as fixes (both added information; both failed). Next:
M2.6 confirmation diagnostics, then M3 as a joint metric, then draft claim-discipline.
---
CAVEAT (2026-06-08, post-M2): M1/M2 verdicts are conditional on verifier validity. M2 applied the
over-claiming fix (source-aware drafting) and grounding got WORSE on 3 of 4 topics, inconsistent
with over-claiming as a simple generation-side root cause. Both experiments held the single-pass
verify_node fixed and used UVR as the metric. M2.5 (verifier + metric validation) must confirm the
verifier is not the dominant error source before treating "over-claiming is the binding constraint"
as settled. If M2.5 shows the verifier marks paraphrased-but-grounded claims unverified, re-interpret
M1/M2 as measuring verifier strictness, not generation/retrieval quality.
---
Date: 2026-06-08

Decision: M1 verdict = FAIL. Retrieval freshness is NOT the dominant grounding lever. The binding constraint on the worst failure topics (Multi-Agent, ReAct) is draft over-claiming, not stale/thin retrieval. Proceed to M2 (source-aware drafting). This RESOLVES the open H-cache vs H-overclaim disambiguation (2026-06-07 "Open / Revisit" entry) in favor of over-claiming as dominant; freshness is real but topic-specific and already largely handled by the score-gate.

Reason: Across 4 failure topics (3 reps) + 2 healthy controls (2 reps), only 1 of 4 failure topics met the >=0.15 UVR-improvement threshold under forced-fresh retrieval. Two of four (CatBoost, ReAct) already retrieved live under the current score-gated policy, so the gate already applies freshness and the toggle added nothing. Of the two genuine cache hits, Embedding improved with fresh data (thin cache) but Multi-Agent did not, and its ~57% unverified rate on fresh data is the signature of over-claiming.

Alternatives Considered: Always-fresh retrieval policy / lower TAVILY_MIN_AVG_SCORE (rejected as primary fix; helps only Embedding-class topics and the score-gate already covers the live-retrieval topics). KB enrichment (deferred; recall@3 = 1.0, KB is not the gap for these topics).

Evidence: M1 results (control UVR -> treatment UVR): CatBoost 0.124 -> 0.344 (control already live), Embedding 0.564 -> 0.287 (cache hit, freshness helped), Multi-Agent 0.602 -> 0.572 (cache hit, freshness did not help), ReAct 0.251 -> 0.173 (control already live). Healthy controls moved <= 0.048 (< 0.05), validating the design.

Tradeoffs: Freshness still helps a minority of topics (Embedding); M2 on fresh sources is expected to subsume that benefit. Retain the existing score-gate (it is doing real work on CatBoost/ReAct); do not remove it.

Status: Accepted (locked). FRESH_RETRIEVAL toggle reverted; experiment complete.

---
Date: 2026-06-07

Decision: KB retrieval is healthy and is NOT the grounding bottleneck. Qdrant + BM25 + RRF (k=60) is the locked retrieval backend.

Reason: An adversarial 35-query golden set (easy/medium/hard/multi-hop/out-of-scope) passes at near-ceiling recall, so no claim fails for lack of a retrievable KB chunk.

Alternatives Considered: ChromaDB (prior backend, retained only as baseline); dense-only retrieval (rejected, misses exact technical terms).

Evidence: outputs/retrieval_eval_qdrant.json — recall@1 0.967, recall@3 1.0, recall@5 1.0, concept-hit 0.867, OOS rejection 0.8.

Tradeoffs: BM25 index rebuilt in-memory per process (cheap at ~160 chunks; revisit above ~10k).

Status: Accepted (locked).

---

Date: 2026-06-07

Decision: The dominant unverified mechanism is "no source attached" (unverified_no_source), not "source attached but mismatched" (unverified_has_source ≈ 0 system-wide). Verifier precision/mismatch is ruled out as the main failure axis.

Reason: When a claim is unverified, it almost never has a source attached, so the failure is upstream of verifier matching: either retrieval did not surface a relevant source, or the draft asserted a claim no source covers.

Alternatives Considered: Verifier model mis-scoring (rejected by the breakdown); ML-domain-specific failure H1 (rejected); system-wide retrieval failure H3 (rejected).

Evidence: grounding_breakdown telemetry (unverified_has_source ≈ 0 across runs); benchmark_step4_grounding_fix_validation.json failure profiles (Multi-Agent 1 verified / 16 weak / 15 unverified; Embedding ~8 verified / 19 unverified; CatBoost 10 verified / 17 unverified). H1/H3 falsified independently in thread p2.

Tradeoffs: The breakdown narrows the cause to two candidates but does not by itself pick between them.

Status: Accepted (locked observation). The disambiguation between the two candidates is OPEN and is exactly what M1 + M2 resolve (see next entry).

---

Date: 2026-06-07

Decision: Two live hypotheses remain for the residual grounding gap, and they are not yet locked: (H-cache) thin/stale Tavily web sources reaching the verifier, vs (H-overclaim) the draft asserting claims beyond available evidence because drafting is blind to retrieval. M1 tests H-cache; M2 addresses H-overclaim. [Note: prior threads numbered these inconsistently — thread p2 used H4=cache, H5=overclaim; the memory summary used H4=overclaim. Use the named labels above to avoid confusion.]

Reason: The cache-latency signal (failure topics retrieve in ~70–100ms, i.e. cache hits) points to H-cache; the Multi-Agent profile (1 verified despite running) and the blind-draft architecture point to H-overclaim.

Alternatives Considered: Treating either hypothesis as already settled (rejected — no experiment has isolated them at claim level on the failure topics).

Evidence: retrieve latency in benchmark_step4 telemetry; draft_node reads only topic/series/feedback and runs before retrieve_node (graph.py edge draft→retrieve), so the draft never sees sources.

Tradeoffs: Resolving this requires M1 before committing M2 scope.

Status: Open / Revisit after M1.

---

Date: 2026-06-05

Decision: R1 (derivation-targeted query reformulation) is REJECTED. Query reformulation is not a worthwhile grounding lever.

Reason: Treatment delta was ~+0.10 against a +0.30 pass threshold, because the control already recovered roughly 90% of formula claims once retrieval was fresh. The signal attributed to reformulation was actually freshness.

Alternatives Considered: Scaling R1 to a 30-run confirmation (rejected as not worth it given the weak delta).

Evidence: R1 MVP run set (thread p1). Key finding: retrieval freshness, not query wording, is the largest observed grounding mover.

Tradeoffs: None material; cheap experiment, clear negative result.

Status: Rejected (locked). Do not revisit query reformulation without new evidence.

---

Date: 2026-06-07

Decision: Claim-level unverified-rate (not scalar grounding_score) is the primary metric for all grounding experiments.

Reason: Scalar grounding is dominated by draft-verbosity variance and is unreliable for comparison.

Alternatives Considered: grounding_score as primary (rejected); FCSAR/formula-specific metric (useful for formula topics only, not general).

Evidence: Topic 1 (Linear & Logistic) grounding ranged 0.39–0.81 across runs (σ≈0.13 at temperature 0.3) with no design change.

Tradeoffs: Requires reading grounding_report / claims_* fields rather than a single number.

Status: Accepted (locked).

---

Date: 2026-06-07

Decision: Stale Tavily cache as a single, primary bottleneck is FALSIFIED. The cache mechanism itself works correctly.

Reason: A 91ms retrieve latency was confirmed as a legitimate fresh cache hit, not a stale miss; cache staleness correlates with but does not deterministically cause failures (one cache-hit topic scored well at 0.773).

Alternatives Considered: Treating cache staleness as THE cause (rejected).

Evidence: thread p2 latency analysis; Agentic AI topic cache-hit at 90ms scoring 0.773.

Tradeoffs: The freshness lever still matters (R1 finding); M1 quantifies it precisely rather than assuming it.

Status: Rejected as sole cause (locked). Freshness magnitude remains under test in M1.

---

Date: 2026-06-07

Decision: Phase 4B multi-agent architecture remains DEFERRED. The single-agent pipeline is the production baseline.

Reason: The failure mode is over-claiming and thin sources, not an architectural limit. A multi-agent verifier would face the same thin sources and would not change the arithmetic. None of the architecture.md §10 gate triggers are met in a way that points to specialization.

Alternatives Considered: Introducing a separate draft/verify agent now (rejected on evidence).

Evidence: recall@3 = 1.0; many topics already reach 0.85–0.94 single-agent; gate report 100/100 runs, 0 failures.

Tradeoffs: If, after M2 (source-aware drafting) and fresh retrieval, grounding still fails on >30% of runs while adequate evidence is demonstrably present in the retrieved set, revisit. Even then, the existing revise loop is the cheaper first move.

Status: Accepted (locked, revisit only on the stated trigger).

---

Date: 2026-06-07

Decision: Adopt the audit's Phase 4A/4B milestone roadmap (M1–M6, B1–B9) as the canonical project roadmap.

Reason: It reflects implementation reality (verified by audit), preserves the "simplest system that works" philosophy, and converts the plan into milestone gates with explicit exit criteria.

Alternatives Considered: Retaining the original day-by-day Phase 4 plan (superseded by reality); a full rewrite to a new architecture (rejected — no astronautics).

Evidence: Production-readiness audit, 2026-06-07; roadmap now in agent.md lines ~452–561.

Tradeoffs: Existing day/step labels are deprecated in favor of milestone IDs.

Status: Accepted (locked).

---

Date: 2026-06-07

Decision: Drafting is currently blind to retrieval (draft runs before retrieve and never reads web_sources/kb_results). Making drafting source-aware (M2) is the #1 architectural priority and the leading candidate root cause of the grounding ceiling.

Reason: The verifier scores claims the draft had no evidence for; this structurally produces unverified_no_source claims on thin-source topics.

Alternatives Considered: Treating the gap as purely a retrieval problem (under test in M1); adding a multi-agent verifier (deferred).

Evidence: draft_node signature (topic/series/feedback only) + graph.py edge order draft→retrieve→verify; topic-bimodal grounding in benchmark_step4.

Tradeoffs: M1 must run first to size how much of the gap M2 must close; do not start M2 before M1 returns.

Status: Accepted as priority (locked); execution gated on M1 outcome.
