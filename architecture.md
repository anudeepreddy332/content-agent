# Content Agent — Accepted Architecture Contract

_Accepted contract synchronized 2026-08-23._

- **Audited runtime reference:** `794851dded770ce87d111e73735d000e23597eb1`
- **Authorized P0-1 implementation base:** `7a606e895fe0a4bc9092659f130881bc7b52bd28`
- **Final validated P0-1 implementation:** `20eb17f2737010dbf72eea0f0e271bf47d5af3de`
- **Canonical integration:** `d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3`, with parents exact
  canonical governance state `904f3efe87e6771329b5088bb3afeb6cd16c90dc` and exact final
  implementation `20eb17f2737010dbf72eea0f0e271bf47d5af3de`
- **P0-2a implementation base (historical):** `ca29d32b4869269daa47142615d298580a577a77`
- **Frozen P0-2a architecture:** `c8b75c3ab069df29e2201c0540b69bfca86e9cf1`
- **P0-2a validated implementation:** `0b707e4e431ea7662eec86aec5d4ed18a3c060dd`
- **Current canonical repository state:** the HEAD of `refs/heads/main`; direct remote main is
  exact `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`. It must contain the P0-2a integration commit,
  the P0-2b slice-1 integration commit, the P0-2b slice-2A integration commit, and the P0-2b
  slice-2B integration commit below; documentation-only descendants do not alter those validated
  runtime/product implementation blobs
- **P0-2b slice 2B canonical integration commit:**
  `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`, with linear ancestry exact
  `e82672936fbb61ee7b1bde7dd3a1ced34f094fa8` →
  `d3422e4252d6e127603109dd1cb0d6bfaa35a5c0` →
  `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`
- **P0-2b slice 2A canonical integration commit:**
  `bc96009d39394039ca019ec0f4da6358cf14be1d`, with linear ancestry exact
  `a659fe9303626f82b1ca83fedfb5410a436b95d0` →
  `0a1363f4bd328cd94fd662531780b7e9fa920376` →
  `bc96009d39394039ca019ec0f4da6358cf14be1d`
- **P0-2a canonical integration commit:** `74523ffcfa8906573a72415f1d868dc02996b561`,
  with parents exact prior closeout `174d8c924a35fc5151a4549725db9a01e96f119b` and exact validated
  implementation `0b707e4`
- **P0-2b slice 1 canonical integration commit:**
  `230314f7f774ed4b112c377269b190fa1279a004`, with linear ancestry exact
  `eea98c367b0f82fcc844dcca73b3935542adeef6` →
  `aa90bfc1c4a7d430f0abeb07c84fa0c5416fce70` →
  `230314f7f774ed4b112c377269b190fa1279a004`

## 0. Authority and status

This file contains accepted architecture only. Candidate ideas and experiment hypotheses do not
belong here until independent review accepts them. Material decision history remains in
`DECISIONS.md`; current implementation/release state remains in `PROJECT_STATUS.md`.

The P0-1 architecture below is **APPROVED, frozen, implemented, validated, and integrated** at exact
canonical main `d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3`. The contract is retained as the
regression boundary. This status does not prove hosting-provider deployment identity or make the
overall system production-ready.

The P0-2a architecture is frozen at exact `c8b75c3ab069df29e2201c0540b69bfca86e9cf1`, and its
four-file implementation is independently validated at exact
`0b707e4e431ea7662eec86aec5d4ed18a3c060dd`. Status is **VALIDATED AND INTEGRATED — CLOSED** at
exact canonical P0-2a runtime/product integration anchor
`74523ffcfa8906573a72415f1d868dc02996b561`. Public main-push CI run `32285001516` passed its
deterministic lint/test gates and skipped the provider-backed evaluation gate. No product-wide
production-readiness claim is implied.

P0-2b slice 2B is **VALIDATED AND INTEGRATED** at exact
`f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`. Public main-push CI run `32644879371` passed
deterministic lint/test (`497 passed`) and skipped provider eval. This records already-accepted
`semantic_trace_v1` persistence: exact drafts per iteration, exact verifier-consumed input, raw
verifier response, pre/post dedup and attribution state, revision feedback/linkage, final UVR_v1
decision evidence, deterministic hashes, reread/tamper validation, failure-state evidence, and
secret-safety boundaries. Production semantic/routing behavior is unchanged. Overall P0-2b
remains **OPEN**; claim completeness remains unresolved; no AFTER paid benchmark has run; no
automatic Slice 2C. Next authorized state is `PRINCIPAL TECHNICAL + BUSINESS PROJECT AUDIT`.

P0-2b slice 2A is **VALIDATED AND INTEGRATED** at exact
`bc96009d39394039ca019ec0f4da6358cf14be1d`. Public main-push CI run `32638253105` passed
deterministic lint/test (`467 passed`) and skipped provider eval. This records the already-accepted
deterministic claim-semantics oracle: 14 frozen fixtures, 18 canonical gold factual atoms,
independent material/full recall denominators, deterministic duplicate/compound/fragment/
qualifier-loss detection, and JSON schema authority plus executable semantic validation. It is
evaluation infrastructure only; production runtime behavior is unchanged. Overall P0-2b remains
**OPEN**; claim completeness remains unresolved; no AFTER paid benchmark has run.

P0-2b slice 1 is **VALIDATED AND INTEGRATED** at exact
`230314f7f774ed4b112c377269b190fa1279a004`. Public main-push CI run `32491216346` passed
deterministic lint/test and skipped provider eval. This records the already-accepted UVR-aware
fail-closed routing, bounded revision/reverification, and Gate-1 HTML-eligibility guard. It
does not add architecture. Overall P0-2b remains **OPEN**; claim completeness remains unresolved;
no AFTER paid benchmark has run.

The implementation-base SHA is a specific independently reviewed exception: `7a606e8` is a direct
documentation-only descendant of the audited runtime reference and does not change its runtime or
product implementation. No other descendant is authorized by implication. A different Cursor base
is `ARCHITECTURE-BLOCKED` unless independently reviewed.

## 1. Current accepted system and serving baseline

The repository remains a supervised, single-operator system:

```text
retrieve -> draft -> verify -> reflect -> optional revise (maximum 2)
         -> human gate 1 -> HTML generation -> human gate 2
         -> local Git integration -> separate human-triggered publish
```

- Retrieval: Tavily web search plus Qdrant `all-MiniLM-L6-v2` dense search and BM25, fused with
  RRF `k=60`.
- Generation/verification/reflection: DeepSeek through the existing LangGraph nodes.
- Review: two mandatory human gates. Gate 1 ordinary approve grants HTML eligibility only when
  semantic verification is accepted (P0-2b slice 1, integrated at `230314f`). Explicit reject
  or blocked approve → END; explicit feedback → draft.
- Execution: FastAPI with SqliteSaver checkpoints and a volatile in-memory run registry.
- Publication: Git integration into a website checkout plus a distinct publish endpoint.

The serving retrieval baseline stays unchanged. The known MiniLM truncation defect remains open;
the `224/32` tokenizer-safe candidate and the exact-73 alternative-model diagnostics have not earned
cutover. Retrieval research is isolated from this security mission.

## 2. P0-1 threat boundary

Treat all of the following as untrusted:

- web pages and retrieved excerpts;
- every model response, including verifier URLs and HTML;
- reviewer feedback that later reaches a model;
- historical/generated article files that were not produced under the trusted policy.

The trusted computing base is deliberately narrow:

- the server-side rendering/sanitization policy;
- repository-authored immutable article shell and CSS;
- canonical retrieved-source records after server validation;
- the reviewer application shell;
- exact artifact hashes, Git object IDs, and expected remote refs verified by server code.

No human approval converts raw model output into trusted HTML. Only successful passage through the
authoritative server policy does.

## 3. P0-1 accepted and implemented contract

### 3.1 One authoritative rendering boundary

Add one canonical module, `agent/html_policy.py`. It is the only authority for:

- Markdown-to-review rendering;
- untrusted fragment sanitization;
- controlled class validation;
- citation URL validation/normalization/resolution;
- deterministic article-shell assembly;
- JSON-LD serialization;
- policy versioning and SHA-256 generation.

It exposes immutable `TrustedFragment` and `TrustedArticle` values. Raw model HTML may exist only in
a local variable between the model return and the immediate sanitizer call. It must never be logged,
checkpointed, placed in `AgentState`, returned by an API, archived, or written into either Git
repository.

Mandatory sequence:

1. Obtain the model fragment locally.
2. Sanitize it immediately and discard the raw candidate.
3. Fail closed on sanitizer exceptions or structural invariant loss.
4. Assemble the full document from escaped scalar fields, trusted fragments, trusted citations,
   and the immutable server-owned shell.
5. Compute `html_sha256` over exact UTF-8 bytes.
6. Only that `TrustedArticle` may enter state, an interrupt, review, archive, or Git.

`html_revise_node` may receive only the sanitized article-body fragment. It never receives or
rewrites the document head, CSP, metadata, navigation, CSS, citations, footer, or JSON-LD.

### 3.2 Sanitizer and renderer dependencies

Pin direct dependencies:

- `nh3==0.3.6`
- `markdown-it-py==4.2.0`

Pin browser-test dependencies:

- `playwright==1.62.0`
- `pytest-playwright==0.9.0`

Gate-1 Markdown uses:

```python
MarkdownIt("js-default", {
    "html": False,
    "linkify": False,
    "typographer": False,
})
```

Disable image and strikethrough rules. Pass its output through the same configured `nh3.Cleaner`.
Strip comments and require sanitizer idempotence: `clean(cleaned) == cleaned`. Regex is not a
security boundary.

### 3.3 Untrusted fragment HTML policy

Allowed elements:

`h2`, `h3`, `h4`, `p`, `ul`, `ol`, `li`, `pre`, `code`, `strong`, `em`, `blockquote`,
`table`, `thead`, `tbody`, `tr`, `th`, `td`, `hr`, `br`, `div`, `span`, `sup`, `sub`.

Allowed attributes:

- `class` on `div`, `span`, `pre`, and `code`, limited to the controlled classes below;
- `scope="row"` or `scope="col"` on `th`;
- no other attributes.

Controlled classes:

`callout`, `callout-info`, `callout-key`, `callout-label`, `sl-definition`, `sl-code-block`,
`sl-code-header`, `sl-code-label`, `sl-code-lang`, `language-python`, `language-text`,
`language-json`, `language-bash`, `language-sql`.

Untrusted fragments may not contain anchors. Trusted server code builds citations separately.

Forbidden content includes every script or event handler; `style` and inline style; `iframe`,
`object`, `embed`; forms and controls; `meta`, `link`, `base`; SVG and MathML; images, media,
canvas and sources; `template`, `noscript`; all URL-bearing attributes; IDs, names, `data-*`,
`contenteditable`, `autofocus`, `tabindex`; and external fonts or subresources.

The trusted shell may add fixed navigation links, fixed non-linking/non-animated inline SVG, one
immutable inline CSS block, trusted citation anchors, and one non-executable
`application/ld+json` block. Generated articles contain no executable JavaScript, external fonts,
stylesheets, scripts, images, media, frames, or other subresource requests.

### 3.4 Gate 1

- Render draft Markdown on the server into `draft_review_html`.
- Embedded Markdown HTML is disabled; Markdown-created anchors are stripped.
- The interrupt contains only trusted review HTML, policy version, grounding metadata, and
  canonical source descriptions. The browser does not render raw Markdown.
- Remove the CDN `marked` dependency.
- Render the trusted document in `<iframe sandbox="" referrerpolicy="no-referrer">`.
- Parent-page dynamic text uses `textContent`, `createElement`, and `replaceChildren`. Dynamic
  `innerHTML` is forbidden.

### 3.5 Gate 2

Gate 2 uses exactly:

```html
<iframe sandbox="" referrerpolicy="no-referrer"></iframe>
```

No `allow-*` sandbox token is permitted. Only `TrustedArticle.html` may be assigned to `srcdoc`.
The document has an opaque origin, cannot execute scripts, submit forms, open popups, navigate the
top page, or download through the blocked sandbox capabilities. Empty sandboxing is not the
authoritative network-denial mechanism and does not by itself prevent subresource requests. The
no-attacker-network-request guarantee comes from the combined trusted HTML policy: model fragments
cannot create URL-bearing/subresource elements or CSS requests; generated and review HTML uses the
frozen restrictive CSP, including `default-src 'none'`; and browser exploit tests must prove that no
attacker-controlled request is emitted.

Remove the new-tab preview control and delete `/ui/runs/{run_id}/preview`. Replace it only with an
in-page expand/collapse control that changes container CSS. Do not use `blob:`, `data:`, a query
token, or another capability URL. A future new-tab requirement requires a separately reviewed
untrusted-preview origin.

### 3.6 Authenticated streaming and bearer handling

Replace `EventSource` with authenticated streaming `fetch`:

- `GET /ui/runs/{run_id}/events`
- `Authorization: Bearer <token>`
- `TextDecoderStream` plus an SSE frame parser
- no query-string authentication

Use `AbortController` when replacing a stream and stop on `segment_end`. On `401`/`403`, do not
retry; clear the in-memory credential and return to login. On network/EOF before `segment_end`,
retry at most three times after 0.5, 1, and 2 seconds, polling authenticated run state before each
retry. Run state is authoritative because the queue is not replayable; do not claim
`Last-Event-ID` support.

Copy the password-field value into closure memory and immediately clear the field. Never store it
in URL, local/session storage, DOM attributes, telemetry, or logs. Redact authorization material
from errors.

### 3.7 Canonical citation policy

Rendered links come only from canonical retrieved server records, never from a verifier string.

Clickable citations must be absolute HTTPS, have no credentials or fragment, use port 443/default,
and use a valid public hostname or globally routable IP. Reject localhost, `.local`, private or
reserved addresses, control characters, backslashes, malformed percent encoding, and values over
2,048 characters.

Normalize by trimming ASCII whitespace; parsing with `urlsplit`; lowercasing scheme and IDNA host;
removing default port 443; using `/` for an empty path; preserving path case and query exactly; and
removing the fragment. A verifier candidate resolves only by exact normalized equality with a
canonical retrieved source. Substring and fuzzy URL matching are forbidden.

Render `grounding_report.source_ref`, never `source_url`. Emit the canonical stored URL and force
`target="_blank" rel="noopener noreferrer nofollow"`. KB references and unresolved sources are
non-clickable text. The server does not follow citation redirects.

### 3.8 Exact approval, artifact, and publication binding

Add state fields:

- `article_body_html`
- `html_sha256`
- `html_policy_version`
- `approved_html_sha256`
- `git_commit_sha`
- `publish_expected_remote_sha`

The enforced invariant is:

```text
approved_html_sha256
  == SHA256(state.html_output UTF-8 bytes)
  == SHA256(local archive bytes)
  == SHA256(git show <git_commit_sha>:<article_path>)
```

- Stop archiving in `html_gen_node`.
- Gate-2 approval records the server's current hash; ignore client-submitted hashes.
- Clear approval on every regeneration or revision.
- After approval, atomically write exact UTF-8 bytes to archive and repository and verify read-back.
- Commit and retain the exact commit SHA.
- Publish `<git_commit_sha>:refs/heads/main`, never ambient mutable `main`.
- Require remote main still equals `publish_expected_remote_sha`; otherwise fail with conflict and
  push nothing.
- After a push, verify remote main equals `git_commit_sha`.

This proves Git artifact/commit equivalence. It does not prove a hosting provider deployed that
commit; deployment identity is a later release gate.

### 3.9 CSP and headers

Split the reviewer UI into external `static/app.js` and `static/app.css`. Its response CSP is:

```text
default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self';
frame-src 'self'; img-src 'self' data:; font-src 'self'; object-src 'none';
base-uri 'none'; form-action 'self'; frame-ancestors 'none'; worker-src 'none';
media-src 'none'
```

Also send `Referrer-Policy: no-referrer`, `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, a restrictive `Permissions-Policy`,
`Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Resource-Policy: same-origin`, and
`Cache-Control: no-store`. SSE also sends `X-Accel-Buffering: no`. Apply the policy in FastAPI and
Caddy. Add HSTS only after the real TLS hostname is verified.

Every generated article embeds a meta CSP that denies everything except the exact SHA-256 hash of
its immutable inline CSS. It specifies `script-src 'none'`, `img-src 'none'`, `connect-src 'none'`,
`frame-src 'none'`, `object-src 'none'`, `base-uri 'none'`, and `form-action 'none'`.

### 3.10 JSON and scalar handling

Escape every scalar placeholder. Serialize JSON-LD with `json.dumps`, additionally escaping `<`,
`>`, `&`, U+2028, and U+2029. Never repair JSON-LD by reversing HTML entities in a completed
document.

## 4. Frozen deterministic acceptance gates and validation record

Implementation started from exact authorized base
`7a606e895fe0a4bc9092659f130881bc7b52bd28` in a clean worktree. The frozen gates were:

1. Lock consistency and the existing fatal Ruff tier.
2. `pytest tests/` only; repository-root pytest is prohibited because it collects a paid,
   top-level evaluator.
3. Pinned headless Chromium security tests with no provider/network calls.
4. Exploit fixtures for scripts, SVG/MathML, event attributes, `srcdoc`, forms, CSS imports/URLs,
   link/meta/base, malformed markup, encoded schemes, private URLs, and JSON-LD breakouts.
5. Browser proof that no iframe script executes, popup/top navigation occurs, attacker request is
   emitted, or request URL contains a token.
6. Static proof of no dynamic `innerHTML`, no query authentication, both empty sandboxes, and no
   preview endpoint.
7. Sanitizer idempotence and exact visible-text/code preservation through layout revision.
8. Byte/hash equivalence across approval, archive, Git blob, exact commit, and pushed ref.
9. Fail-closed behavior on any exploit survival, hash mismatch, remote-ref race, unsafe CSP token,
   dependency incompatibility, paid-call attempt, or browser failure.

No threshold lowering, retry-until-green, or aggregate-score masking is allowed.

Validation closed on exact final implementation
`20eb17f2737010dbf72eea0f0e271bf47d5af3de`: lock synchronization and Ruff fatal tier passed,
`215` deterministic tests passed, and `4` pinned-Chromium browser-security tests passed. Read-only
integration analysis and subsequent merge preserved every reviewed implementation blob. Exact
integration `d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3` has two successful public push CI runs; run
`32143196853` and run `32230388649` succeeded; the latter recorded Ruff success and
`215 passed, 3 warnings`. The PR-only paid evaluation job was correctly skipped and is not claimed
as integration evidence.

## 5. Validated implementation file boundary

Existing files:

- `agent/nodes.py`, `agent/state.py`
- `api/server.py`
- `static/index.html`
- `prompts/html_template.md`, `prompts/html_revise_system.md`
- `Caddyfile`, `pyproject.toml`, `uv.lock`
- `.github/workflows/ci.yml`
- relevant existing tests under `tests/`

New files:

- `agent/html_policy.py`
- `static/app.js`, `static/app.css`
- focused HTML-policy, security-boundary, artifact-equivalence, and browser-security tests

No retrieval, verifier-rubric, tenant, deployment-target, or unrelated code was included in P0-1.

## 6. Stop conditions

The implementation mission required `ARCHITECTURE-BLOCKED` if its base differed from exact SHA
`7a606e895fe0a4bc9092659f130881bc7b52bd28`; unrelated work overlaps an authorized file; the pinned
sanitizer cannot run on Python 3.14/target platforms; a legitimate feature needs model-controlled
active content; sandbox/CSP behavior fails in supported Chromium; visible content must change during
layout-only revision; artifact bytes change after approval; publication would require force-push or
a changed remote parent; target deployment changes become necessary; any paid call is required; or
passing requires weakening this policy.

## 7. Deferred risks and capabilities

P0-1 blocks the reviewed active-content, credential-transport, citation-authority and artifact/push
binding failures. It does not make model text semantically
trustworthy. Web/reviewer prompt injection can still influence prose, reasoning, and source
selection while producing inert HTML. Also deferred: remaining P0-2b verifier completeness
semantics; P0-3
identity/tenant isolation; durable registry/queue recovery; HA/scale; dependency supply-chain
hardening; browser parser differentials beyond the frozen test matrix; and hosting-provider deploy
identity.

P0-2 is split sequentially: first fail-closed evaluation scope/cardinality and evidence binding,
then verifier status/confidence/completeness/routing semantics. P0-2b slice 1 integrated the
accepted UVR-aware fail-closed routing contract. The correct completeness contract for a
production model-extracted, unknown-sized claim set remains an experiment question; exact
count checks for known-cardinality fixtures do not answer it. Overall P0-2b remains OPEN.

## 8. Engineering sequence after P0-1

The accepted product-hardening sequence is:

1. P0-1 canonical closeout — complete;
2. P0-2a fail-closed evaluation scope/cardinality/release-evidence binding — validated and
   integrated, closed;
3. first trusted real 20-topic V1 BEFORE baseline — captured as immutable GitHub Actions run
   `32480353168` (20/20 complete and scorable; 19/20 at or below UVR 0.15; topic 10 UVR =
   5/21 = 0.238095...; overall FAIL; zero benchmark retries). Preserved unchanged;
4. P0-2b slice 1 UVR-aware fail-closed routing — VALIDATED AND INTEGRATED at exact
   `230314f7f774ed4b112c377269b190fa1279a004`;
5. P0-2b slice 2A deterministic claim-semantics oracle — VALIDATED AND INTEGRATED at exact
   `bc96009d39394039ca019ec0f4da6358cf14be1d`. Evaluation infrastructure only; production
   runtime unchanged;
6. P0-2b slice 2B semantic_trace_v1 persistence — VALIDATED AND INTEGRATED at exact
   `f6cc5a96e3e8fedec3bb4d2859c7e77183aa19d6`. Reconstructable evidence only; production
   semantic/routing behavior unchanged;
7. remaining P0-2b verifier semantics — OPEN. Next authorized state:
   `PRINCIPAL TECHNICAL + BUSINESS PROJECT AUDIT`. Decision outcomes before further major
   implementation: GO, NARROW, PORTFOLIO-CLOSE, PIVOT. No automatic Slice 2C. No AFTER
   paid benchmark has run;
8. P0-3 tenant/ACL isolation;
9. durability, recovery, and publishing integrity;
10. retrieval/chunking/embedding redesign;
11. enterprise ingestion and later scale/cost/observability work.

MCP and A2A are optional developer-workflow infrastructure, outside this product sequence, and must
not block it. The frozen P0-2a contract is
`docs/P0_2_VERIFIER_EVALUATION_INTEGRITY_ARCHITECTURE.md`. Its exact implementation has passed
independent review and exact canonical runtime/product integration at `74523ff`. Proposed
S3/Postgres/event-driven ingestion, tenant/ACL, model-tier, and related roadmap ideas remain later
missions and are not accepted by the P0-2a contract.

P0-2a release qualification is main-only. The immutable BEFORE baseline is GitHub Actions run
`32480353168`. A feature-branch, mid-run-drifted, or in-place-mutated V1 run cannot earn release
PASS. No AFTER paid benchmark has run after P0-2b slice 2B.

## 9. P0-2a validated and integrated boundary — closed

- Architecture authority: exact `c8b75c3ab069df29e2201c0540b69bfca86e9cf1`.
- Validated implementation: exact `0b707e4e431ea7662eec86aec5d4ed18a3c060dd`, linearly descended
  from canonical product main `ca29d32b4869269daa47142615d298580a577a77`.
- Canonical integration: exact `74523ffcfa8906573a72415f1d868dc02996b561`, with first parent
  exact prior closeout `174d8c924a35fc5151a4549725db9a01e96f119b` and second parent exact
  validated implementation `0b707e4e431ea7662eec86aec5d4ed18a3c060dd`.
- Cumulative implementation scope is exactly `scripts/benchmark.py`,
  `tests/test_evaluation_integrity.py`, `.github/workflows/eval.yml`, and
  `evals/benchmark_release_contract.json`.
- Deterministic acceptance: `205` focused tests, `31` verifier/failure regressions, and `406` full
  tests passed with three dependency deprecation warnings; Ruff's fatal tier and range diff check
  passed.
- Adversarial acceptance: unchanged roots produced one trusted PASS; real contract and manifest
  mutation after PASS write and reread produced controlled FAIL, no PASS print, no surviving PASS or
  temporary artifact, and one validated sanitized FAIL report. Real pre-write loader failures for
  both roots were likewise controlled.
- Public main-push CI run `32285001516` passed deterministic lint/test and skipped the
  provider-backed evaluation gate. Provider calls/spend during integration were zero. The real
  paid 20-topic release benchmark had not run. The later immutable BEFORE baseline is GitHub
  Actions run `32480353168`.
- P0-2a closes evaluator scope/cardinality/release-evidence integrity only. It does not provide
  permanent cryptographic authenticity of exported JSON, validate remaining P0-2b, publish, deploy,
  or make the Content Agent production-ready.

## 10. P0-2b slice 1 validated and integrated boundary

- Exact integration: `230314f7f774ed4b112c377269b190fa1279a004`.
- Ancestry: `eea98c367b0f82fcc844dcca73b3935542adeef6` →
  `aa90bfc1c4a7d430f0abeb07c84fa0c5416fce70` →
  `230314f7f774ed4b112c377269b190fa1279a004`.
- Cumulative slice-1 scope is exactly `agent/nodes.py`, `config.py`,
  `tests/test_uvr_fail_closed_routing.py`, `tests/test_failure_injection.py`, `DECISIONS.md`,
  and `PROJECT_STATUS.md`.
- Integrated contract, already decided in D-2026-08-21-01 and D-2026-08-21-02: UVR-aware
  fail-closed routing; bounded revision and reverification; ordinary HITL approve cannot grant
  HTML generation when semantic verification is not accepted.
- Public main-push CI run `32491216346` passed deterministic lint/test and skipped provider eval.
- Retrieval, prompts, models, evaluator, and quality thresholds are unchanged.
- Immutable BEFORE baseline `32480353168` is preserved unchanged. No AFTER paid benchmark has
  run. Claim completeness remains unresolved. Overall P0-2b remains OPEN. This section records
  integration status only; it does not add architecture.

## 11. P0-2b slice 2A validated and integrated boundary

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
  skipped provider eval. Provider calls/spend during integration were zero.
- Slice 2A is evaluation infrastructure only, not a production completeness guarantee.
  Retrieval, prompts, models, production verifier, evaluator release gate, and quality
  thresholds are unchanged. Immutable BEFORE baseline `32480353168` is preserved unchanged.
  No AFTER paid benchmark has run. Claim completeness remains unresolved. Overall P0-2b
  remains OPEN. This section records integration status only; it does not add production
  architecture.

## 12. P0-2b slice 2B validated and integrated boundary

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
  Production semantic/routing behavior did not change.
- Public main-push CI run `32644879371` passed deterministic lint/test (`497 passed`) and
  skipped provider eval. Provider calls/spend during integration were zero.
- Retrieval, prompts, models, production verifier, evaluator release gate, and quality
  thresholds are unchanged. Immutable BEFORE baseline `32480353168` is preserved unchanged.
  No AFTER paid benchmark has run. Claim completeness remains unresolved. Overall P0-2b
  remains OPEN. No automatic Slice 2C.
- Recorded, not fixed, as inputs to the principal audit: runtime trace does not yet bind a
  verified Git/code SHA; crash telemetry may lack complete mid-run evidence; verifier
  source_context is already truncated before trace capture; claim completeness remains
  unknown; Slice-2A claim-semantics oracle is not yet production-connected. This section
  records integration status only; it does not add production architecture.
