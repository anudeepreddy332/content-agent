# Evidence Exposure 2C/2D Contract

_Frozen design contract for Stage 2C deterministic qualification and Stage 2D provider qualification. Last synchronized: 2026-09-04._

## Purpose

Stage 2C mechanically tests whether successfully retrieved source text is actually visible to draft and verifier consumers under the current prefix-exposure policy.

The causal chain under test:

`retrieved source (rank 1, frozen) → model-visible evidence → verifier-visible evidence`

Retrieval quality is frozen for Stage 2C. No embedding, BM25, RRF, chunking, query rewrite, or retrieval redesign changes are in scope.

## Semantic boundary

Validated claim labels remain frozen:

- `verified`
- `weak`
- `unverified`

Stage 2C does **not** test whether those semantic boundaries are correct. It tests whether the evidence needed to apply them is visible.

## Stage 2C exposure arms

| Arm | Policy ID | Behavior |
| --- | --- | --- |
| A | `arm_a_current_prefix` | Current production prefix: web 1500 chars, KB 2000 chars |
| B | `arm_b_complete_source` | Complete frozen retrieved source (diagnostic ceiling) |
| C | `arm_c_gold_relevant_context` | Frozen truth-relevant spans plus required surrounding context |

Arms B and C are diagnostic only. Production runtime is **not** authorized to adopt them from Stage 2C alone.

## Frozen fixture pack

- Pack: `evals/fixtures/evidence_exposure_2c.json`
- Evaluator: `scripts/evaluate_evidence_exposure_2c.py`
- Cases: `E2C-W01` … `E2C-K04` (eight frozen ASCII cases)
- Every correct source is rank 1

### Case catalog

| Case ID | Scenario |
| --- | --- |
| E2C-W01 | Web early support — evidence inside 1500-char prefix |
| E2C-K01 | KB early support — evidence inside 2000-char prefix |
| E2C-W02 | Web late support — evidence after 1500 |
| E2C-K02 | KB late support — evidence after 2000 |
| E2C-W03 | Web late contradiction — multi-span; contradiction after prefix |
| E2C-K03 | KB late qualifier/condition — multi-span |
| E2C-W04 | Web late exact numeric/threshold evidence |
| E2C-K04 | KB early decisive evidence + irrelevant late content |

Each case freezes: case ID, requirement ID, source kind/rank/ID, complete source bytes + SHA-256, claim text + SHA-256, materiality, independent semantic label, truth spans, required context windows, expected Arm A/B/C visibility, and per-consumer results.

Multi-span requirements count as retained only if **all** required windows are visible. Partial qualifier/contradiction exposure is **not** success.

## Stage 2C metrics (`k=1`)

1. `gold_evidence_requirement_recall_at_k.v1`
2. `draft_gold_evidence_recall.v1`
3. `verifier_gold_evidence_recall.v1`
4. `retrieved_to_draft_evidence_retention.v1`
5. `retrieved_to_verifier_evidence_retention.v1`
6. `relevant_span_truncation_violation_rate.v1`

Raw numerators and denominators are preserved. Zero denominators are explicit/N/A, never silently PASS.

## Stage 2C pass gates

Before Stage 2C qualification passes:

- rank-1 source exists for every case
- retrieval requirement recall@1 = 1.0 for every case
- Arm B draft/verifier recall = 1.0 per case
- Arm C draft/verifier recall = 1.0 per case
- Arms B/C truncation violations = 0
- early controls remain visible in Arm A
- late cases E2C-W02, E2C-K02, E2C-W03, E2C-K03, E2C-W04 measure exact Arm A loss (not assumed)
- ambiguous mapping/hash/span identity = INVALID

## Exposure telemetry

Deterministic evidence records include:

- complete pre-exposure source and hash
- original character length
- exposure policy ID
- exact exposed text and hash
- original/exposed offsets
- consumer (`draft`, `verifier`)
- requirement visibility result
- truncation reason when evidence is lost

This harness is separate from production `semantic_trace_v1` and does not alter it.

## Stage 2D (NOT authorized for provider execution until independent preflight review)

Stage 2D is bounded real-provider qualification using the corrected Semantic P0
ruler. It tests whether changing **only** evidence exposure changes verifier
verified / weak / unverified decisions in a materially correct direction.

### Frozen pre-provider contract (D-2026-09-04-06)

- Pack: `evals/fixtures/evidence_exposure_2d.json`
- Preflight harness: `scripts/evidence_exposure_2d_preflight.py`
- Price schedule evidence: `evals/evidence_exposure_2d_price_schedule.json`
- Assets: **P1–P7** (seven fixed drafts; no drafting node, retrieval, HITL, HTML, or publish path)
- Provider cells: **exactly ten** — `P1-PREFIX`, `P1-COMPLETE`, `P2-COMPLETE`,
  `P3-COMPLETE`, `P4-COMPLETE`, `P5-COMPLETE`, `P6-PREFIX`, `P6-COMPLETE`,
  `P7-PREFIX`, `P7-COMPLETE`
- Causal isolation: paired P1/P6/P7 differ only in exposure arm (`prefix` vs `complete`)
- One attempt per cell; provider SDK retries disabled; timeout/transport failure ⇒ INVALID
- Hard spend ceiling: **$0.08 USD** (preflight must refuse above ceiling)
- Provider execution default: **disabled** (`EVIDENCE_EXPOSURE_2D_EXECUTE=1` required)
- Shadow current-runtime UVR acceptance computed alongside corrected oracle; production routing unchanged

### Asset catalog

| Asset | Scenario | Cells | 2C source reuse |
| --- | --- | --- | --- |
| P1 | Late supported material claim | PREFIX, COMPLETE | E2C-W02 |
| P2 | Unsupported material claim | COMPLETE | new |
| P3 | Partial / weak material claim | COMPLETE | new |
| P4 | Supported-new material claim | COMPLETE | new |
| P5 | Required material omission | COMPLETE | new |
| P6 | Late contradiction | PREFIX, COMPLETE | E2C-W03 |
| P7 | Late qualifier | PREFIX, COMPLETE | E2C-K03 |

### Zero-tolerance future gates

- `material_false_verification_rate.v2` numerator = 0
- `automatic_semantic_false_pass_rate.v2` numerator = 0
- Expected FAIL: P2, P3, P5, P6-complete, P7-complete; P6-prefix / P7-prefix are exposure-sensitive high-risk cells
- Expected PASS: P1-complete (verified + binding), P4 (both material claims resolve)

Stage 2D results are **not** recorded until provider execution completes.

## Explicit exclusions

Stage 2C does **not** authorize:

- provider calls
- changes to `_build_source_context()` production behavior
- changes to 1500/2000 limits
- prompt, verifier semantic boundary, retrieval, embedding, or chunking changes
- paid 20-topic benchmark execution
