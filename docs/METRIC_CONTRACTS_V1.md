# P0 Metric Registry Contract v1

Status: **EVALUATION-ONLY CONTRACT**. This slice changes zero production
behavior. It does not call providers, modify the verifier, alter routing,
change prompts/models/retrieval, or run the paid benchmark.

## 1. Purpose

Register five official metric identities — four new P0 semantic metrics plus
one immutable historical diagnostic — and implement the corrected
`claim_semantics_v2` oracle that computes them deterministically from frozen
human-adjudicated fixtures.

Historical `claim_semantics_v1` remains immutable. V2 fixes match-credit
eligibility by treating `qualifier_loss` and `invention` as match-ineligible
alongside `compound` and `fragment`.

## 2. Registered identities

| Identity | Version | Direction |
| --- | --- | --- |
| `material_claim_recall.v2` | v2 | higher is better |
| `material_claim_unresolved_rate.v1` | v1 | lower is better |
| `material_false_verification_rate.v1` | v1 | lower is better |
| `automatic_semantic_false_pass_rate.v1` | v1 | lower is better |
| `unverified_verifier_row_rate.UVR_v1` | UVR_v1 | lower is better (historical) |

Machine-readable registration: `evals/metric_registry_v1.json` validated by
`evals/metric_contract_v1.schema.json`. Every entry declares all twenty
contract concepts explicitly.

## 3. Semantic labels

Verifier-visible disposition labels for gold and predictions:

- **verified** — evidence fully entails every truth-relevant condition
- **weak** — partial or indirect support; at least one condition unentailed
- **unverified** — missing, topical-only, mismatched, or contradictory evidence

**unresolved** is a downstream oracle disposition, not a fourth verifier
label. A material atom is unresolved when any of: omitted extraction, only
weak final disposition, unverified gold label, forbidden-role-only
representation, invalid evidence binding, non-entailing binding, or
oracle-proven false verification.

## 4. V2 match eligibility

Candidates with any role in `{compound, fragment, qualifier_loss,
invention}` are match-ineligible. An `allowed_matches` edge referencing such
a candidate fails closed. Duplicate role is not intrinsically forbidden but
one-to-one matching prevents denominator gaming.

Matching uses frozen human-adjudicated edges only, maximum-cardinality
one-to-one assignment, deterministic ID-sorted tie-breaking. No fuzzy
matching, embeddings, edit distance, or LLM calls.

## 5. Metric formulas

### material_claim_recall.v2

\[
\frac{|\{g \in G_{\mathrm{mat}} : g \text{ matched by eligible candidate}\}|}{|G_{\mathrm{mat}}|}
\]

Measures extraction coverage only.

### material_claim_unresolved_rate.v1

\[
\frac{|\{g \in G_{\mathrm{mat}} : g \text{ unresolved}\}|}{|G_{\mathrm{mat}}|}
\]

Required target for automatic semantic PASS: **0 unresolved material atoms**.

### material_false_verification_rate.v1

\[
\frac{|\{g \in G_{\mathrm{mat}}^{-} : \hat{y}(g) = \mathrm{verified}\}|}{|G_{\mathrm{mat}}^{-}|}
\]

where \(G_{\mathrm{mat}}^{-}\) is material gold with label weak or unverified.
Safety authority: numerator must be zero for qualification.

### automatic_semantic_false_pass_rate.v1

\[
\frac{|\{a : \mathrm{oracle}(a) = \mathrm{FAIL} \land \mathrm{route}(a) = \mathrm{PASS}\}|}{|\{a : \mathrm{oracle}(a) = \mathrm{FAIL}\}|}
\]

Safety authority: numerator must be zero.

### unverified_verifier_row_rate.UVR_v1 (historical)

\[
\frac{U}{V + W + U}
\]

Post-dedup emitted verifier rows only. Ten weak rows and zero verified or
unverified rows yield \(0/10 = 0\) by historical design. Not the semantic
oracle.

## 6. INVALID vs N/A

- **INVALID** — malformed, missing, inconsistent, or tampered required
  identity/evidence; evaluation fails closed (raises error).
- **N/A / undefined** — otherwise valid unit but zero applicable denominator;
  `value` is `null` with explicit `undefined_reason`. Never silently coerced
  to zero or one.

## 7. Core fixture ADV01 hand results

| Metric | Numerator | Denominator | Value |
| --- | --- | --- | --- |
| material_claim_recall.v2 | 3 | 4 | 0.75 |
| material_claim_unresolved_rate.v1 | 3 | 4 | 0.75 |
| material_false_verification_rate.v1 | 1 | 2 | 0.5 |
| automatic_semantic_false_pass_rate.v1 | 1 | 1 | 1.0 |
| unverified_verifier_row_rate.UVR_v1 | 0 | 5 | 0.0 |

Historical UVR coexists with semantic failure: all five verifier rows are
verified or weak; three material atoms remain semantically unresolved.

## 8. Runtime boundary

Allowed files for this slice:

- `docs/METRIC_CONTRACTS_V1.md`
- `evals/metric_contract_v1.schema.json`
- `evals/metric_registry_v1.json`
- `evals/claim_semantics_v2.schema.json`
- `evals/fixtures/claim_semantics_v2.json`
- `scripts/evaluate_claim_semantics_v2.py`
- `tests/test_claim_semantics_v2_evaluator.py`

No production module is imported. V1 evaluator artifacts are read-only
historical references.

## 9. Schema authority

- `evals/claim_semantics_v2.schema.json` — external/static contract
- `scripts/evaluate_claim_semantics_v2.py` — executable semantic validator
- `evals/metric_contract_v1.schema.json` — metric registry contract
- `evals/metric_registry_v1.json` — registered identities and hand results

Parity between frozen JSON Schema and runtime validator constants is checked
on every official pack load. Semantic rules JSON Schema cannot express
(SHA-256 equality, span-text equality, forbidden-role edge rejection) remain
in the Python validator.

## 10. Determinism

Normalized output is `json.dumps(..., sort_keys=True, indent=2,
ensure_ascii=True)` plus a trailing newline. Repeated evaluation of the same
pack must be byte-identical. Input, edge, and candidate ordering must not
change results.
