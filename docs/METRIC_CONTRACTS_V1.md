# P0 Metric Registry Contract v1

Status: **EVALUATION-ONLY CONTRACT**. This slice changes zero production
behavior. It does not call providers, modify the verifier, alter routing,
change prompts/models/retrieval, or run the paid benchmark.

## 1. Purpose

Register official P0 semantic metric identities and implement the
`claim_semantics_v2` oracle that computes them deterministically from frozen
human-adjudicated fixtures.

Historical `claim_semantics_v1` remains immutable. V2 fixes match-credit
eligibility by treating `qualifier_loss` and `invention` as match-ineligible
alongside `compound` and `fragment`. F-02 (`f02-r1`) adds a second
population — independently adjudicated final content claims — without
changing required-gold recall, required unresolved rate, or historical UVR.

## 2. Registered identities

| Identity | Version | Direction |
| --- | --- | --- |
| `material_claim_recall.v2` | v2 | higher is better |
| `material_claim_unresolved_rate.v1` | v1 | lower is better |
| `material_false_verification_rate.v1` | v1 | lower is better |
| `material_false_verification_rate.v2` | v2 | lower is better |
| `automatic_semantic_false_pass_rate.v1` | v1 | lower is better |
| `automatic_semantic_false_pass_rate.v2` | v2 | lower is better |
| `final_material_claim_unresolved_rate.v1` | v1 | lower is better |
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

Required target for automatic semantic PASS on **required gold only**: **0
unresolved required material atoms**. F-02 automatic PASS additionally
requires 0 unresolved final material atoms.

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

Safety authority: numerator must be zero. G-only oracle. Preserved
historically; F-02 does not reinterpret this identity.

### material_false_verification_rate.v2

\[
\frac{|\{c \in C_{\mathrm{mat}}^{-} : \hat{y}(c) = \mathrm{verified}\}|}{|C_{\mathrm{mat}}^{-}|}
\]

where \(C_{\mathrm{mat}}^{-}\) is the independently frozen FIXED
CLASSIFICATION CASE catalog restricted to material rows with independent
label weak or unverified. The catalog includes required, final-content, and
unmatched-final cases. Selection does not depend on extraction success,
`final_atoms` membership, or required-gold matching.

### automatic_semantic_false_pass_rate.v2

\[
\frac{|\{a : \mathrm{oracle}_{G+F}(a) = \mathrm{FAIL} \land \mathrm{route}(a) = \mathrm{PASS}\}|}{|\{a : \mathrm{oracle}_{G+F}(a) = \mathrm{FAIL}\}|}
\]

Uses the corrected required+final oracle. v1 remains historically registered.

### final_material_claim_unresolved_rate.v1

\[
\frac{|\{f \in F_{\mathrm{mat}} : f \text{ unresolved}\}|}{|F_{\mathrm{mat}}|}
\]

Automatic PASS requires raw unresolved numerator = 0. Zero final material
claims yield N/A for this *rate* with an explicit reason. A zero F
denominator does not by itself make the asset-level semantic predicate FAIL.

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
| material_false_verification_rate.v2 | 0 | 0 | N/A (no final inventory) |
| automatic_semantic_false_pass_rate.v2 | 0 | 0 | N/A (no final inventory) |
| final_material_claim_unresolved_rate.v1 | 0 | 0 | N/A (no final inventory) |
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
- `evals/fixtures/claim_semantics_v2_f02.json`
- `evals/claim_semantics_v2_f02.manifest.json`
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
pack must be byte-identical. Input, edge, candidate, and final-atom
ordering must not change results.

## 11. F-02 required vs final populations (`f02-r1`)

Two independent inventories:

- **G — required gold claims.** Existing `material_claim_recall.v2` and
  `material_claim_unresolved_rate.v1` remain G-only. New/unmatched final
  claims are never added to those denominators.
- **F — final content claims.** Every independently adjudicated factual claim
  asserted in the exact final draft, including required-equivalent,
  supported-new, weak-new, and unsupported-new claims. Unmatched is not
  automatically unsafe. A supported new claim may PASS. An unsupported
  material new claim must FAIL. Materiality is independently frozen input.
  A required claim omitted from the final article remains in G and must not
  be inserted into F.

Corrected automatic oracle for structurally valid F-02 inputs:

`semantic_pass = (U_G == 0 AND U_F == 0)`

`U_G` and `U_F` are unresolved *counts*. They are 0 when a population is
empty. Metric *rates* may be N/A when a denominator is zero. N/A is not
coerced into PASS or FAIL. Qualification adequacy of an empty pack is
separate from asset-level semantic disposition.

Invalid inputs are INVALID with `semantic_pass = null` (fail closed).

A final material claim is resolved only when independent semantic label =
verified, a qualifying automated prediction exists and equals verified,
binding targets that final claim, binding is valid and fully entailing, and
required identities are structurally valid.

Historical revisionless ADV01 bytes are not reinterpreted. F-02 lives in a
separate fixture pack with an approved manifest digest so deleting an unsafe
final atom or changing materiality is detected.
