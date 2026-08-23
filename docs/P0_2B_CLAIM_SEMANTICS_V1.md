# P0-2b Slice 2A — Lean Claim-Semantics Oracle

Status: **EVALUATION-ONLY CONTRACT**. This slice changes zero production
behavior. It does not call providers, modify the verifier, alter routing,
change prompts/models/retrieval, or run the paid benchmark.

## 1. Purpose

Prove the metric definitions and evaluator mathematics for claim
completeness and atomicity.

The oracle measures frozen, human-adjudicated candidate sets against
frozen gold atoms. It does not extract claims from production drafts and
does not invent semantic equivalence.

Out of scope (later slices only if evidence justifies them): a 40-draft
corpus, provider-backed extraction, fuzzy/embedding/LLM matching,
production claim extraction, dashboards, and unrelated metrics.

## 2. Semantic contract

An atomic factual claim is the smallest self-contained proposition that
preserves all truth-relevant:

- subject/object
- scope
- time
- modality
- polarity
- conditions
- quantities
- units
- comparison baseline

Split propositions when two parts could reasonably receive different
evidence verdicts.

Forbidden credit:

- one candidate covering multiple independent gold claims
- multiple fragments collectively faking coverage of one claim
- duplicate rows increasing recall or a denominator
- dropped qualifiers counting as equivalent

Repeated or paraphrased equivalent occurrences share one canonical gold
claim ID. The recall denominator is the count of distinct canonical gold
IDs, not gold rows.

Non-factual prose is excluded, not scored as gold:

- opinion
- transition
- rhetorical question
- instruction/advice without a factual premise
- explicit hypothetical
- draft-own-code-only description

Each exclusion records an explicit reason.

A gold atom that is `material` must also be `factual`. Material atoms
are the subset of factual atoms that remain after excluding generic
connective facts. This lean pack marks every factual atom as material;
the two recall formulas stay distinct.

## 3. Matching

Use **only** frozen human-adjudicated `allowed_matches` edges.

- Maximum-cardinality one-to-one matching (Kuhn / DFS augmenting paths)
- Deterministic ID tie-breaking: candidates then golds are visited in
  sorted ID order
- Compound and fragment candidates are ineligible to match
- The matcher itself still assigns a candidate at most one gold even if
  multiple edges exist

The evaluator must not use embeddings, edit distance, token overlap,
LLMs, fuzzy matching, or network/provider calls.

## 4. Exact metric definitions

Let \(G\) be the set of gold rows, \(C\) the set of candidate rows, and
\(M\) the set of matched pairs \((c, g)\) from the deterministic
maximum-cardinality matching over eligible candidates.

Canonicalization:

- \(G^* = \{g.\mathrm{canonical\_id} \mid g \in G\}\)
- \(G^*_{\mathrm{mat}} = \{g.\mathrm{canonical\_id} \mid g \in G \land g.\mathrm{material}\}\)
- \(G^*_{\mathrm{fact}} = \{g.\mathrm{canonical\_id} \mid g \in G \land g.\mathrm{factual}\}\)
- \(C^* = \{c.\mathrm{canonical\_id} \mid c \in C\}\)
- \(M^*_G = \{g.\mathrm{canonical\_id} \mid (c, g) \in M\}\)
- \(M^*_C = \{c.\mathrm{canonical\_id} \mid (c, g) \in M\}\)

| Metric | Numerator | Denominator | Value |
| --- | --- | --- | --- |
| Material Claim Recall | \(\|M^*_G \cap G^*_{\mathrm{mat}}\|\) | \(\|G^*_{\mathrm{mat}}\|\) | num / den when den > 0 |
| Full Factual Recall | \(\|M^*_G \cap G^*_{\mathrm{fact}}\|\) | \(\|G^*_{\mathrm{fact}}\|\) | num / den when den > 0 |
| Extraction Precision | \(\|M^*_C\|\) | \(\|C^*\|\) | num / den when den > 0 |
| Extraction F1 | \(2 \cdot P_{\mathrm{num}} \cdot R_{\mathrm{num}}\) | \(P_{\mathrm{num}} \cdot R_{\mathrm{den}} + R_{\mathrm{num}} \cdot P_{\mathrm{den}}\) | num / den when both P and R are defined and P + R > 0 |
| Duplicate count | \(\|C\| - \|C^*\|\) | — | integer |
| Duplicate rate | \(\|C\| - \|C^*\|\) | \(\|C\|\) | num / den when den > 0 |
| Atomicity violations | count of candidate **rows** whose roles include `compound` | — | integer |
| Fragmentation violations | count of candidate **rows** whose roles include `fragment` | — | integer |

Extraction F1 uses Extraction Precision as \(P\) and Full Factual Recall
as \(R\).

Every metric object reports raw integer numerator and denominator. When
a denominator is 0, or when F1's harmonic-mean denominator is 0, the
value is `null` (N/A) plus an explicit `undefined_reason`. The evaluator
never silently returns 0 or 1 for an undefined ratio.

## 5. Zero-denominator behavior

| Situation | Material / factual recall | Precision | F1 |
| --- | --- | --- | --- |
| Empty candidates, nonempty gold | `0 / \|G^*\|` (defined 0) | undefined: zero canonical candidate claims | undefined: precision undefined |
| Zero-gold fixture, empty candidates | undefined: zero material/factual gold atoms | undefined: zero canonical candidate claims | undefined |
| Zero-gold fixture, nonempty candidates | undefined | `0 / \|C^*\|` if no matches | undefined: recall undefined |
| Precision = 0 and recall = 0 | defined 0 / defined 0 | defined 0 | undefined: harmonic mean undefined when precision and recall are both 0 |

## 6. Frozen fixture catalog

Exactly 14 fixtures. Canonical gold-atom count is 18.

| ID | Trap | Canonical gold atoms |
| --- | --- | --- |
| F01 | Adam first/second moment; omission | 2 |
| F02 | L2 repeated paraphrase; duplicate | 1 (2 gold rows, one canonical ID) |
| F03 | Dropout masks + calibration; compound | 2 |
| F04 | BatchNorm train vs infer statistics | 2 |
| F05 | Gradient clipping with year/norm/model/experiment qualifiers | 1 |
| F06 | 12 layers / width 768 / 110M parameters | 3 |
| F07 | Advice + hypothetical; zero factual atoms | 0 |
| F08 | Hypothetical accuracy + real benchmark accuracy | 1 |
| F09 | Draft-own-code description; excluded | 0 |
| F10 | Clipping can reduce explosions; does not guarantee convergence | 2 |
| F11 | Conditional BCE \(y=1\), \(p=0.8\) → \(-\log(0.8)\) | 1 |
| F12 | 8192-token context repeated by paraphrase | 1 (2 gold rows, one canonical ID) |
| F13 | Sigmoid saturation causes gradient shrink | 1 |
| F14 | Method A 2 ms faster than B on the same hardware | 1 |

Candidate sets exist only as needed to prove: perfect extraction,
omission, duplicate, invention, compound merging, fragmentation,
qualifier loss, reordered inputs, and empty extraction.

## 7. Integrity failures

The evaluator fails closed (does not score) when:

- `draft_sha256` does not equal SHA-256 of UTF-8 `draft_text`
- a gold, exclusion, or spanned candidate text does not equal
  `draft_text[start:end]`
- a span is inverted, empty, or out of range
- a `null` span appears on a non-invention / non-qualifier_loss candidate
- an `allowed_matches` edge names an unknown candidate or gold ID
- gold rows that share a `canonical_id` disagree on `factual`/`material`
- a material gold atom is not factual
- fixture IDs escape the frozen `F01`–`F14` catalog in the official pack

## 8. Determinism

Normalized output is `json.dumps(..., sort_keys=True, indent=2,
ensure_ascii=True)` plus a trailing newline. Repeated evaluation of the
same pack must be byte-identical. Candidate-set order must not change
metrics: matching pairs are sorted by `(candidate_id, gold_id)`.

## 9. Runtime boundary

Allowed files for this slice:

- `docs/P0_2B_CLAIM_SEMANTICS_V1.md`
- `evals/claim_semantics_v1.schema.json`
- `evals/fixtures/claim_semantics_v1.json`
- `scripts/evaluate_claim_semantics.py`
- `tests/test_claim_semantics_evaluator.py`

No production module is imported. Overall P0-2b remains OPEN. This
oracle does not define a production completeness contract for an
unknown-sized model-extracted claim set.

## 10. Schema authority

Two artifacts define the fixture contract. They must not drift.

- `evals/claim_semantics_v1.schema.json` is the **external/static
  contract** (JSON Schema Draft 2020-12).
- `scripts/evaluate_claim_semantics.py` is the **executable semantic
  validator** (span-text equality, draft SHA-256, uniqueness, overlap,
  matching eligibility, frozen F01-F14 catalog order).

`DEFAULT_SCHEMA` is loaded on every official pack load. A stdlib subset
checker applies that frozen schema to the instance before semantic
validation. A parity check compares schema-extracted contract fields
with the runtime validator's constants:

- required and allowed keys at pack, fixture, gold, exclusion,
  candidate, candidate-set, and allowed-match layers
- `pack_id`, `schema_version`, `evaluator_id`
- fixture cardinality
- span structure/types
- candidate roles
- exclusion reasons
- candidate-set IDs
- `additionalProperties: false`

Mismatch fails closed. This is not a general JSON Schema engine and
does not use embeddings, fuzzy matching, or provider calls. Semantic
rules that JSON Schema cannot express (hash equality, span-text
equality, gold/exclusion overlap, material-implies-factual) remain in
the Python validator.
