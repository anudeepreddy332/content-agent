"""Phase 4 Slice 2A: deterministic claim inventory + draft anchoring.

Authority split (frozen for this slice):

- Call A (LLM) owns INVENTORY and MATERIALITY classification only. It never sees
  evidence/sources, so retrieval cannot influence what counts as a claim.
- This module owns everything deterministic: strict schema validation (rejecting
  model-generated IDs and offsets), exact-string draft anchoring, content-derived
  claim IDs, duplicate merge, the required⇒material override, citation-requirement
  derivation, and Call-B roster selection.
- Call B / the status engine remains the ONLY semantic verification authority.
  Nothing here assigns or influences VERIFIED/WEAK/UNVERIFIED.

Core reviewer correction implemented here: ``claim_text`` (atomic semantic
proposition adjudicated by Call B) and ``anchor_quote`` (exact verbatim draft
substring used only for location binding) are SEPARATE fields. A compound source
sentence may anchor several atomic propositions.
"""
from __future__ import annotations

import json
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.semantic_analyzer.status_engine import sha256_utf8

INVENTORY_SCHEMA_VERSION = 1

CLAIM_TYPES = frozenset({"factual", "editorial", "code", "definition"})
MATERIAL_UNKNOWN = "unknown"

ANCHOR_BOUND = "bound"
ANCHOR_BOUND_MULTIPLE = "bound_multiple"
ANCHOR_FAILED = "ANCHOR_FAILED"
ANCHOR_AMBIGUOUS = "ANCHOR_AMBIGUOUS"
ANCHOR_UNRESOLVED = frozenset({ANCHOR_FAILED, ANCHOR_AMBIGUOUS})

# 64-bit content identity: accidental collision probability is negligible at
# article-scale inventories (birthday bound ~4 billion claims).
CLAIM_ID_HASH_LEN = 16


class ClaimInventoryError(ValueError):
    """Call-A output violates the claim-inventory contract. Fail closed."""


class BriefRequirement(BaseModel):
    """Fixed brief-derived requirement identity (Slice 2b schema foundation).

    Lives in state, external to any single draft's claims. Deleting a claim on
    revision cannot remove a requirement's identity. Deterministic
    (machine-checkable) matching may come later; qualitative 'adequately
    addressed' coverage remains UNKNOWN until separately qualified.
    """

    model_config = ConfigDict(extra="forbid")
    req_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    mandatory: bool = True
    description: str = ""


class ClaimExtractorRow(BaseModel):
    """Strict Call-A output row. ``extra="forbid"`` rejects model-generated
    claim IDs, start/end offsets, and occurrence spans — those are assigned or
    derived by Python only."""

    model_config = ConfigDict(extra="forbid")
    claim_text: str = Field(min_length=1)
    anchor_quote: str = Field(min_length=1)
    section: Literal[
        "problem_framing", "technical_dive", "code_snippets", "takeaways"
    ] | None = None
    claim_type: Literal["factual", "editorial", "code", "definition"]
    material: Literal[True, False, "unknown"]
    materiality_reason_code: str | None = None
    materiality_rationale: str | None = None
    satisfies_req_ids: list[str] = Field(default_factory=list)
    specificity: Literal["substantive", "generic"] = "generic"
    requires_citation: bool | None = None


def _extract_json_array(raw: str) -> list:
    """Tolerant JSON-array extraction (code-fence strip + outermost slice)."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = s.find("["), s.rfind("]")
    if start != -1 and end > start:
        parsed = json.loads(s[start:end + 1])
        if isinstance(parsed, list):
            return parsed
    raise ClaimInventoryError("no JSON array found in claim-inventory output")


def parse_claim_inventory_rows(raw: str) -> list[dict]:
    """Strictly parse + validate Call-A output. Any schema violation fails the
    whole extraction (fail closed) — no partial inventories."""
    items = _extract_json_array(raw)
    try:
        return [ClaimExtractorRow.model_validate(item).model_dump() for item in items]
    except ValidationError as exc:
        raise ClaimInventoryError(f"claim-inventory schema invalid: {exc}") from exc


def canonical_claim_text(text: str) -> str:
    """NFC + whitespace collapse + strip. Case is significant."""
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def compute_claim_id(claim_text: str) -> str:
    """Deterministic content-derived logical claim identity. Python-assigned;
    model-supplied IDs are never consulted. A changed proposition hashes to a
    changed identity."""
    return "clm-" + sha256_utf8(canonical_claim_text(claim_text))[:CLAIM_ID_HASH_LEN]


def bind_anchor_occurrences(draft_markdown: str, anchor_quote: str) -> list[tuple[int, int]]:
    """Exact-string binding of ``anchor_quote`` in the exact current
    ``draft_markdown``. Zero-based half-open [start, end) spans, all matches in
    document order. No fuzzy matching, no model offsets."""
    if not anchor_quote:
        return []
    spans: list[tuple[int, int]] = []
    start = draft_markdown.find(anchor_quote)
    while start != -1:
        spans.append((start, start + len(anchor_quote)))
        start = draft_markdown.find(anchor_quote, start + 1)
    return spans


def _row_anchor_validity(claim_text: str, anchor_quote: str, occurrences: list[tuple[int, int]]) -> str:
    """Deterministic anchor disposition for one extractor row.

    - 0 matches -> ANCHOR_FAILED (retained in inventory; fails closed later).
    - 1 match  -> bound.
    - >1 matches: when the canonical proposition IS the anchor text, every
      occurrence restates the same claim -> bound_multiple with all occurrences.
      When the proposition is a normalized/derived form of a repeated anchor,
      the intended occurrence set cannot be established deterministically ->
      ANCHOR_AMBIGUOUS (unresolved; never silently dropped).
    """
    if not occurrences:
        return ANCHOR_FAILED
    if len(occurrences) == 1:
        return ANCHOR_BOUND
    if canonical_claim_text(claim_text) == canonical_claim_text(anchor_quote):
        return ANCHOR_BOUND_MULTIPLE
    return ANCHOR_AMBIGUOUS


def call_b_eligible(claim_type: str, material: Any) -> bool:
    """Return whether a Call-A inventory row enters Call B.

    Every successfully anchored inventory claim is semantically adjudicated.
    ``claim_type`` and ``material`` remain descriptive/policy metadata only:
    model-produced metadata must never decide whether a proposition receives a
    semantic disposition.  Anchor resolution is checked separately so an
    unresolved row fails the inventory closed rather than silently vanishing.

    The arguments remain for compatibility with existing callers and fixtures;
    neither influences eligibility.
    """
    del claim_type, material
    return True


def _derive_requires_citation(claim_type: str, material: Any, model_value: Any) -> bool:
    """Reviewer policy: MATERIAL + FACTUAL => requires citation. Period.

    UNKNOWN materiality is never converted to false; for factual claims it
    fails toward requiring citation (favor over-citation until Slice 2c). A
    material definition carries checkable content, so it requires citation too.
    For nonmaterial factual claims the model's explicit classification is
    preserved, but it is NOT publication-policy authority in this slice.
    """
    if claim_type == "factual" and material is not False:
        return True
    if claim_type == "definition" and material is True:
        return True
    return bool(model_value)


def normalize_brief_requirements(raw: Any) -> list[dict]:
    """Validate/normalize the state-carried brief requirements. Malformed
    entries are dropped (internal trusted input, but never trusted blindly)."""
    out: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            out.append(BriefRequirement.model_validate(item).model_dump())
        except ValidationError:
            continue
    return out


def build_claim_inventory(
    *,
    run_id: str,
    iteration: int,
    draft_markdown: str,
    raw_claims: list[dict],
    brief_requirements: list[dict] | None = None,
) -> dict:
    """Build the canonical ClaimInventory for ONE exact draft version.

    Pure/deterministic: anchors are bound against ``draft_markdown`` as given,
    IDs derive from canonical claim_text, duplicates of one logical proposition
    merge into one claim with multiple occurrence anchors, and the
    required⇒material override is applied after Call-A classification.

    Identity: (run_id, iteration, draft_sha256). An inventory built from draft N
    can never certify draft N+1 — the sha is recomputed every verify pass.
    """
    requirements = normalize_brief_requirements(brief_requirements)
    mandatory_req_ids = {r["req_id"] for r in requirements if r.get("mandatory") is True}

    merged: dict[str, dict] = {}
    order: list[str] = []
    for row in raw_claims:
        claim_text = row["claim_text"].strip()
        anchor_quote = row["anchor_quote"]
        claim_id = compute_claim_id(claim_text)
        occurrences = bind_anchor_occurrences(draft_markdown, anchor_quote)
        validity = _row_anchor_validity(claim_text, anchor_quote, occurrences)

        if claim_id not in merged:
            order.append(claim_id)
            merged[claim_id] = {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "anchor_quote": anchor_quote,
                "occurrences": [],
                "anchor_validity": ANCHOR_FAILED,
                "section": row.get("section"),
                "claim_type": row["claim_type"],
                "material": row["material"],
                "materiality_reason_code": row.get("materiality_reason_code"),
                "materiality_rationale": row.get("materiality_rationale"),
                "materiality_override": None,
                "requires_citation": None,
                "satisfies_req_ids": [],
                "specificity": row.get("specificity", "generic"),
                "_model_requires_citation": row.get("requires_citation"),
            }
        claim = merged[claim_id]
        # Merge repeated instances of one logical proposition: union of
        # occurrence spans; material fails toward true/unknown, never down to
        # false; req linkage unions.
        seen = {(o["start"], o["end"]) for o in claim["occurrences"]}
        for start, end in occurrences:
            if (start, end) not in seen:
                seen.add((start, end))
                claim["occurrences"].append({"start": start, "end": end})
        claim["occurrences"].sort(key=lambda o: o["start"])
        if validity == ANCHOR_AMBIGUOUS or claim["anchor_validity"] == ANCHOR_AMBIGUOUS:
            claim["anchor_validity"] = ANCHOR_AMBIGUOUS
        elif claim["occurrences"]:
            claim["anchor_validity"] = (
                ANCHOR_BOUND if len(claim["occurrences"]) == 1 else ANCHOR_BOUND_MULTIPLE
            )
        if row["material"] is True:
            claim["material"] = True
        elif row["material"] == MATERIAL_UNKNOWN and claim["material"] is not True:
            claim["material"] = MATERIAL_UNKNOWN
        for req_id in row.get("satisfies_req_ids") or []:
            if req_id not in claim["satisfies_req_ids"]:
                claim["satisfies_req_ids"].append(req_id)
        if row.get("materiality_reason_code") and not claim["materiality_reason_code"]:
            claim["materiality_reason_code"] = row["materiality_reason_code"]
        if row.get("materiality_rationale") and not claim["materiality_rationale"]:
            claim["materiality_rationale"] = row["materiality_rationale"]

    claims: list[dict] = []
    for claim_id in order:
        claim = merged[claim_id]
        # Deterministic override with precedence: a claim linked to a mandatory
        # brief requirement is material regardless of Call-A classification.
        if set(claim["satisfies_req_ids"]) & mandatory_req_ids and claim["material"] is not True:
            claim["material"] = True
            claim["materiality_override"] = "required_by_brief"
        claim["requires_citation"] = _derive_requires_citation(
            claim["claim_type"], claim["material"], claim.pop("_model_requires_citation")
        )
        claim["call_b_eligible"] = call_b_eligible(claim["claim_type"], claim["material"])
        claims.append(claim)

    satisfied = sorted(
        {req_id for claim in claims for req_id in claim["satisfies_req_ids"]} & mandatory_req_ids
    )
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "run_id": run_id,
        "iteration": iteration,
        "draft_sha256": sha256_utf8(draft_markdown),
        "brief_requirements": requirements,
        "claims": claims,
        "satisfied_req_ids": satisfied,
        "counts": inventory_counts(claims),
    }


def inventory_counts(claims: list[dict]) -> dict:
    return {
        "total": len(claims),
        "by_type": {t: sum(1 for c in claims if c["claim_type"] == t) for t in sorted(CLAIM_TYPES)},
        "material_true": sum(1 for c in claims if c["material"] is True),
        "material_false": sum(1 for c in claims if c["material"] is False),
        "material_unknown": sum(1 for c in claims if c["material"] == MATERIAL_UNKNOWN),
        "call_b_eligible": sum(1 for c in claims if c["call_b_eligible"]),
        "anchor_failed": sum(1 for c in claims if c["anchor_validity"] == ANCHOR_FAILED),
        "anchor_ambiguous": sum(1 for c in claims if c["anchor_validity"] == ANCHOR_AMBIGUOUS),
    }


def eligible_claims(inventory: dict) -> list[dict]:
    return [c for c in (inventory or {}).get("claims", []) if c.get("call_b_eligible")]


def inventory_critical_failures(inventory: dict) -> list[dict]:
    """Inventory claims whose draft anchor is unresolved.

    Every inventory claim is eligible for Call B, so an unresolved anchor is
    always acceptance-critical and prevents silent omission.
    """
    return [
        c for c in eligible_claims(inventory)
        if c.get("anchor_validity") in ANCHOR_UNRESOLVED
    ]


def call_b_roster(inventory: dict) -> list[dict]:
    """Canonical current-version factual claims for Call B.

    ``claim_text`` is the proposition adjudicated semantically; the anchor is
    NEVER a substitute. claim_span is the first bound occurrence (eligible
    claims are anchor-bound by the time this is called)."""
    roster: list[dict] = []
    for claim in eligible_claims(inventory):
        first = claim["occurrences"][0]
        roster.append({
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "claim_span": [first["start"], first["end"]],
        })
    return roster
