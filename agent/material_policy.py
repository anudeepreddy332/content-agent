"""Phase 4 Slice 2B: material-claim + required-content acceptance policy.

A SMALL deterministic policy layer around the existing semantic-verification
output. It does NOT re-derive semantic status, does NOT call any provider,
and does NOT change the LangGraph topology.

Authority split (frozen for this slice):

- ``semantic_verification_accepted()`` (nodes) remains the sole owner of
  evidence-relative semantic correctness over the current claim inventory.
- THIS module owns material / required-content publication safety: which
  factual claims are MATERIAL, whether each material claim is RESOLVED
  against the current semantic disposition, and whether every MANDATORY
  brief requirement has a current-draft basis not undermined by an
  unresolved material claim or an UNKNOWN-materiality claim. Call-A
  ``satisfies_req_ids`` is candidate/advisory linkage only — it does NOT
  prove semantic requirement fulfillment (Slice 2b P1 correction).
- Citation safety remains Slice 2c and is intentionally NOT consulted here.
  ``material_policy_pass`` is therefore NOT final publication eligibility.

The layer fails closed: stale inventories, stale/mismatched grounding rosters,
unresolved anchors, material INVALID rows, UNKNOWN materiality, unresolved
material claims, and missing/unresolved/unknown mandatory requirements all
prevent automatic progression. ``material_verified_rate`` is observability
only and can never override a single critical failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.claim_inventory import (
    MATERIAL_UNKNOWN,
    inventory_critical_failures,
)
from agent.semantic_analyzer.status_engine import sha256_utf8

MATERIAL_PASS = "material_policy_pass"
MATERIAL_REVISION = "revision_required"
MATERIAL_HITL = "hitl_required"

REQ_SATISFIED = "satisfied"
REQ_UNRESOLVED = "unresolved"
REQ_MISSING = "missing"
REQ_UNKNOWN = "unknown"

STRUCTURAL_REQ_KINDS = frozenset({"section_presence"})
BLOCKING_BLOCKER_KINDS = frozenset({"contradiction", "limitation"})
SEMANTIC_COVERAGE_UNQUALIFIED = "semantic_requirement_coverage_unqualified"


@dataclass
class MaterialPolicyResult:
    """Smallest deterministic policy result (spec §13)."""

    material_safety_state: str
    unresolved_material_claim_ids: list[str] = field(default_factory=list)
    unknown_materiality_claim_ids: list[str] = field(default_factory=list)
    invalid_material_claim_ids: list[str] = field(default_factory=list)
    mandatory_requirement_states: dict[str, str] = field(default_factory=dict)
    missing_requirement_ids: list[str] = field(default_factory=list)
    unresolved_requirement_ids: list[str] = field(default_factory=list)
    unknown_requirement_ids: list[str] = field(default_factory=list)
    material_verified_rate: float | None = None
    material_claim_count: int = 0
    resolved_material_claim_count: int = 0
    unresolved_material_claim_count: int = 0
    unknown_materiality_count: int = 0
    total_factual_claims: int = 0
    mandatory_requirements_total: int = 0
    satisfied_requirement_count: int = 0
    decision: str = MATERIAL_HITL
    reason_codes: list[str] = field(default_factory=list)
    material_claim_resolution: dict[str, str] = field(default_factory=dict)
    requirement_detail: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "material_safety_state": self.material_safety_state,
            "unresolved_material_claim_ids": list(self.unresolved_material_claim_ids),
            "unknown_materiality_claim_ids": list(self.unknown_materiality_claim_ids),
            "invalid_material_claim_ids": list(self.invalid_material_claim_ids),
            "mandatory_requirement_states": dict(self.mandatory_requirement_states),
            "missing_requirement_ids": list(self.missing_requirement_ids),
            "unresolved_requirement_ids": list(self.unresolved_requirement_ids),
            "unknown_requirement_ids": list(self.unknown_requirement_ids),
            "material_verified_rate": self.material_verified_rate,
            "material_claim_count": self.material_claim_count,
            "resolved_material_claim_count": self.resolved_material_claim_count,
            "unresolved_material_claim_count": self.unresolved_material_claim_count,
            "unknown_materiality_count": self.unknown_materiality_count,
            "total_factual_claims": self.total_factual_claims,
            "mandatory_requirements_total": self.mandatory_requirements_total,
            "satisfied_requirement_count": self.satisfied_requirement_count,
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "material_claim_resolution": dict(self.material_claim_resolution),
            "requirement_detail": {k: dict(v) for k, v in self.requirement_detail.items()},
        }


def _is_material(claim: dict) -> bool:
    return claim.get("material") is True


def _is_unknown_materiality(claim: dict) -> bool:
    return claim.get("material") == MATERIAL_UNKNOWN


def _has_applicable_blocker(row: dict) -> bool:
    for blocker in row.get("blockers") or []:
        if isinstance(blocker, dict) and blocker.get("kind") in BLOCKING_BLOCKER_KINDS:
            return True
    return False


def _resolve_material_claim(claim: dict, row: dict | None) -> tuple[bool, str]:
    """Resolve one material factual claim against its current semantic row.

    RESOLVED only if all hold (spec §2): a current semantic disposition exists,
    analysis_validity == VALID, semantic status == VERIFIED, and no
    unresolved applicable contradiction or claim-invalidating limitation.
    Current-draft version and valid anchor are checked by the caller.
    """
    if row is None:
        return False, "missing_semantic_disposition"
    if str(row.get("analysis_validity", "VALID")).upper() != "VALID":
        return False, "invalid_analysis_validity"
    if row.get("status") != "verified":
        return False, f"semantic_status_{row.get('status')}"
    if _has_applicable_blocker(row):
        return False, "unresolved_applicable_blocker"
    return True, "resolved"


def _is_deterministically_checkable(req: dict) -> bool:
    """True when existing objective machinery can establish fulfillment."""
    return req.get("kind") in STRUCTURAL_REQ_KINDS


def _structural_check(req: dict, draft_markdown: str) -> bool | None:
    """Deterministic structural check for machine-checkable requirements.

    Returns True/False when a deterministic check is available, or None when
    the requirement is not structurally checkable (caller falls back to
    claim linkage). ``section_presence``: the requirement description names a
    section/phrase that must appear in the current draft.
    """
    kind = req.get("kind")
    if kind not in STRUCTURAL_REQ_KINDS:
        return None
    description = (req.get("description") or "").strip()
    if not description:
        return None
    return description in draft_markdown


def _candidate_claim_detail(claim: dict, row: dict | None) -> dict:
    return {
        "claim_id": claim.get("claim_id"),
        "claim_text": claim.get("claim_text") or "",
        "semantic_status": (row or {}).get("status"),
        "material": claim.get("material"),
        "materiality_override": claim.get("materiality_override"),
    }


def unknown_requirement_obligations(
    result: MaterialPolicyResult,
    _grounding_report: list[dict] | None = None,
) -> list[dict]:
    """HITL payload rows for mandatory requirements in UNKNOWN state (spec §11)."""
    if not result.unknown_requirement_ids:
        return []
    obligations: list[dict] = []
    for req_id in result.unknown_requirement_ids:
        detail = result.requirement_detail.get(req_id, {})
        linked_ids = detail.get("linked_claim_ids") or []
        candidates = detail.get("candidate_claims") or []
        if not candidates and linked_ids:
            candidates = [
                {"claim_id": cid, "claim_text": "", "semantic_status": None, "material": None}
                for cid in linked_ids
            ]
        obligations.append({
            "req_id": req_id,
            "requirement": detail.get("description") or "",
            "candidate_claim_ids": linked_ids,
            "candidate_claims": candidates,
            "reason": detail.get("reason") or SEMANTIC_COVERAGE_UNQUALIFIED,
            "semantic_statuses": {
                c.get("claim_id"): c.get("semantic_status") for c in candidates if c.get("claim_id")
            },
            "materiality": {
                c.get("claim_id"): c.get("material") for c in candidates if c.get("claim_id")
            },
        })
    return obligations


def _integrity_failure(
    inventory: dict | None,
    grounding_report: list[dict] | None,
    draft_markdown: str,
) -> str | None:
    """Version/roster integrity (spec §12). Returns a reason code when the
    inputs cannot certify the current draft, else None."""
    if not isinstance(inventory, dict):
        return "missing_inventory"
    if inventory.get("draft_sha256") != sha256_utf8(draft_markdown or ""):
        return "stale_inventory_draft_sha"
    if inventory_critical_failures(inventory):
        return "unresolved_anchor"
    expected_ids = [c.get("claim_id") for c in inventory.get("claims", [])]
    observed_ids = [r.get("claim_id") for r in (grounding_report or [])]
    if any(not isinstance(i, str) or not i for i in expected_ids):
        return "grounding_roster_mismatch"
    if any(not isinstance(i, str) or not i for i in observed_ids):
        return "grounding_roster_mismatch"
    if len(expected_ids) != len(set(expected_ids)):
        return "grounding_roster_mismatch"
    if len(observed_ids) != len(set(observed_ids)):
        return "grounding_roster_mismatch"
    if set(expected_ids) != set(observed_ids):
        return "grounding_roster_mismatch"
    return None


def evaluate_material_policy(*, state: dict, max_iterations: int) -> MaterialPolicyResult:
    """Compute the material + required-content policy result for one state.

    Pure/deterministic. No provider calls. ``max_iterations`` is the revision
    budget ceiling (config.MAX_ITERATIONS); the caller passes it so this
    module stays free of config imports.
    """
    inventory = state.get("claim_inventory")
    grounding_report = state.get("grounding_report") or []
    draft_markdown = state.get("draft_markdown") or ""
    iterations = state.get("iterations", 0)

    # Absent inventory (pre-2A / hand-built test states): keep Slice-1
    # behavior only. The material/required-content layer is authoritative
    # solely when a current-draft claim inventory exists (production verify
    # always sets one). This preserves the qualified fail-safe behavior.
    if inventory is None:
        return MaterialPolicyResult(
            material_safety_state="no_inventory",
            decision=MATERIAL_PASS,
            reason_codes=[],
        )

    integrity = _integrity_failure(inventory, grounding_report, draft_markdown)
    if integrity is not None:
        return MaterialPolicyResult(
            material_safety_state="integrity_failure",
            decision=MATERIAL_HITL,
            reason_codes=[integrity],
        )

    claims = inventory.get("claims", [])
    row_by_id = {r.get("claim_id"): r for r in grounding_report}

    total_factual = sum(1 for c in claims if c.get("claim_type") == "factual")
    unknown_materiality_ids: list[str] = []
    material_claim_ids: list[str] = []
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    invalid_material_ids: list[str] = []
    resolution_detail: dict[str, str] = {}

    for claim in claims:
        claim_id = claim.get("claim_id")
        if _is_unknown_materiality(claim):
            unknown_materiality_ids.append(claim_id)
            continue
        if not _is_material(claim):
            continue
        material_claim_ids.append(claim_id)
        row = row_by_id.get(claim_id)
        resolved, reason = _resolve_material_claim(claim, row)
        resolution_detail[claim_id] = reason
        if resolved:
            resolved_ids.append(claim_id)
        else:
            unresolved_ids.append(claim_id)
            # INVALID analysis_validity is a fail-closed verifier structural
            # failure, not a content defect repairable by revision (spec §19F).
            if reason == "invalid_analysis_validity":
                invalid_material_ids.append(claim_id)

    material_verified_rate: float | None = None
    if material_claim_ids:
        material_verified_rate = round(len(resolved_ids) / len(material_claim_ids), 6)

    requirements = inventory.get("brief_requirements") or []
    mandatory_reqs = [r for r in requirements if r.get("mandatory") is True]
    req_states: dict[str, str] = {}
    missing_ids: list[str] = []
    unresolved_req_ids: list[str] = []
    unknown_req_ids: list[str] = []
    satisfied_req_ids: list[str] = []
    requirement_detail: dict[str, dict] = {}

    claims_by_req: dict[str, list[dict]] = {}
    for claim in claims:
        for req_id in claim.get("satisfies_req_ids") or []:
            claims_by_req.setdefault(req_id, []).append(claim)

    for req in mandatory_reqs:
        req_id = req.get("req_id")
        linked = claims_by_req.get(req_id, [])
        detail: dict = {
            "description": req.get("description") or "",
            "linked_claim_ids": [c.get("claim_id") for c in linked],
            "candidate_claims": [
                _candidate_claim_detail(c, row_by_id.get(c.get("claim_id")))
                for c in linked
            ],
            "reason": "",
        }
        deterministic = _is_deterministically_checkable(req)
        structural = _structural_check(req, draft_markdown) if deterministic else None

        if deterministic and structural is True:
            state_ = REQ_SATISFIED
            detail["reason"] = "structural_check_passed"
            satisfied_req_ids.append(req_id)
        elif not linked:
            state_ = REQ_MISSING
            detail["reason"] = (
                "structural_check_failed" if structural is False
                else "no_linked_current_claims"
            )
            missing_ids.append(req_id)
        elif any(_is_unknown_materiality(c) for c in linked):
            state_ = REQ_UNKNOWN
            detail["reason"] = "linked_unknown_materiality_claim"
            unknown_req_ids.append(req_id)
        else:
            material_linked = [c for c in linked if _is_material(c)]
            unresolved_material_linked = [
                c for c in material_linked if c.get("claim_id") in unresolved_ids
            ]
            if material_linked and unresolved_material_linked:
                state_ = REQ_UNRESOLVED
                detail["reason"] = "supported_by_unresolved_material_claim"
                unresolved_req_ids.append(req_id)
            elif deterministic:
                state_ = REQ_MISSING
                detail["reason"] = "structural_check_failed"
                missing_ids.append(req_id)
            else:
                # Semantic/content requirement: candidate linkage is advisory
                # only — verified material claims cannot auto-satisfy coverage.
                state_ = REQ_UNKNOWN
                detail["reason"] = SEMANTIC_COVERAGE_UNQUALIFIED
                unknown_req_ids.append(req_id)
        req_states[req_id] = state_
        requirement_detail[req_id] = detail

    unresolved_material_count = len(unresolved_ids)
    unknown_materiality_count = len(unknown_materiality_ids)
    unknown_req_count = len(unknown_req_ids)
    invalid_material_count = len(invalid_material_ids)

    reason_codes: list[str] = []
    if unknown_materiality_count:
        reason_codes.append("unknown_materiality")
    if unknown_req_count:
        reason_codes.append("unknown_mandatory_requirement")
    if invalid_material_count:
        reason_codes.append("invalid_material_claim")
    if unresolved_material_count:
        reason_codes.append("unresolved_material_claim")
    if missing_ids:
        reason_codes.append("missing_mandatory_requirement")
    if unresolved_req_ids:
        reason_codes.append("unresolved_mandatory_requirement")

    if not reason_codes:
        decision = MATERIAL_PASS
        safety_state = "pass"
    elif unknown_materiality_count or unknown_req_count or invalid_material_count:
        # UNKNOWN materiality, INVALID material rows, and UNKNOWN
        # requirements never auto-pass and never trigger a stochastic
        # revision retry (spec §3, §19F). Expose to HITL/HOLD.
        decision = MATERIAL_HITL
        safety_state = "unknown_materiality"
    elif iterations >= max_iterations:
        decision = MATERIAL_HITL
        safety_state = "exhausted"
    else:
        decision = MATERIAL_REVISION
        safety_state = "repairable"

    return MaterialPolicyResult(
        material_safety_state=safety_state,
        unresolved_material_claim_ids=unresolved_ids,
        unknown_materiality_claim_ids=unknown_materiality_ids,
        invalid_material_claim_ids=invalid_material_ids,
        mandatory_requirement_states=req_states,
        missing_requirement_ids=missing_ids,
        unresolved_requirement_ids=unresolved_req_ids,
        unknown_requirement_ids=unknown_req_ids,
        material_verified_rate=material_verified_rate,
        material_claim_count=len(material_claim_ids),
        resolved_material_claim_count=len(resolved_ids),
        unresolved_material_claim_count=unresolved_material_count,
        unknown_materiality_count=unknown_materiality_count,
        total_factual_claims=total_factual,
        mandatory_requirements_total=len(mandatory_reqs),
        satisfied_requirement_count=len(satisfied_req_ids),
        decision=decision,
        reason_codes=reason_codes,
        material_claim_resolution=resolution_detail,
        requirement_detail=requirement_detail,
    )


def material_policy_passed(result: MaterialPolicyResult) -> bool:
    """Hard authority (spec §4, §7): automatic progression requires zero
    unresolved material claims, zero UNKNOWN-materiality claims, and zero
    missing/unresolved/unknown mandatory requirements. No tolerance."""
    return result.decision == MATERIAL_PASS


def material_policy_repairable(result: MaterialPolicyResult) -> bool:
    """True when the policy failure is repairable via targeted revision
    (i.e. NOT UNKNOWN materiality and NOT an integrity failure)."""
    return result.decision == MATERIAL_REVISION


def format_required_content_feedback(result: MaterialPolicyResult) -> str:
    """Targeted revision feedback for material/requirement failures (spec §8).

    Identifies req_id, requirement description, missing/unresolved reason,
    linked claim IDs, and the relevant semantic obligations. Never a generic
    'improve completeness'. Returns "" when no repairable failure exists.
    """
    if result.decision != MATERIAL_REVISION:
        return ""

    lines: list[str] = [
        "REVISION - TARGETED MATERIAL / REQUIRED-CONTENT FEEDBACK:",
        "Verification found material factual-claim or mandatory-requirement",
        "failures. Repair EACH one specifically; this is NOT a generic",
        "completeness request. Do NOT resolve a failure by deleting required",
        "substantive content: repair, qualify, or ground the claim instead, and",
        "keep the required deliverable complete (all four sections - problem",
        "framing, technical deep-dive, code, takeaways - on the assigned",
        "topic/card). Required-content completeness is measured against the",
        "fixed brief requirements, so deleting a difficult claim does NOT",
        "remove the obligation.",
    ]

    if result.unresolved_material_claim_ids:
        lines.append("")
        lines.append("Unresolved MATERIAL factual claims (repair, do not delete):")
        for claim_id in result.unresolved_material_claim_ids:
            reason = result.material_claim_resolution.get(claim_id, "unresolved")
            lines.append(f"- claim_id: {claim_id}")
            lines.append(f"  resolution_reason: {reason}")

    req_failures = (
        [(rid, "missing") for rid in result.missing_requirement_ids]
        + [(rid, "unresolved") for rid in result.unresolved_requirement_ids]
    )
    if req_failures:
        lines.append("")
        lines.append("Mandatory requirement failures (add or repair the required content):")
        for req_id, label in req_failures:
            detail = result.requirement_detail.get(req_id, {})
            description = detail.get("description") or ""
            linked = detail.get("linked_claim_ids") or []
            reason = detail.get("reason") or label
            lines.append(f"- req_id: {req_id}")
            lines.append(f"  state: {label}")
            if description:
                lines.append(f"  requirement: {description}")
            lines.append(f"  reason: {reason}")
            if linked:
                lines.append(f"  linked_claim_ids: {', '.join(linked)}")
            else:
                lines.append("  linked_claim_ids: (none - requirement is uncovered)")

    return "\n".join(lines)

