"""Phase 4 Slice 2C: citation correctness, completeness, and placement.

A SMALL deterministic policy layer around Call-B bound support. It does NOT
call any provider, does NOT introduce a citation LLM judge, and does NOT
mutate the canonical verified draft.

Authority split (frozen for this slice):

- ``semantic_verification_accepted()`` remains the sole owner of evidence-
  relative semantic correctness.
- ``agent/material_policy.py`` remains the owner of material / required-content
  safety.
- THIS module owns citation safety: which current material+VERIFIED claims
  require citation, the COMPLETE Call-B support-evidence set for each exact
  anchor occurrence, whether rendered attachments cover that set without
  extras, and whether those attachments sit at the verified occurrence.
- Model-declared citation IDs / claim↔source tags are UNTRUSTED presentation
  hints. They satisfy policy only when they reconcile exactly with the
  authoritative plan. They never become support authority.

``citation_policy_pass`` is one conjunct of ``publication_safety_pass``. It is
NOT a human publication decision and does NOT autonomously publish.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent.html_policy import normalize_citation_url
from agent.material_policy import _integrity_failure
from agent.semantic_analyzer.contract import KB_SOURCE_LIMIT, WEB_SOURCE_LIMIT
from agent.semantic_analyzer.status_engine import sha256_utf8

CITATION_PASS = "citation_policy_pass"
CITATION_HITL = "citation_hitl_required"

# Model-written citation IDs in draft Markdown. Untrusted unless they
# reconcile exactly with the authoritative plan.
MODEL_CITE_RE = re.compile(r"\[((?:WEB|KB)-\d{3}|E\d+)\]")

# Sentinel inserted into a TEMPORARY markdown copy (never written back to
# canonical draft_markdown). High-offset-first insertion uses this mark.
CITE_MARK_PREFIX = "⟦cite:"
CITE_MARK_SUFFIX = "⟧"


@dataclass
class CitationPolicyResult:
    """Smallest deterministic citation-policy result (spec §18)."""

    citation_safety_state: str
    applicable_claim_ids: list[str] = field(default_factory=list)
    anchor_groups: list[dict] = field(default_factory=list)
    required_evidence_ids: list[str] = field(default_factory=list)
    attached_evidence_ids: list[str] = field(default_factory=list)
    missing_citation_obligations: list[dict] = field(default_factory=list)
    invalid_citation_obligations: list[dict] = field(default_factory=list)
    placement_failures: list[dict] = field(default_factory=list)
    dangling_citations: list[str] = field(default_factory=list)
    stale_plan_failures: list[str] = field(default_factory=list)
    citation_coverage_rate: float | None = None
    invalid_citation_count: int = 0
    missing_required_citation_count: int = 0
    citation_placement_failure_count: int = 0
    dangling_citation_count: int = 0
    applicable_material_claim_count: int = 0
    citation_anchor_group_count: int = 0
    total_required_support_evidence_ids: int = 0
    total_rendered_evidence_ids: int = 0
    decision: str = CITATION_HITL
    reason_codes: list[str] = field(default_factory=list)
    citation_plan: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "citation_safety_state": self.citation_safety_state,
            "applicable_claim_ids": list(self.applicable_claim_ids),
            "anchor_groups": [dict(g) for g in self.anchor_groups],
            "required_evidence_ids": list(self.required_evidence_ids),
            "attached_evidence_ids": list(self.attached_evidence_ids),
            "missing_citation_obligations": [dict(o) for o in self.missing_citation_obligations],
            "invalid_citation_obligations": [dict(o) for o in self.invalid_citation_obligations],
            "placement_failures": [dict(p) for p in self.placement_failures],
            "dangling_citations": list(self.dangling_citations),
            "stale_plan_failures": list(self.stale_plan_failures),
            "citation_coverage_rate": self.citation_coverage_rate,
            "invalid_citation_count": self.invalid_citation_count,
            "missing_required_citation_count": self.missing_required_citation_count,
            "citation_placement_failure_count": self.citation_placement_failure_count,
            "dangling_citation_count": self.dangling_citation_count,
            "applicable_material_claim_count": self.applicable_material_claim_count,
            "citation_anchor_group_count": self.citation_anchor_group_count,
            "total_required_support_evidence_ids": self.total_required_support_evidence_ids,
            "total_rendered_evidence_ids": self.total_rendered_evidence_ids,
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "citation_plan": dict(self.citation_plan),
        }


def _is_material(claim: dict) -> bool:
    return claim.get("material") is True


def _is_citation_applicable(claim: dict, row: dict | None) -> bool:
    """material + current VALID VERIFIED disposition. claim_type cannot exempt."""
    if not _is_material(claim):
        return False
    if row is None:
        return False
    if str(row.get("analysis_validity", "VALID")).upper() != "VALID":
        return False
    return row.get("status") == "verified"


def _support_evidence_ids(row: dict | None) -> list[str]:
    """COMPLETE distinct Call-B support set. Not first-support only."""
    if not row:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for span in row.get("support_spans") or []:
        if not isinstance(span, dict):
            continue
        eid = span.get("evidence_id")
        if not isinstance(eid, str) or not eid or eid in seen:
            continue
        seen.add(eid)
        ordered.append(eid)
    return ordered


def _occurrence_spans(claim: dict) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for occ in claim.get("occurrences") or []:
        if isinstance(occ, dict):
            start, end = occ.get("start"), occ.get("end")
        elif isinstance(occ, (list, tuple)) and len(occ) == 2:
            start, end = occ[0], occ[1]
        else:
            continue
        if isinstance(start, int) and isinstance(end, int) and end > start:
            spans.append((start, end))
    return spans


def _spans_overlap_nonidentical(a: tuple[int, int], b: tuple[int, int]) -> bool:
    if a == b:
        return False
    return a[0] < b[1] and b[0] < a[1]


def build_evidence_registry(state: dict) -> dict[str, dict]:
    """Current canonical evidence items. Prefer an explicit test registry,
    then reconstruct WEB-00N / KB-00N from retrieved sources (production
    manifest identity). Support-span IDs without metadata still register
    as provenance-only entries so bound Call-B support can resolve."""
    registry: dict[str, dict] = {}
    provided = state.get("evidence_registry") or {}
    if isinstance(provided, dict):
        for eid, meta in provided.items():
            if isinstance(eid, str) and eid:
                registry[eid] = dict(meta) if isinstance(meta, dict) else {"evidence_id": eid}

    web_sources = list(state.get("web_sources") or [])[:WEB_SOURCE_LIMIT]
    for index, source in enumerate(web_sources, start=1):
        eid = f"WEB-{index:03d}"
        if eid not in registry:
            url = source.get("url")
            registry[eid] = {
                "evidence_id": eid,
                "kind": "web",
                "url": url if isinstance(url, str) else None,
                "title": source.get("title"),
                "source_ref": url if isinstance(url, str) else None,
            }

    kb_results = list(state.get("kb_results") or [])[:KB_SOURCE_LIMIT]
    for index, result in enumerate(kb_results, start=1):
        eid = f"KB-{index:03d}"
        if eid not in registry:
            source = result.get("source") or "KB"
            registry[eid] = {
                "evidence_id": eid,
                "kind": "kb",
                "url": None,
                "title": source,
                "source_ref": f"kb:{source}",
            }

    for row in state.get("grounding_report") or []:
        for span in (row.get("support_spans") or []):
            if not isinstance(span, dict):
                continue
            eid = span.get("evidence_id")
            if isinstance(eid, str) and eid and eid not in registry:
                registry[eid] = {
                    "evidence_id": eid,
                    "kind": "unknown",
                    "url": None,
                    "title": None,
                    "source_ref": None,
                    "from_support_span": True,
                }
    return registry


def _roster_ids(inventory: dict, grounding_report: list) -> tuple[list[str], list[str]]:
    expected = [c.get("claim_id") for c in inventory.get("claims", [])]
    observed = [r.get("claim_id") for r in grounding_report]
    return expected, observed


def _plan_stale_reasons(plan: dict, inventory: dict, grounding_report: list, draft_sha: str) -> list[str]:
    reasons: list[str] = []
    if plan.get("draft_sha256") != draft_sha:
        reasons.append("stale_citation_plan_draft_sha")
    expected, observed = _roster_ids(inventory, grounding_report)
    plan_claims = list(plan.get("claim_roster_ids") or [])
    plan_grounding = list(plan.get("grounding_roster_ids") or [])
    if plan_claims and plan_claims != expected:
        reasons.append("stale_citation_plan_claim_roster")
    if plan_grounding and plan_grounding != observed:
        reasons.append("stale_citation_plan_grounding_roster")
    return reasons


def parse_model_written_citations(draft_markdown: str) -> list[dict]:
    """Untrusted draft citation hints. Offsets are in the canonical draft."""
    found: list[dict] = []
    for match in MODEL_CITE_RE.finditer(draft_markdown or ""):
        found.append({
            "evidence_id": match.group(1),
            "start": match.start(),
            "end": match.end(),
        })
    return found


def _attach_model_cites_to_groups(
    groups: list[dict],
    model_cites: list[dict],
) -> tuple[dict[tuple[int, int], list[str]], list[str]]:
    """Map model citations onto exact anchor groups when they sit at/after the
    occurrence. Citations that do not bind to a group remain unplaced."""
    by_group: dict[tuple[int, int], list[str]] = {}
    claimed_indexes: set[int] = set()
    for group in groups:
        key = (group["anchor_start"], group["anchor_end"])
        bound: list[str] = []
        for idx, cite in enumerate(model_cites):
            if idx in claimed_indexes:
                continue
            # Adjacent: inside the span or immediately after the occurrence.
            if group["anchor_start"] <= cite["start"] <= group["anchor_end"] + 1:
                bound.append(cite["evidence_id"])
                claimed_indexes.add(idx)
        if bound:
            by_group[key] = bound
    unplaced = [
        cite["evidence_id"] for idx, cite in enumerate(model_cites) if idx not in claimed_indexes
    ]
    return by_group, unplaced


def build_citation_plan(state: dict) -> dict:
    """Deterministic sidecar: required support union per exact anchor group.

    Production attachments equal the complete required set. Canonical
    ``draft_markdown`` is not mutated.
    """
    inventory = state.get("claim_inventory") or {}
    grounding_report = state.get("grounding_report") or []
    draft_markdown = state.get("draft_markdown") or ""
    draft_sha = sha256_utf8(draft_markdown)
    row_by_id = {r.get("claim_id"): r for r in grounding_report}
    expected, observed = _roster_ids(inventory, grounding_report)

    grouped: dict[tuple[int, int], dict] = {}
    applicable: list[str] = []
    for claim in inventory.get("claims") or []:
        claim_id = claim.get("claim_id")
        row = row_by_id.get(claim_id)
        if not _is_citation_applicable(claim, row):
            continue
        applicable.append(claim_id)
        support = _support_evidence_ids(row)
        for start, end in _occurrence_spans(claim):
            key = (start, end)
            bucket = grouped.setdefault(key, {
                "anchor_start": start,
                "anchor_end": end,
                "claim_ids": [],
                "required_evidence_ids": [],
            })
            if claim_id not in bucket["claim_ids"]:
                bucket["claim_ids"].append(claim_id)
            for eid in support:
                if eid not in bucket["required_evidence_ids"]:
                    bucket["required_evidence_ids"].append(eid)

    groups = [grouped[k] for k in sorted(grouped)]
    for group in groups:
        required = list(group["required_evidence_ids"])
        group["attached_evidence_ids"] = list(required)
        group["claim_ids"] = list(group["claim_ids"])

    return {
        "draft_sha256": draft_sha,
        "claim_roster_ids": list(expected),
        "grounding_roster_ids": list(observed),
        "applicable_claim_ids": applicable,
        "anchor_groups": groups,
    }


def _hitl_result(*, state_name: str, reasons: list[str], plan: dict | None = None, **extra: Any) -> CitationPolicyResult:
    result = CitationPolicyResult(
        citation_safety_state=state_name,
        decision=CITATION_HITL,
        reason_codes=list(reasons),
        citation_plan=plan or {},
    )
    for key, value in extra.items():
        setattr(result, key, value)
    return result


def evaluate_citation_policy(*, state: dict) -> CitationPolicyResult:
    """Compute citation correctness / completeness / placement for one state.

    Pure/deterministic. Zero provider calls. Recomputed from the current
    inventory + grounding report + draft sha unless a bound sidecar plan is
    supplied (and then that plan must match current identity).
    """
    inventory = state.get("claim_inventory")
    grounding_report = state.get("grounding_report") or []
    draft_markdown = state.get("draft_markdown") or ""

    # Absent inventory (pre-2A / hand-built): keep Slice-1 citation HTML path.
    # Slice 2c is authoritative only when a current-draft inventory exists.
    if inventory is None:
        return CitationPolicyResult(
            citation_safety_state="no_inventory",
            decision=CITATION_PASS,
            reason_codes=[],
        )

    integrity = _integrity_failure(inventory, grounding_report, draft_markdown)
    if integrity is not None:
        return _hitl_result(state_name="integrity_failure", reasons=[integrity])

    draft_sha = sha256_utf8(draft_markdown)
    supplied_plan = state.get("citation_plan")
    stale: list[str] = []
    if isinstance(supplied_plan, dict) and supplied_plan:
        stale = _plan_stale_reasons(supplied_plan, inventory, grounding_report, draft_sha)
        if stale:
            return _hitl_result(
                state_name="stale_citation_plan",
                reasons=list(stale),
                stale_plan_failures=list(stale),
                citation_plan=supplied_plan,
            )
        plan = supplied_plan
    else:
        plan = build_citation_plan(state)

    registry = build_evidence_registry(state)
    groups = [dict(g) for g in plan.get("anchor_groups") or []]

    # Overlapping non-identical applicable anchors: do not guess placement.
    spans = [(g["anchor_start"], g["anchor_end"]) for g in groups]
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            if _spans_overlap_nonidentical(a, b):
                return _hitl_result(
                    state_name="placement_integrity_failure",
                    reasons=["overlapping_nonidentical_anchors"],
                    plan=plan,
                    applicable_claim_ids=list(plan.get("applicable_claim_ids") or []),
                    anchor_groups=groups,
                    placement_failures=[{
                        "reason": "overlapping_nonidentical_anchors",
                        "spans": [list(a), list(b)],
                    }],
                    citation_placement_failure_count=1,
                )

    overlay = state.get("citation_render_overlay")
    overlay_by_key: dict[tuple[int, int], list[str]] = {}
    if isinstance(overlay, list):
        for item in overlay:
            if not isinstance(item, dict):
                continue
            key = (item.get("anchor_start"), item.get("anchor_end"))
            ids = [e for e in (item.get("evidence_ids") or []) if isinstance(e, str) and e]
            if isinstance(key[0], int) and isinstance(key[1], int):
                overlay_by_key[key] = ids

    model_cites = parse_model_written_citations(draft_markdown)
    model_by_group, unplaced_model = _attach_model_cites_to_groups(groups, model_cites)

    missing: list[dict] = []
    invalid: list[dict] = []
    placement: list[dict] = []
    dangling: list[str] = []
    all_required: list[str] = []
    all_attached: list[str] = []
    covered_required = 0
    total_required_slots = 0

    evaluated_groups: list[dict] = []
    for group in groups:
        key = (group["anchor_start"], group["anchor_end"])
        required = list(group.get("required_evidence_ids") or [])
        if overlay_by_key:
            attached = list(overlay_by_key.get(key, []))
        else:
            attached = list(group.get("attached_evidence_ids") or required)
        model_here = list(model_by_group.get(key, []))
        # Model extras join the rendered set for correctness. They cannot
        # satisfy completeness unless they reconcile exactly and no overlay
        # is substituting a deficient rendering.
        rendered = list(attached)
        for eid in model_here:
            if eid not in rendered:
                rendered.append(eid)

        required_set = set(required)
        rendered_set = set(rendered)
        missing_ids = [e for e in required if e not in rendered_set]
        extra_ids = [e for e in rendered if e not in required_set]

        for eid in required:
            if eid not in all_required:
                all_required.append(eid)
        for eid in rendered:
            if eid not in all_attached:
                all_attached.append(eid)

        total_required_slots += len(required_set)
        covered_required += len(required_set & rendered_set)

        if missing_ids:
            missing.append({
                "anchor_start": group["anchor_start"],
                "anchor_end": group["anchor_end"],
                "claim_ids": list(group.get("claim_ids") or []),
                "missing_evidence_ids": missing_ids,
            })
        elif not required:
            missing.append({
                "anchor_start": group["anchor_start"],
                "anchor_end": group["anchor_end"],
                "claim_ids": list(group.get("claim_ids") or []),
                "missing_evidence_ids": [],
                "reason": "empty_support_basis",
            })
        if extra_ids:
            invalid.append({
                "anchor_start": group["anchor_start"],
                "anchor_end": group["anchor_end"],
                "claim_ids": list(group.get("claim_ids") or []),
                "invalid_evidence_ids": extra_ids,
                "reason": "unsupported_extra_citation",
            })
        # Placement: a required ID that appears only away from this occurrence
        # does not satisfy this group. Overlay attaching elsewhere is a
        # placement failure when this group has required IDs but no local attach.
        if required and not rendered:
            elsewhere = [
                eid for eid in required
                if any(eid in ids and ok != key for ok, ids in overlay_by_key.items())
                or eid in unplaced_model
            ]
            placement.append({
                "anchor_start": group["anchor_start"],
                "anchor_end": group["anchor_end"],
                "claim_ids": list(group.get("claim_ids") or []),
                "evidence_ids": elsewhere or list(required),
                "reason": "citation_not_at_occurrence",
            })

        for eid in rendered:
            if eid not in registry and eid not in dangling:
                dangling.append(eid)

        evaluated_groups.append({
            **group,
            "attached_evidence_ids": rendered,
            "required_evidence_ids": required,
        })

    # Non-material / non-applicable rendered extras: overlay or model cites
    # attached to a span that is not an applicable group still must be valid.
    applicable_keys = {(g["anchor_start"], g["anchor_end"]) for g in groups}
    if overlay_by_key:
        for key, ids in overlay_by_key.items():
            if key in applicable_keys:
                continue
            if not ids:
                continue
            # A citation rendered at a non-applicable occurrence: still must
            # resolve and belong to SOME current support basis at that span.
            row_by_id = {r.get("claim_id"): r for r in grounding_report}
            local_support: set[str] = set()
            for claim in inventory.get("claims") or []:
                if key in set(_occurrence_spans(claim)):
                    local_support.update(_support_evidence_ids(row_by_id.get(claim.get("claim_id"))))
            extras = [e for e in ids if e not in local_support]
            for eid in extras:
                invalid.append({
                    "anchor_start": key[0],
                    "anchor_end": key[1],
                    "claim_ids": [],
                    "invalid_evidence_ids": [eid],
                    "reason": "unsupported_extra_citation",
                })
                if eid not in all_attached:
                    all_attached.append(eid)
                if eid not in registry and eid not in dangling:
                    dangling.append(eid)

    if unplaced_model:
        # Cited elsewhere: if the ID is required at some group that did not
        # receive it locally, that is a placement miss (already recorded when
        # rendered was empty). Remaining unplaced IDs are unsupported extras
        # unless they reconcile with a group's required set AND that group
        # already has them attached (duplicate mention elsewhere is allowed
        # for the same source, but does not satisfy a missing local cluster).
        required_anywhere = set(all_required)
        for eid in unplaced_model:
            if eid not in all_attached:
                all_attached.append(eid)
            if eid not in registry and eid not in dangling:
                dangling.append(eid)
            if eid in required_anywhere:
                # Does not satisfy placement for groups that needed it locally.
                still_missing = any(
                    eid in (m.get("missing_evidence_ids") or []) for m in missing
                )
                if still_missing:
                    placement.append({
                        "anchor_start": None,
                        "anchor_end": None,
                        "claim_ids": [],
                        "evidence_ids": [eid],
                        "reason": "citation_not_at_occurrence",
                    })
                # If already attached at the occurrence, an extra elsewhere is
                # the same source at multiple sites — allowed (spec §24).
            else:
                invalid.append({
                    "anchor_start": None,
                    "anchor_end": None,
                    "claim_ids": [],
                    "invalid_evidence_ids": [eid],
                    "reason": "unsupported_extra_citation",
                })

    # Model IDs that disagree with authoritative support at a group cannot
    # satisfy completeness (completeness already uses attached/overlay, not
    # model-only). Disagreement extras are already in invalid[].

    missing_count = 0
    for item in missing:
        n = len(item.get("missing_evidence_ids") or [])
        missing_count += n if n else (1 if item.get("reason") == "empty_support_basis" else 0)
    invalid_count = sum(len(i["invalid_evidence_ids"]) for i in invalid)
    placement_count = len(placement)
    dangling_count = len(dangling)

    reasons: list[str] = []
    if missing_count:
        reasons.append("missing_required_citation")
    if invalid_count:
        reasons.append("unsupported_extra_citation")
    if placement_count:
        reasons.append("citation_placement_failure")
    if dangling_count:
        reasons.append("dangling_citation")

    coverage = None
    if total_required_slots:
        coverage = round(covered_required / total_required_slots, 6)

    applicable_ids = list(plan.get("applicable_claim_ids") or [])
    passed = not reasons
    result = CitationPolicyResult(
        citation_safety_state="citation_policy_pass" if passed else "citation_hitl_required",
        applicable_claim_ids=applicable_ids,
        anchor_groups=evaluated_groups,
        required_evidence_ids=all_required,
        attached_evidence_ids=all_attached,
        missing_citation_obligations=missing,
        invalid_citation_obligations=invalid,
        placement_failures=placement,
        dangling_citations=dangling,
        stale_plan_failures=[],
        citation_coverage_rate=coverage,
        invalid_citation_count=invalid_count,
        missing_required_citation_count=missing_count,
        citation_placement_failure_count=placement_count,
        dangling_citation_count=dangling_count,
        applicable_material_claim_count=len(applicable_ids),
        citation_anchor_group_count=len(evaluated_groups),
        total_required_support_evidence_ids=len(all_required),
        total_rendered_evidence_ids=len(all_attached),
        decision=CITATION_PASS if passed else CITATION_HITL,
        reason_codes=reasons,
        citation_plan=plan,
    )
    return result


def citation_policy_passed(result: CitationPolicyResult) -> bool:
    """Hard authority: raw zero-counts, never a coverage-rate threshold."""
    return (
        result.decision == CITATION_PASS
        and result.invalid_citation_count == 0
        and result.missing_required_citation_count == 0
        and result.citation_placement_failure_count == 0
        and result.dangling_citation_count == 0
        and not result.stale_plan_failures
    )


def insert_citation_markers(draft_markdown: str, groups: list[dict]) -> str:
    """Insert citation sentinels into a TEMPORARY copy, highest offset first.

    Does not mutate the caller's string in place; returns a new string.
    """
    text = draft_markdown
    ordered = sorted(
        (g for g in groups if g.get("attached_evidence_ids")),
        key=lambda g: (g["anchor_end"], g["anchor_start"]),
        reverse=True,
    )
    for group in ordered:
        ids = list(group["attached_evidence_ids"])
        mark = f"{CITE_MARK_PREFIX}{','.join(ids)}{CITE_MARK_SUFFIX}"
        end = group["anchor_end"]
        if not isinstance(end, int) or end < 0 or end > len(text):
            continue
        text = text[:end] + mark + text[end:]
    return text


def evidence_numbering(evidence_ids: list[str]) -> dict[str, int]:
    """Stable 1-based numbers for cited evidence. Sorted, deduplicated."""
    unique = sorted(dict.fromkeys(evidence_ids))
    return {eid: index for index, eid in enumerate(unique, start=1)}


def render_inline_cluster_html(evidence_ids: list[str], numbering: dict[str, int]) -> str:
    """Link-free in-body cluster (article fragments cannot contain <a>)."""
    nums = []
    seen: set[int] = set()
    for eid in evidence_ids:
        num = numbering.get(eid)
        if num is None or num in seen:
            continue
        seen.add(num)
        nums.append(str(num))
    if not nums:
        return ""
    return "<sup>" + ",".join(nums) + "</sup>"


def apply_inline_clusters_to_html(
    html: str,
    *,
    draft_markdown: str,
    groups: list[dict],
    numbering: dict[str, int],
) -> str:
    """Place clusters after the matching occurrence text in already-sanitized
    HTML. Identical quotes use occurrence order. Highest markdown offset first
    so earlier matches are not shifted before later insertions in HTML that
    preserves source order; we still search by nth-occurrence identity."""
    import html as html_module

    # Process from the end of the document's matching occurrences so earlier
    # identical quotes keep stable indexes.
    work = html
    ordered = sorted(
        (g for g in groups if g.get("attached_evidence_ids")),
        key=lambda g: (g["anchor_start"], g["anchor_end"]),
        reverse=True,
    )
    for group in ordered:
        start, end = group["anchor_start"], group["anchor_end"]
        quote = draft_markdown[start:end]
        if not quote:
            continue
        cluster = render_inline_cluster_html(group["attached_evidence_ids"], numbering)
        if not cluster:
            continue
        needle = html_module.escape(quote)
        nth = draft_markdown[:start].count(quote)
        idx = -1
        cursor = 0
        for _ in range(nth + 1):
            idx = work.find(needle, cursor)
            if idx < 0:
                break
            cursor = idx + len(needle)
        if idx < 0:
            continue
        insert_at = idx + len(needle)
        work = work[:insert_at] + cluster + work[insert_at:]
    return work


def bibliography_items(
    evidence_ids: list[str],
    registry: dict[str, dict],
) -> list[dict[str, str | None]]:
    """Cited-only bibliography rows. Never include uncited retrieved sources."""
    items: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for eid in evidence_ids:
        if eid in seen:
            continue
        seen.add(eid)
        meta = registry.get(eid) or {}
        kind = str(meta.get("kind") or "unknown")
        title = meta.get("title")
        label = str(title).strip() if isinstance(title, str) and title.strip() else eid
        url = meta.get("url") or meta.get("source_ref")
        normalized = None
        if kind == "web" and isinstance(url, str):
            normalized = normalize_citation_url(url)
        if kind == "kb":
            ref = str(meta.get("source_ref") or label)
            if ref.startswith("kb:"):
                label = ref[3:]
            items.append({"kind": "kb", "label": label, "url": None, "claim": eid})
            continue
        if normalized:
            items.append({"kind": "web", "label": label, "url": normalized, "claim": eid})
        else:
            items.append({"kind": "plain", "label": label, "url": None, "claim": eid})
    return items
