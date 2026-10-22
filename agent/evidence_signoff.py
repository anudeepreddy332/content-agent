"""Gate-1 human evidence signoff bound to the current draft and source passages."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from agent.html_policy import normalize_citation_url
from agent.semantic_analyzer.contract import build_evidence_manifest
from agent.semantic_analyzer.status_engine import sha256_utf8

CONTRACT_VERSION = "material-evidence-signoff-v1"
DECISIONS = frozenset({"SUPPORTED", "UNSUPPORTED", "CANNOT_DETERMINE"})


class EvidenceSignoffError(ValueError):
    """The review cannot certify the current material-claim evidence."""


def _digest(value: object) -> str:
    return sha256_utf8(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")))


def _context(text: str, start: int, end: int, margin: int = 180) -> dict:
    left, right = max(0, start - margin), min(len(text), end + margin)
    return {"start": left, "end": right, "text": text[left:right]}


def _bound_span(raw: dict, manifest: dict[str, dict]) -> dict:
    if not isinstance(raw, dict):
        raise EvidenceSignoffError("malformed evidence span")
    evidence_id, start, end = raw.get("evidence_id"), raw.get("start"), raw.get("end")
    entry = manifest.get(evidence_id)
    if (entry is None or type(start) is not int or type(end) is not int
            or start < 0 or end <= start or end > len(entry["source_text"])):
        raise EvidenceSignoffError("unbound evidence span")
    text = entry["source_text"]
    quote = text[start:end]
    if raw.get("text") != quote or not quote.strip():
        raise EvidenceSignoffError("stale evidence quotation")
    provenance = entry["provenance"]
    clickable_url = (normalize_citation_url(provenance.get("url"))
                     if provenance.get("kind") == "web" else None)
    return {
        "evidence_id": evidence_id, "start": start, "end": end, "quote": quote,
        "source_sha256": entry["source_sha256"],
        "source_context": _context(text, start, end),
        "provenance": provenance, "clickable_url": clickable_url,
        "citation_identity": evidence_id,
    }


def material_review_bundle(state: dict) -> dict:
    """Build only from current checkpoint state; never accept client source text."""
    from agent import workflow

    inventory = state.get("claim_inventory")
    if not workflow.inventory_current(state) or not isinstance(inventory.get("claims"), list):
        raise EvidenceSignoffError("stale claim inventory")
    draft = state.get("draft_markdown")
    if not isinstance(draft, str) or not draft.strip():
        raise EvidenceSignoffError("missing draft")
    report = state.get("grounding_report")
    if not isinstance(report, list):
        raise EvidenceSignoffError("missing verification report")
    report_by_id = {row.get("claim_id"): row for row in report if isinstance(row, dict)}
    if len(report_by_id) != len(report):
        raise EvidenceSignoffError("duplicate verification observation")
    claim_ids = [claim.get("claim_id") for claim in inventory["claims"]]
    if len(set(claim_ids)) != len(claim_ids) or set(claim_ids) != set(report_by_id):
        raise EvidenceSignoffError("inventory and verification roster differ")
    try:
        evidence = build_evidence_manifest(
            request_id=f"{state['run_id']}:{state['iterations']}",
            web_sources=state.get("web_sources"), kb_results=state.get("kb_results"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceSignoffError("source manifest unavailable") from exc
    manifest = {entry["evidence_id"]: entry for entry in evidence}
    claims = []
    for claim in inventory["claims"]:
        if claim.get("material") is not True:
            continue
        claim_id = claim["claim_id"]
        row = report_by_id[claim_id]
        if row.get("claim") != claim.get("claim_text"):
            raise EvidenceSignoffError("claim text changed since verification")
        occurrences = claim.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise EvidenceSignoffError("material claim has no draft anchor")
        draft_contexts = []
        for occurrence in occurrences:
            start, end = occurrence.get("start"), occurrence.get("end")
            if (type(start) is not int or type(end) is not int or start < 0
                    or end <= start or end > len(draft)
                    or draft[start:end] != claim.get("anchor_quote")):
                raise EvidenceSignoffError("stale material-claim anchor")
            draft_contexts.append({"start": start, "end": end,
                                   "quote": draft[start:end], "context": _context(draft, start, end)})
        raw_spans = row.get("support_spans")
        if not isinstance(raw_spans, list) or not raw_spans:
            raise EvidenceSignoffError("material claim has no bound support")
        support = [_bound_span(span, manifest) for span in raw_spans]
        blockers = []
        for blocker in row.get("blockers") or []:
            if not isinstance(blocker, dict):
                raise EvidenceSignoffError("malformed limitation")
            blockers.append({"kind": blocker.get("kind"),
                             "explanation": blocker.get("explanation"),
                             "evidence_spans": [_bound_span(span, manifest)
                                                for span in blocker.get("evidence_spans") or []]})
        claims.append({
            "claim_id": claim_id, "claim_text": claim["claim_text"],
            "draft_contexts": draft_contexts, "support_spans": support,
            "blockers": blockers, "limitations": list(row.get("reason_codes") or []),
            "materiality_rationale": claim.get("materiality_rationale"),
            "requires_citation": claim.get("requires_citation"),
        })
    core = {
        "contract_version": CONTRACT_VERSION,
        "run_id": state["run_id"],
        "draft_sha256": sha256_utf8(draft),
        "inventory_sha256": _digest(inventory),
        "material_claim_ids": [claim["claim_id"] for claim in claims],
        "claims": claims,
    }
    return {**core, "evidence_binding_sha256": _digest(core)}


def validate_submission(state: dict, submission: dict, *, reviewer_id: str,
                        origin: str, review_id: str) -> dict:
    """Mint the receipt on the server after checking every exact binding."""
    bundle = material_review_bundle(state)
    if not isinstance(submission, dict) or not review_id:
        raise EvidenceSignoffError("missing review decision or server review ID")
    if submission.get("evidence_binding_sha256") != bundle["evidence_binding_sha256"]:
        raise EvidenceSignoffError("stale evidence review")
    coverage = submission.get("coverage_attestation")
    if not isinstance(coverage, dict) or coverage != {
        "no_omitted_material_claims": True, "materiality_classification_correct": True,
    }:
        raise EvidenceSignoffError("full-draft coverage was not affirmed")
    rows = submission.get("decisions")
    if not isinstance(rows, list) or len(rows) != len(bundle["material_claim_ids"]):
        raise EvidenceSignoffError("missing or extra material-claim decision")
    seen = set()
    validated = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"claim_id", "decision", "rationale"}:
            raise EvidenceSignoffError("malformed claim decision")
        claim_id, decision, rationale = row["claim_id"], row["decision"], row["rationale"]
        if claim_id in seen or claim_id not in bundle["material_claim_ids"]:
            raise EvidenceSignoffError("duplicate or foreign claim decision")
        if decision not in DECISIONS or decision != "SUPPORTED":
            raise EvidenceSignoffError("material claim was not affirmatively supported")
        if not isinstance(rationale, str) or not 8 <= len(rationale.strip()) <= 500:
            raise EvidenceSignoffError("each claim requires a short evidence rationale")
        seen.add(claim_id)
        validated.append({"claim_id": claim_id, "decision": decision,
                          "rationale": rationale.strip()})
    if seen != set(bundle["material_claim_ids"]):
        raise EvidenceSignoffError("material roster was not covered")
    return {
        "contract_version": CONTRACT_VERSION, "run_id": state["run_id"],
        "review_id": review_id, "reviewer_id": reviewer_id, "origin": origin,
        "draft_sha256": bundle["draft_sha256"],
        "inventory_sha256": bundle["inventory_sha256"],
        "material_claim_ids": bundle["material_claim_ids"],
        "evidence_binding_sha256": bundle["evidence_binding_sha256"],
        "claim_evidence_bindings": [
            {"claim_id": claim["claim_id"], "spans": [
                {"evidence_id": span["evidence_id"], "start": span["start"],
                 "end": span["end"], "source_sha256": span["source_sha256"]}
                for span in claim["support_spans"]]}
            for claim in bundle["claims"]
        ],
        "decisions": validated, "coverage_attestation": coverage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def receipt_valid(state: dict, *, publishing: bool = False) -> bool:
    from agent import workflow

    from agent.reviewer_identity import approval_origin_valid

    if not workflow.gate1_eligible(state) or not approval_origin_valid(state, 1, publishing=publishing):
        return False
    receipt = state.get("gate1_evidence_signoff")
    if not isinstance(receipt, dict):
        return False
    try:
        bundle = material_review_bundle(state)
    except (EvidenceSignoffError, KeyError, TypeError):
        return False
    if publishing and receipt.get("origin") != "authenticated":
        return False
    if not isinstance(receipt.get("review_id"), str) or not receipt["review_id"]:
        return False
    for key in ("contract_version", "run_id", "draft_sha256", "inventory_sha256",
                "material_claim_ids", "evidence_binding_sha256"):
        if receipt.get(key) != bundle.get(key):
            return False
    if (receipt.get("reviewer_id") != state.get("gate1_reviewer_id")
            or receipt.get("origin") != state.get("gate1_review_origin")):
        return False
    if receipt.get("coverage_attestation") != {
        "no_omitted_material_claims": True, "materiality_classification_correct": True,
    }:
        return False
    expected_bindings = [
        {"claim_id": claim["claim_id"], "spans": [
            {"evidence_id": span["evidence_id"], "start": span["start"],
             "end": span["end"], "source_sha256": span["source_sha256"]}
            for span in claim["support_spans"]]}
        for claim in bundle["claims"]
    ]
    if receipt.get("claim_evidence_bindings") != expected_bindings:
        return False
    rows = receipt.get("decisions")
    if not isinstance(rows, list) or len(rows) != len(bundle["material_claim_ids"]):
        return False
    return (
        {row.get("claim_id") for row in rows if isinstance(row, dict)}
        == set(bundle["material_claim_ids"])
        and all(isinstance(row, dict) and row.get("decision") == "SUPPORTED"
                and isinstance(row.get("rationale"), str) and 8 <= len(row["rationale"]) <= 500
                for row in rows)
    )
