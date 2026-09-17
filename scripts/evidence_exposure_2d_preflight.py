"""Stage 2D evidence-exposure provider preflight harness. No provider calls by default."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import tiktoken
import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI
from openai import APIStatusError, AuthenticationError, BadRequestError, RateLimitError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_FIXTURES = REPO_ROOT / "evals" / "fixtures" / "evidence_exposure_2d.json"
DEFAULT_PRICE_SCHEDULE = REPO_ROOT / "evals" / "evidence_exposure_2d_price_schedule.json"
DEFAULT_2C_FIXTURES = REPO_ROOT / "evals" / "fixtures" / "evidence_exposure_2c.json"
VERIFY_SYSTEM_PATH = REPO_ROOT / "prompts" / "verify_system.md"

PACK_ID = "evidence_exposure_2d"
EVALUATOR_ID = "evidence_exposure_2d_preflight"
SCHEMA_VERSION = 1
STAGE = "2d"

HARD_SPEND_CEILING_USD = 0.08
MAX_OUTPUT_TOKENS = 4000
VERIFY_TEMPERATURE = 0.1
MAX_ATTEMPTS_PER_CELL = 1
PROVIDER_RETRIES_DISABLED = True
STAGE2D_PROVIDER_PATH = "stage2d_direct_openai"
HIGH_RISK_PREFIX_CELLS = frozenset({"P6-PREFIX", "P7-PREFIX"})

ASSET_IDS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
CELL_IDS = (
    "P1-PREFIX",
    "P1-COMPLETE",
    "P2-COMPLETE",
    "P3-COMPLETE",
    "P4-COMPLETE",
    "P5-COMPLETE",
    "P6-PREFIX",
    "P6-COMPLETE",
    "P7-PREFIX",
    "P7-COMPLETE",
)
PAIRED_ASSETS = ("P1", "P6", "P7")
EXPOSURE_ARMS = frozenset({"prefix", "complete"})


class EvidenceExposure2DError(ValueError):
    """Stage 2D pack or preflight asset is not evaluable."""


class SemanticMapping2DError(EvidenceExposure2DError):
    """Live provider output could not be mapped into Stage 2D semantic fixture inputs."""


class SemanticValidation2DError(EvidenceExposure2DError):
    """Constructed Stage 2D semantic fixture failed validation before metric computation."""


class ExecutionAuthorization2DError(EvidenceExposure2DError):
    """Stage 2D execution authorization missing or invalid at the transport boundary."""


@dataclass(frozen=True)
class Stage2DExecutionAuthorization:
    """Validated execution authorization produced only after full preflight checks."""

    authorization_token: str
    approved_execution_config_hash: str
    execution_git_sha: str
    budget_authorized: bool
    paired_isolation_ok: bool
    provider_execution_authorized: bool


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceExposure2DError(message)


def provider_execution_authorized() -> bool:
    return os.getenv("EVIDENCE_EXPOSURE_2D_EXECUTE", "").strip() == "1"


@dataclass
class RunExecutionState:
    """Cross-cell execution state for Stage 2D qualification runs."""

    frozen_returned_model_identity: str | None = None
    cell_artifacts: list[dict[str, Any]] = field(default_factory=list)
    run_aborted: bool = False
    abort_reason: str | None = None
    model_identity_invalid: bool = False

    def check_returned_model_identity(self, returned_model: str | None) -> str | None:
        if not returned_model:
            self.run_aborted = True
            self.model_identity_invalid = True
            return "missing returned model identity"
        if self.frozen_returned_model_identity is None:
            self.frozen_returned_model_identity = returned_model
            return None
        if returned_model != self.frozen_returned_model_identity:
            self.run_aborted = True
            self.model_identity_invalid = True
            return (
                f"model identity drift: {returned_model!r} != "
                f"{self.frozen_returned_model_identity!r}"
            )
        return None

    def should_stop_run(self) -> bool:
        return self.run_aborted or self.model_identity_invalid


def load_2c_case(case_id: str) -> dict[str, Any]:
    pack = json.loads(DEFAULT_2C_FIXTURES.read_text(encoding="utf-8"))
    for case in pack["cases"]:
        if case["id"] == case_id:
            return case
    raise EvidenceExposure2DError(f"2C case {case_id} not found")


def expose_source_context(*, source_kind: str, source_text: str, source_url: str, exposure_arm: str) -> str:
    from scripts.evaluate_evidence_exposure_2c import expose_arm_a, expose_arm_b

    if exposure_arm == "prefix":
        exposed, _, _ = expose_arm_a(source_kind, source_text)
    elif exposure_arm == "complete":
        exposed, _, _ = expose_arm_b(source_text)
    else:
        raise EvidenceExposure2DError(f"unknown exposure arm {exposure_arm!r}")
    label = "WEB" if source_kind == "web" else "KB"
    return f"[{label}] {source_url}\n{exposed}"


def load_verify_system() -> str:
    return VERIFY_SYSTEM_PATH.read_text(encoding="utf-8")


def build_verifier_user_message(draft_text: str, source_context: str) -> str:
    return f"""
Draft to verify:
{draft_text}

Available sources:
{source_context}

Return a JSON array. Each element:
{{"claim": "...", "source_url": "..." or null, "confidence": 0.0-1.0,
  "status": "verified" | "weak" | "unverified",
  "specificity": "substantive" | "generic"}}

Return ONLY the JSON array. No preamble.
""".strip()


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def shadow_runtime_acceptance(
    grounding_report: list[dict[str, Any]],
    *,
    verification_status: str = "completed",
    uvr_threshold: float = 0.15,
) -> dict[str, Any]:
    """Shadow current-runtime UVR gate without mutating production routing."""
    if verification_status != "completed":
        return {"accepted": False, "reason": "verification_status_not_completed", "uvr": None}
    if not grounding_report:
        return {"accepted": False, "reason": "empty_grounding_report", "uvr": None}
    unverified = sum(1 for row in grounding_report if row.get("status") == "unverified")
    uvr = unverified / len(grounding_report)
    accepted = uvr <= uvr_threshold
    return {
        "accepted": accepted,
        "reason": "uvr_within_threshold" if accepted else "uvr_above_threshold",
        "uvr": uvr,
        "uvr_threshold": uvr_threshold,
    }


def evaluate_semantic_oracle(
    semantic_fixture: dict[str, Any],
    *,
    template_fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from scripts.evaluate_claim_semantics_v2 import _compute_fixture_metrics

    fixture = canonicalize_semantic_fixture(semantic_fixture)
    template = canonicalize_semantic_fixture(template_fixture or semantic_fixture)
    try:
        validate_stage2d_semantic_fixture(fixture, template)
    except SemanticValidation2DError:
        raise
    except EvidenceExposure2DError as exc:
        raise SemanticValidation2DError(str(exc)) from exc
    return _compute_fixture_metrics(fixture)


def validate_stage2d_semantic_fixture(
    live_fixture: dict[str, Any],
    template_fixture: dict[str, Any],
) -> None:
    """Validate a Stage 2D semantic fixture before metric computation.

    Official ``evaluate_fixture()`` rejects Stage 2D fixture IDs (ADV02-ADV08).
    This applies the same structural validators from ``evaluate_claim_semantics_v2``
    without catalog-ID gating, then enforces the frozen per-asset template contract.
    """
    from scripts.evaluate_claim_semantics_v2 import (
        AUTOMATIC_ROUTE_REQUIRED_KEYS,
        EXCLUSION_REASONS,
        EXCLUSION_REQUIRED_KEYS,
        F02_FIXTURE_REQUIRED_KEYS,
        GOLD_REQUIRED_KEYS,
        ROUTE_DECISIONS,
        SEMANTIC_STATUSES,
        ClaimSemanticsV2Error,
        _is_f02_fixture,
        _require_keys,
        _validate_candidates_and_matches,
        _validate_evidence_bindings,
        _validate_final_atoms,
        _validate_fixed_classification_cases,
        _validate_sha256,
        _validate_span,
        _validate_verifier_rows,
    )

    try:
        _validate_stage2d_semantic_fixture_body(
            live_fixture,
            template_fixture,
            AUTOMATIC_ROUTE_REQUIRED_KEYS=AUTOMATIC_ROUTE_REQUIRED_KEYS,
            EXCLUSION_REASONS=EXCLUSION_REASONS,
            EXCLUSION_REQUIRED_KEYS=EXCLUSION_REQUIRED_KEYS,
            F02_FIXTURE_REQUIRED_KEYS=F02_FIXTURE_REQUIRED_KEYS,
            GOLD_REQUIRED_KEYS=GOLD_REQUIRED_KEYS,
            ROUTE_DECISIONS=ROUTE_DECISIONS,
            SEMANTIC_STATUSES=SEMANTIC_STATUSES,
            _is_f02_fixture=_is_f02_fixture,
            _require_keys=_require_keys,
            _validate_candidates_and_matches=_validate_candidates_and_matches,
            _validate_evidence_bindings=_validate_evidence_bindings,
            _validate_final_atoms=_validate_final_atoms,
            _validate_fixed_classification_cases=_validate_fixed_classification_cases,
            _validate_sha256=_validate_sha256,
            _validate_span=_validate_span,
            _validate_verifier_rows=_validate_verifier_rows,
        )
    except ClaimSemanticsV2Error as exc:
        raise SemanticValidation2DError(str(exc)) from exc


def _validate_stage2d_semantic_fixture_body(
    live_fixture: dict[str, Any],
    template_fixture: dict[str, Any],
    **validators: Any,
) -> None:
    AUTOMATIC_ROUTE_REQUIRED_KEYS = validators["AUTOMATIC_ROUTE_REQUIRED_KEYS"]
    EXCLUSION_REASONS = validators["EXCLUSION_REASONS"]
    EXCLUSION_REQUIRED_KEYS = validators["EXCLUSION_REQUIRED_KEYS"]
    F02_FIXTURE_REQUIRED_KEYS = validators["F02_FIXTURE_REQUIRED_KEYS"]
    GOLD_REQUIRED_KEYS = validators["GOLD_REQUIRED_KEYS"]
    ROUTE_DECISIONS = validators["ROUTE_DECISIONS"]
    SEMANTIC_STATUSES = validators["SEMANTIC_STATUSES"]
    _is_f02_fixture = validators["_is_f02_fixture"]
    _require_keys = validators["_require_keys"]
    _validate_candidates_and_matches = validators["_validate_candidates_and_matches"]
    _validate_evidence_bindings = validators["_validate_evidence_bindings"]
    _validate_final_atoms = validators["_validate_final_atoms"]
    _validate_fixed_classification_cases = validators["_validate_fixed_classification_cases"]
    _validate_sha256 = validators["_validate_sha256"]
    _validate_span = validators["_validate_span"]
    _validate_verifier_rows = validators["_validate_verifier_rows"]
    live = canonicalize_semantic_fixture(live_fixture)
    template = canonicalize_semantic_fixture(template_fixture)
    fixture_id = live["id"]
    _require(_is_f02_fixture(live), "Stage 2D semantic fixture must include final_atoms")
    _require_keys(live, F02_FIXTURE_REQUIRED_KEYS, "fixture")
    _require(live["id"] == template["id"], "fixture id drift from template")
    _require(live["draft_text"] == template["draft_text"], "draft text drift from template")
    _require(live["draft_sha256"] == template["draft_sha256"], "draft hash drift from template")

    template_material_gold = [gold for gold in template["gold_atoms"] if gold["material"]]
    _require(
        len(live["gold_atoms"]) > 0 or not template_material_gold,
        "empty gold_atoms invalid when template requires material gold",
    )
    if template_material_gold:
        _require(
            any(gold.get("material") for gold in live["gold_atoms"]),
            "required material gold population missing",
        )

    template_gold_ids = {gold["id"] for gold in template["gold_atoms"]}
    live_gold_ids = {gold["id"] for gold in live["gold_atoms"]}
    _require(live_gold_ids == template_gold_ids, "gold atom identity set drift from template")

    template_candidate_ids = {candidate["id"] for candidate in template["candidates"]}
    live_candidate_ids = {candidate["id"] for candidate in live["candidates"]}
    _require(
        live_candidate_ids == template_candidate_ids,
        "candidate identity set drift from template",
    )

    template_final_ids = {atom["id"] for atom in template["final_atoms"]}
    live_final_ids = {atom["id"] for atom in live["final_atoms"]}
    _require(live_final_ids == template_final_ids, "final atom identity set drift from template")

    draft = live["draft_text"]
    _validate_sha256(live["draft_sha256"], f"{fixture_id} draft_sha256")
    _require(live["draft_sha256"] == sha256_text(draft), f"{fixture_id} draft_sha256 is stale or invalid")

    gold_atoms = live["gold_atoms"]
    exclusions = live["exclusions"]
    _require(isinstance(gold_atoms, list), f"{fixture_id} gold_atoms must be a list")
    _require(isinstance(exclusions, list), f"{fixture_id} exclusions must be a list")

    gold_ids: set[str] = set()
    canonical_flags: dict[str, tuple[bool, bool, str]] = {}
    gold_spans: list[tuple[int, int, str]] = []
    for gold in gold_atoms:
        _require(isinstance(gold, dict), f"{fixture_id} gold atom must be an object")
        _require_keys(gold, GOLD_REQUIRED_KEYS, f"{fixture_id} gold atom")
        gold_id = gold["id"]
        _require(gold_id not in gold_ids, f"{fixture_id} duplicate gold id {gold_id}")
        gold_ids.add(gold_id)
        _require(gold["gold_semantic_status"] in SEMANTIC_STATUSES, f"{gold_id} invalid gold semantic status")
        _require(type(gold["factual"]) is bool, f"{gold_id} factual must be bool")
        _require(type(gold["material"]) is bool, f"{gold_id} material must be bool")
        _require(not gold["material"] or gold["factual"], f"{gold_id} material atom must be factual")
        span = gold["span"]
        if span is None:
            _require(
                gold["text"] not in draft,
                f"{gold_id} omitted required claim must be absent from the final draft",
            )
        elif gold["text"] not in draft:
            # Frozen omission assets may keep advisory spans for absent required claims.
            pass
        else:
            _validate_span(span, gold["text"], draft, gold_id)
            gold_spans.append((span[0], span[1], gold_id))
        flags = (gold["factual"], gold["material"], gold["gold_semantic_status"])
        previous = canonical_flags.get(gold["canonical_id"])
        if previous is None:
            canonical_flags[gold["canonical_id"]] = flags
        else:
            _require(previous == flags, f"{gold_id} canonical flags disagree with sibling rows")

    exclusion_ids: set[str] = set()
    for exclusion in exclusions:
        _require(isinstance(exclusion, dict), f"{fixture_id} exclusion must be an object")
        _require_keys(exclusion, EXCLUSION_REQUIRED_KEYS, f"{fixture_id} exclusion")
        exclusion_id = exclusion["id"]
        _require(exclusion_id not in exclusion_ids, f"{fixture_id} duplicate exclusion id {exclusion_id}")
        _require(exclusion_id not in gold_ids, f"{exclusion_id} cannot also be a gold id")
        exclusion_ids.add(exclusion_id)
        _require(exclusion["reason"] in EXCLUSION_REASONS, f"{exclusion_id} unknown exclusion reason")
        _validate_span(exclusion["span"], exclusion["text"], draft, exclusion_id)
        start, end = exclusion["span"]
        for gold_start, gold_end, overlap_gold_id in gold_spans:
            _require(
                end <= gold_start or start >= gold_end,
                f"{exclusion_id} overlaps gold {overlap_gold_id}",
            )

    candidates = live["candidates"]
    matches = live["allowed_matches"]
    _require(isinstance(candidates, list), f"{fixture_id} candidates must be a list")
    _require(isinstance(matches, list), f"{fixture_id} allowed_matches must be a list")
    candidate_by_id = _validate_candidates_and_matches(fixture_id, gold_ids, draft, candidates, matches)

    bindings = live["evidence_bindings"]
    _require(isinstance(bindings, list), f"{fixture_id} evidence_bindings must be a list")
    _validate_evidence_bindings(fixture_id, gold_ids, set(candidate_by_id), bindings)

    verifier_rows = live["verifier_rows"]
    _require(isinstance(verifier_rows, list), f"{fixture_id} verifier_rows must be a list")
    _validate_verifier_rows(fixture_id, verifier_rows)

    automatic_route = live["automatic_route"]
    _require(isinstance(automatic_route, dict), f"{fixture_id} automatic_route must be an object")
    _require_keys(automatic_route, AUTOMATIC_ROUTE_REQUIRED_KEYS, f"{fixture_id} automatic_route")
    _require(
        automatic_route["decision"] in ROUTE_DECISIONS,
        f"{fixture_id} invalid automatic route decision",
    )

    _validate_final_atoms(fixture_id, draft, gold_ids, live["final_atoms"])
    _validate_fixed_classification_cases(fixture_id, gold_ids, live["fixed_classification_cases"])
    _validate_stage2d_template_semantic_contract(live, template)


def _validate_stage2d_template_semantic_contract(
    live: dict[str, Any],
    template: dict[str, Any],
) -> None:
    """Ensure immutable Stage 2D semantic identities remain bound to the frozen template."""
    template_final_by_id = {atom["id"]: atom for atom in template["final_atoms"]}
    for live_atom in live["final_atoms"]:
        template_atom = template_final_by_id[live_atom["id"]]
        for contract_field in (
            "prediction_id",
            "required_gold_id",
            "reference_relationship",
            "independent_semantic_label",
            "material",
            "text",
        ):
            _require(
                live_atom.get(contract_field) == template_atom.get(contract_field),
                f"{live_atom['id']} {contract_field} drift from frozen template",
            )

    template_cases = {case["id"]: case for case in template.get("fixed_classification_cases", [])}
    for live_case in live.get("fixed_classification_cases", []):
        template_case = template_cases[live_case["id"]]
        for contract_field in (
            "prediction_id",
            "required_gold_id",
            "final_atom_id",
            "source",
            "material",
            "independent_semantic_label",
        ):
            _require(
                live_case.get(contract_field) == template_case.get(contract_field),
                f"{live_case['id']} {contract_field} drift from frozen template",
            )


def validate_frozen_prediction_references(fixture: dict[str, Any]) -> None:
    """Reject frozen templates with unresolved prediction references."""
    canonical = canonicalize_semantic_fixture(fixture)
    candidate_ids = {candidate["id"] for candidate in canonical["candidates"]}
    for final_atom in canonical["final_atoms"]:
        prediction_id = final_atom.get("prediction_id")
        if prediction_id is not None:
            _require(
                prediction_id in candidate_ids,
                f"invalid final_atom prediction_id {prediction_id!r}",
            )
    for case in canonical.get("fixed_classification_cases", []):
        prediction_id = case.get("prediction_id")
        if prediction_id is not None:
            _require(
                prediction_id in candidate_ids,
                f"invalid classification prediction_id {prediction_id!r}",
            )


def validate_stage2d_semantic_fixtures(pack: dict[str, Any]) -> None:
    """Validate semantic fixture contents for every frozen asset."""
    for asset in pack["assets"]:
        _require("semantic_fixture" in asset, f"missing semantic mapping contract for {asset['asset_id']}")
        fixture = asset["semantic_fixture"]
        validate_frozen_prediction_references(fixture)
        evaluate_semantic_oracle(fixture)


def validate_frozen_model_config(pack: dict[str, Any]) -> None:
    model_config = pack["model_config"]
    _require(model_config["max_tokens"] == MAX_OUTPUT_TOKENS, "model_config max_tokens must be 4000")
    _require(model_config["temperature"] == VERIFY_TEMPERATURE, "model_config temperature mismatch")
    _require(model_config["model_alias"] == "deepseek-chat", "model_config model_alias mismatch")


def build_approved_execution_config_hash(pack: dict[str, Any]) -> str:
    validate_frozen_model_config(pack)
    requests = build_all_requests(pack)
    for request in requests:
        _require(
            request["model_config"]["max_tokens"] == MAX_OUTPUT_TOKENS,
            "cell request max_tokens must match approved budget bound",
        )
    payload = {
        "model_config": pack["model_config"],
        "cell_configs": [
            {
                "cell_id": request["cell_id"],
                "model_config": request["model_config"],
                "exposed_context_sha256": request["exposed_context_sha256"],
                "draft_sha256": request["draft_sha256"],
                "source_sha256": request["source_sha256"],
                "verify_system_sha256": request["verify_system_sha256"],
                "max_attempts": request["max_attempts"],
            }
            for request in requests
        ],
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=True))


def issue_stage2d_execution_authorization(
    pack: dict[str, Any],
    *,
    schedule_path: Path | str = DEFAULT_PRICE_SCHEDULE,
) -> Stage2DExecutionAuthorization:
    """Issue validated execution authorization after all preflight gates pass."""
    if not provider_execution_authorized():
        raise ExecutionAuthorization2DError(
            "provider execution disabled; set EVIDENCE_EXPOSURE_2D_EXECUTE=1 after independent preflight review"
        )
    validate_pack(pack)
    validate_frozen_model_config(pack)
    validate_stage2d_semantic_fixtures(pack)
    budget = preflight_budget(pack, schedule_path=schedule_path)
    if not budget["budget_authorized"]:
        raise ExecutionAuthorization2DError("budget not authorized")
    paired = verify_paired_isolation(pack)
    paired_ok = all(item["isolated"] for item in paired.values())
    if not paired_ok:
        raise ExecutionAuthorization2DError("paired isolation failed")
    git_sha = get_execution_git_sha()
    approved_hash = build_approved_execution_config_hash(pack)
    token_payload = {
        "approved_execution_config_hash": approved_hash,
        "execution_git_sha": git_sha,
        "budget_authorized": budget["budget_authorized"],
        "paired_isolation_ok": paired_ok,
        "provider_execution_authorized": True,
        "cell_ids": list(CELL_IDS),
    }
    return Stage2DExecutionAuthorization(
        authorization_token=sha256_text(json.dumps(token_payload, sort_keys=True, ensure_ascii=True)),
        approved_execution_config_hash=approved_hash,
        execution_git_sha=git_sha,
        budget_authorized=budget["budget_authorized"],
        paired_isolation_ok=paired_ok,
        provider_execution_authorized=True,
    )


def _assert_transport_authorized(
    pack: dict[str, Any],
    request: dict[str, Any],
    execution_auth: Stage2DExecutionAuthorization | None,
) -> None:
    if execution_auth is None:
        raise ExecutionAuthorization2DError("missing Stage 2D execution authorization")
    if not execution_auth.provider_execution_authorized:
        raise ExecutionAuthorization2DError("provider execution not authorized")
    if not execution_auth.budget_authorized:
        raise ExecutionAuthorization2DError("budget not authorized")
    if not execution_auth.paired_isolation_ok:
        raise ExecutionAuthorization2DError("paired isolation not authorized")
    approved_hash = request.get("approved_execution_config_hash")
    if approved_hash != execution_auth.approved_execution_config_hash:
        raise ExecutionAuthorization2DError("request execution config hash mismatch")
    validate_cell_request_artifact(pack, request)
    _require(
        request["model_config"]["max_tokens"] == MAX_OUTPUT_TOKENS,
        "executed max_tokens differs from approved budget configuration",
    )
    rebuilt_hash = build_approved_execution_config_hash(pack)
    if rebuilt_hash != execution_auth.approved_execution_config_hash:
        raise ExecutionAuthorization2DError("execution config drift from approved preflight hash")


def canonicalize_semantic_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Ensure v2 schema-required fields for ephemeral oracle evaluation."""
    fid = fixture["id"]
    out = json.loads(json.dumps(fixture))
    for item in out.get("gold_atoms", []):
        item.setdefault("canonical_id", f"{fid}.ATOM.{item['id']}")
    for item in out.get("candidates", []):
        item.setdefault("canonical_id", f"{fid}.C.{item['id']}")
    return out


def _source_record(
    *,
    source_id: str,
    source_kind: str,
    source_url: str,
    source_text: str,
    source_rank: int = 1,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_kind": source_kind,
        "source_url": source_url,
        "source_rank": source_rank,
        "source_text": source_text,
        "source_sha256": sha256_text(source_text),
        "source_character_length": len(source_text),
    }


def build_frozen_pack() -> dict[str, Any]:
    w02 = load_2c_case("E2C-W02")
    w03 = load_2c_case("E2C-W03")
    k03 = load_2c_case("E2C-K03")

    p2_source = (
        "FluxCap array service note FC-881. "
        "FluxCap array stores at most 4096 records per shard under nominal load."
    )
    p3_source = (
        "Project NEXUS-7 internal field manual. "
        "NEXUS-7 bounded replay retains exactly 512 events per shard."
    )
    p4_source = (
        "Inventory ledger IL-44. There are 10 items in partition A. "
        "There are 15 items in partition B."
    )
    p5_source = "Warehouse count sheet WC-12. There are 10 items in the active bin."

    assets: list[dict[str, Any]] = []

    p1_draft = "Quorum latch engages only after seven replica acknowledgements."
    p1_source = _source_record(
        source_id="SRC-P1-W02",
        source_kind=w02["source_kind"],
        source_url=w02["source_url"],
        source_text=w02["source_text"],
    )
    assets.append(
        {
            "asset_id": "P1",
            "title": "late supported material claim",
            "source": p1_source,
            "draft_text": p1_draft,
            "draft_sha256": sha256_text(p1_draft),
            "cells": [
                {"cell_id": "P1-PREFIX", "exposure_arm": "prefix"},
                {"cell_id": "P1-COMPLETE", "exposure_arm": "complete"},
            ],
            "expected_corrected_semantic_pass": {"P1-PREFIX": False, "P1-COMPLETE": True},
            "expected_primary_label_complete": "verified",
            "semantic_fixture": {
                "id": "ADV02",
                "title": "P1 late supported claim",
                "draft_text": p1_draft,
                "draft_sha256": sha256_text(p1_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p1_draft,
                        "span": [0, len(p1_draft)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "verified",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p1_draft,
                        "span": [0, len(p1_draft)],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    }
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P1.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P1.EV.quorum",
                        "valid": True,
                        "fully_entailed": True,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p1_draft,
                        "span": [0, len(p1_draft)],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "verified",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "verified",
                        "binding": {
                            "id": "P1.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P1.EV.quorum",
                            "valid": True,
                            "fully_entailed": True,
                        },
                    }
                ],
                "verifier_rows": [{"id": "P1.VR.c1", "status": "verified"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P1.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "verified",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    }
                ],
            },
        }
    )

    p2_draft = "FluxCap array stores 8192 records per shard under nominal load."
    p2_src = _source_record(
        source_id="SRC-P2",
        source_kind="kb",
        source_url="kb://fluxcap/array-fc881",
        source_text=p2_source,
    )
    assets.append(
        {
            "asset_id": "P2",
            "title": "unsupported material claim",
            "source": p2_src,
            "draft_text": p2_draft,
            "draft_sha256": sha256_text(p2_draft),
            "cells": [{"cell_id": "P2-COMPLETE", "exposure_arm": "complete"}],
            "expected_corrected_semantic_pass": {"P2-COMPLETE": False},
            "expected_primary_label_complete": "unverified",
            "semantic_fixture": {
                "id": "ADV03",
                "title": "P2 unsupported claim",
                "draft_text": p2_draft,
                "draft_sha256": sha256_text(p2_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p2_draft,
                        "span": [0, len(p2_draft)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "unverified",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p2_draft,
                        "span": [0, len(p2_draft)],
                        "roles": [],
                        "predicted_semantic_status": "unverified",
                    }
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P2.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P2.EV.fluxcap",
                        "valid": False,
                        "fully_entailed": False,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p2_draft,
                        "span": [0, len(p2_draft)],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "unverified",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "unverified",
                        "binding": {
                            "id": "P2.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P2.EV.fluxcap",
                            "valid": False,
                            "fully_entailed": False,
                        },
                    }
                ],
                "verifier_rows": [{"id": "P2.VR.c1", "status": "unverified"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P2.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "unverified",
                        "predicted_semantic_status": "unverified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    }
                ],
            },
        }
    )

    p3_draft = "NEXUS-7 bounded replay retains thousands of events per shard."
    p3_src = _source_record(
        source_id="SRC-P3",
        source_kind="web",
        source_url="https://example.test/nexus-7/manual",
        source_text=p3_source,
    )
    assets.append(
        {
            "asset_id": "P3",
            "title": "partial weak material claim",
            "source": p3_src,
            "draft_text": p3_draft,
            "draft_sha256": sha256_text(p3_draft),
            "cells": [{"cell_id": "P3-COMPLETE", "exposure_arm": "complete"}],
            "expected_corrected_semantic_pass": {"P3-COMPLETE": False},
            "expected_primary_label_complete": "weak",
            "semantic_fixture": {
                "id": "ADV04",
                "title": "P3 partial weak claim",
                "draft_text": p3_draft,
                "draft_sha256": sha256_text(p3_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p3_draft,
                        "span": [0, len(p3_draft)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "weak",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p3_draft,
                        "span": [0, len(p3_draft)],
                        "roles": [],
                        "predicted_semantic_status": "weak",
                    }
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P3.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P3.EV.nexus",
                        "valid": True,
                        "fully_entailed": False,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p3_draft,
                        "span": [0, len(p3_draft)],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "weak",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "weak",
                        "binding": {
                            "id": "P3.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P3.EV.nexus",
                            "valid": True,
                            "fully_entailed": False,
                        },
                    }
                ],
                "verifier_rows": [{"id": "P3.VR.c1", "status": "weak"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P3.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "weak",
                        "predicted_semantic_status": "weak",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    }
                ],
            },
        }
    )

    p4_draft = "There are 10 items in partition A. There are 15 items in partition B."
    p4_src = _source_record(
        source_id="SRC-P4",
        source_kind="kb",
        source_url="kb://inventory/ledger-il44",
        source_text=p4_source,
    )
    assets.append(
        {
            "asset_id": "P4",
            "title": "supported-new material claim",
            "source": p4_src,
            "draft_text": p4_draft,
            "draft_sha256": sha256_text(p4_draft),
            "cells": [{"cell_id": "P4-COMPLETE", "exposure_arm": "complete"}],
            "expected_corrected_semantic_pass": {"P4-COMPLETE": True},
            "expected_primary_label_complete": "verified",
            "semantic_fixture": {
                "id": "ADV05",
                "title": "P4 supported-new claim",
                "draft_text": p4_draft,
                "draft_sha256": sha256_text(p4_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": "There are 10 items in partition A.",
                        "span": [0, 34],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "verified",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": "There are 10 items in partition A.",
                        "span": [0, 34],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    },
                    {
                        "id": "c2",
                        "text": "There are 15 items in partition B.",
                        "span": [35, 69],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    },
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P4.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P4.EV.ledger",
                        "valid": True,
                        "fully_entailed": True,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": "There are 10 items in partition A.",
                        "span": [0, 34],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "verified",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "verified",
                        "binding": {
                            "id": "P4.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P4.EV.ledger",
                            "valid": True,
                            "fully_entailed": True,
                        },
                    },
                    {
                        "id": "f2",
                        "text": "There are 15 items in partition B.",
                        "span": [35, 69],
                        "material": True,
                        "reference_relationship": "unmatched",
                        "required_gold_id": None,
                        "independent_semantic_label": "verified",
                        "prediction_id": "c2",
                        "predicted_semantic_status": "verified",
                        "binding": {
                            "id": "P4.FBIND.f2",
                            "target_final_claim_id": "f2",
                            "evidence_id": "P4.EV.ledger",
                            "valid": True,
                            "fully_entailed": True,
                        },
                    },
                ],
                "verifier_rows": [
                    {"id": "P4.VR.c1", "status": "verified"},
                    {"id": "P4.VR.c2", "status": "verified"},
                ],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P4.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "verified",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    },
                    {
                        "id": "P4.CLASS.f2",
                        "source": "unmatched-final",
                        "material": True,
                        "independent_semantic_label": "verified",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c2",
                        "required_gold_id": None,
                        "final_atom_id": "f2",
                    },
                ],
            },
        }
    )

    p5_draft = "The active bin is labeled WC-12."
    p5_gold = "There are 10 items in the active bin."
    p5_src = _source_record(
        source_id="SRC-P5",
        source_kind="web",
        source_url="https://example.test/warehouse/wc12",
        source_text=p5_source,
    )
    assets.append(
        {
            "asset_id": "P5",
            "title": "required material omission",
            "source": p5_src,
            "draft_text": p5_draft,
            "draft_sha256": sha256_text(p5_draft),
            "cells": [{"cell_id": "P5-COMPLETE", "exposure_arm": "complete"}],
            "expected_corrected_semantic_pass": {"P5-COMPLETE": False},
            "expected_primary_label_complete": "verified",
            "semantic_fixture": {
                "id": "ADV06",
                "title": "P5 omission",
                "draft_text": p5_draft,
                "draft_sha256": sha256_text(p5_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p5_gold,
                        "span": [0, len(p5_gold)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "verified",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p5_draft,
                        "span": [0, len(p5_draft)],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    }
                ],
                "allowed_matches": [],
                "evidence_bindings": [],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p5_draft,
                        "span": [0, len(p5_draft)],
                        "material": False,
                        "reference_relationship": "unmatched",
                        "required_gold_id": None,
                        "independent_semantic_label": "verified",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "verified",
                        "binding": None,
                    }
                ],
                "verifier_rows": [{"id": "P5.VR.c1", "status": "verified"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P5.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "verified",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": None,
                    }
                ],
            },
        }
    )

    p6_draft = "Helix gate permits export when mode flag HG-ENABLE is set."
    p6_source = _source_record(
        source_id="SRC-P6-W03",
        source_kind=w03["source_kind"],
        source_url=w03["source_url"],
        source_text=w03["source_text"],
    )
    assets.append(
        {
            "asset_id": "P6",
            "title": "late contradiction",
            "source": p6_source,
            "draft_text": p6_draft,
            "draft_sha256": sha256_text(p6_draft),
            "cells": [
                {"cell_id": "P6-PREFIX", "exposure_arm": "prefix"},
                {"cell_id": "P6-COMPLETE", "exposure_arm": "complete"},
            ],
            "expected_corrected_semantic_pass": {"P6-PREFIX": False, "P6-COMPLETE": False},
            "expected_primary_label_complete": "weak",
            "semantic_fixture": {
                "id": "ADV07",
                "title": "P6 contradiction",
                "draft_text": p6_draft,
                "draft_sha256": sha256_text(p6_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p6_draft,
                        "span": [0, len(p6_draft)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "weak",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p6_draft,
                        "span": [0, len(p6_draft)],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    }
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P6.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P6.EV.helix",
                        "valid": True,
                        "fully_entailed": False,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p6_draft,
                        "span": [0, len(p6_draft)],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "weak",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "verified",
                        "binding": {
                            "id": "P6.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P6.EV.helix",
                            "valid": True,
                            "fully_entailed": False,
                        },
                    }
                ],
                "verifier_rows": [{"id": "P6.VR.c1", "status": "verified"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P6.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "weak",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    }
                ],
            },
        }
    )

    p7_draft = "Cache tier T3 admits writes when isolation level IL-2 holds."
    p7_source = _source_record(
        source_id="SRC-P7-K03",
        source_kind=k03["source_kind"],
        source_url=k03["source_url"],
        source_text=k03["source_text"],
    )
    assets.append(
        {
            "asset_id": "P7",
            "title": "late qualifier",
            "source": p7_source,
            "draft_text": p7_draft,
            "draft_sha256": sha256_text(p7_draft),
            "cells": [
                {"cell_id": "P7-PREFIX", "exposure_arm": "prefix"},
                {"cell_id": "P7-COMPLETE", "exposure_arm": "complete"},
            ],
            "expected_corrected_semantic_pass": {"P7-PREFIX": False, "P7-COMPLETE": False},
            "expected_primary_label_complete": "weak",
            "semantic_fixture": {
                "id": "ADV08",
                "title": "P7 qualifier loss",
                "draft_text": p7_draft,
                "draft_sha256": sha256_text(p7_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p7_draft,
                        "span": [0, len(p7_draft)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "weak",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p7_draft,
                        "span": [0, len(p7_draft)],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    }
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P7.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P7.EV.cache",
                        "valid": True,
                        "fully_entailed": False,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p7_draft,
                        "span": [0, len(p7_draft)],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "weak",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "verified",
                        "binding": {
                            "id": "P7.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P7.EV.cache",
                            "valid": True,
                            "fully_entailed": False,
                        },
                    }
                ],
                "verifier_rows": [{"id": "P7.VR.c1", "status": "verified"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P7.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "weak",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    }
                ],
            },
        }
    )

    verify_system = load_verify_system()
    pack = {
        "pack_id": PACK_ID,
        "schema_version": SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "stage": STAGE,
        "description": "Frozen Stage 2D provider preflight pack",
        "provider_execution_default": "disabled",
        "hard_spend_ceiling_usd": HARD_SPEND_CEILING_USD,
        "max_attempts_per_cell": MAX_ATTEMPTS_PER_CELL,
        "provider_retries_disabled": PROVIDER_RETRIES_DISABLED,
        "model_config": {
            "model_alias": "deepseek-chat",
            "temperature": VERIFY_TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
        },
        "prompt_identity": {
            "verify_system_path": "prompts/verify_system.md",
            "verify_system_sha256": sha256_text(verify_system),
        },
        "zero_tolerance_gates": {
            "material_false_verification_rate.v2_numerator": 0,
            "automatic_semantic_false_pass_rate.v2_numerator": 0,
        },
        "assets": assets,
    }
    validate_pack(pack)
    for asset in pack["assets"]:
        asset["semantic_fixture"] = canonicalize_semantic_fixture(asset["semantic_fixture"])
    return pack


def build_stage2d_client_config() -> dict[str, Any]:
    """Retry-free Stage 2D provider client configuration."""
    from config import DEEPSEEK_BASE_URL, LLM_TIMEOUT_S

    return {
        "provider_path": STAGE2D_PROVIDER_PATH,
        "base_url": DEEPSEEK_BASE_URL,
        "timeout_s": LLM_TIMEOUT_S,
        "max_retries": 0,
        "follow_redirects": False,
        "uses_production_llm_call": False,
        "uses_tenacity_retry_wrapper": False,
    }


def build_stage2d_http_client(*, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    from config import LLM_TIMEOUT_S

    return httpx.Client(
        transport=transport,
        timeout=LLM_TIMEOUT_S,
        follow_redirects=False,
    )


def build_stage2d_client(
    *,
    api_key: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> OpenAI:
    """Construct a dedicated retry-free OpenAI-compatible client for Stage 2D."""
    from config import DEEPSEEK_BASE_URL, LLM_TIMEOUT_S

    return OpenAI(
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
        timeout=LLM_TIMEOUT_S,
        max_retries=0,
        http_client=build_stage2d_http_client(transport=transport),
    )


def parse_verifier_output(raw: str) -> list[dict[str, Any]]:
    """Parse verifier JSON using production parsing rules without production retry path."""
    from agent.nodes import _parse_verifier_verdicts

    return _parse_verifier_verdicts(raw)


def _match_provider_row(candidate_text: str, grounding_report: list[dict[str, Any]]) -> dict[str, Any]:
    target = candidate_text.strip()
    matches = [row for row in grounding_report if (row.get("claim") or "").strip() == target]
    if len(matches) == 1:
        return matches[0]
    raise SemanticMapping2DError(f"cannot map candidate claim {target!r} to provider output")


def _apply_live_prediction_statuses(
    fixture: dict[str, Any],
    candidate_by_id: dict[str, dict[str, Any]],
) -> None:
    for final_atom in fixture["final_atoms"]:
        final_atom.pop("predicted_semantic_status", None)
        prediction_id = final_atom.get("prediction_id")
        if prediction_id is None:
            continue
        if prediction_id not in candidate_by_id:
            raise SemanticMapping2DError(
                f"unresolved final_atom prediction_id {prediction_id!r}"
            )
        final_atom["predicted_semantic_status"] = candidate_by_id[prediction_id][
            "predicted_semantic_status"
        ]

    for case in fixture.get("fixed_classification_cases", []):
        case.pop("predicted_semantic_status", None)
        prediction_id = case.get("prediction_id")
        if prediction_id is None:
            continue
        if prediction_id not in candidate_by_id:
            raise SemanticMapping2DError(
                f"unresolved classification prediction_id {prediction_id!r}"
            )
        case["predicted_semantic_status"] = candidate_by_id[prediction_id][
            "predicted_semantic_status"
        ]


def map_provider_to_semantic_fixture(
    asset: dict[str, Any],
    grounding_report: list[dict[str, Any]],
) -> dict[str, Any]:
    """Map live verifier output into validated Semantic P0 fixture inputs."""
    fixture = canonicalize_semantic_fixture(json.loads(json.dumps(asset["semantic_fixture"])))
    if not grounding_report:
        raise SemanticMapping2DError("empty grounding report cannot be mapped")

    candidate_by_id = {candidate["id"]: candidate for candidate in fixture["candidates"]}
    for candidate in fixture["candidates"]:
        row = _match_provider_row(candidate["text"], grounding_report)
        status = row.get("status")
        if status not in {"verified", "weak", "unverified"}:
            raise SemanticMapping2DError(f"invalid provider status {status!r}")
        candidate["predicted_semantic_status"] = status

    for index, verifier_row in enumerate(fixture["verifier_rows"]):
        verifier_row["status"] = fixture["candidates"][index]["predicted_semantic_status"]

    _apply_live_prediction_statuses(fixture, candidate_by_id)

    shadow = shadow_runtime_acceptance(grounding_report)
    fixture["automatic_route"] = {"decision": "PASS" if shadow["accepted"] else "FAIL"}
    return fixture


def validate_price_schedule(schedule: dict[str, Any]) -> None:
    required = (
        "schedule_id",
        "model_alias",
        "input_cost_per_million_tokens_usd",
        "output_cost_per_million_tokens_usd",
        "max_output_tokens_per_cell",
    )
    for key in required:
        _require(key in schedule, f"price schedule missing {key}")
    _require(schedule["max_output_tokens_per_cell"] == MAX_OUTPUT_TOKENS, "output token bound mismatch")


def validate_cell_request_artifact(pack: dict[str, Any], request: dict[str, Any]) -> None:
    rebuilt = build_cell_request(pack, request["cell_id"])
    _require(
        rebuilt["exposed_context_sha256"] == request["exposed_context_sha256"],
        "stale or mismatched exposed-context hash",
    )
    _require(rebuilt["draft_sha256"] == request["draft_sha256"], "stale or mismatched draft hash")
    _require(rebuilt["source_sha256"] == request["source_sha256"], "stale or mismatched source hash")
    _require(rebuilt["model_config"] == request["model_config"], "stale or mismatched model config")
    _require(
        request["model_config"]["max_tokens"] == MAX_OUTPUT_TOKENS,
        "executed max_tokens differs from approved budget configuration",
    )


def _user_messages_differ_only_by_context(prefix_req: dict[str, Any], complete_req: dict[str, Any]) -> bool:
    prefix_user = prefix_req["messages"][1]["content"]
    complete_user = complete_req["messages"][1]["content"]
    prefix_context = prefix_req["exposed_verifier_context"]
    complete_context = complete_req["exposed_verifier_context"]
    draft = prefix_req["draft_text"]
    _require(prefix_req["draft_text"] == complete_req["draft_text"], "paired draft mismatch")
    expected_prefix = build_verifier_user_message(draft, prefix_context)
    expected_complete = build_verifier_user_message(draft, complete_context)
    return prefix_user == expected_prefix and complete_user == expected_complete


def validate_pack(pack: dict[str, Any]) -> None:
    _require(pack["pack_id"] == PACK_ID, "pack_id mismatch")
    _require(pack["evaluator_id"] == EVALUATOR_ID, "evaluator_id mismatch")
    _require(pack["schema_version"] == SCHEMA_VERSION, "schema_version mismatch")
    _require(pack.get("stage") == STAGE, "stage must be 2d")
    validate_frozen_model_config(pack)
    assets = pack["assets"]
    _require(len(assets) == len(ASSET_IDS), "asset count mismatch")
    asset_ids: set[str] = set()
    cell_ids: set[str] = set()
    requirement_ids: set[str] = set()
    for asset in assets:
        _require(asset["asset_id"] in ASSET_IDS, "unknown asset id")
        _require(asset["asset_id"] not in asset_ids, f"duplicate asset id {asset['asset_id']}")
        asset_ids.add(asset["asset_id"])
        _require(asset["draft_sha256"] == sha256_text(asset["draft_text"]), "draft hash mismatch")
        source = asset["source"]
        _require(source["source_sha256"] == sha256_text(source["source_text"]), "source hash mismatch")
        _require(source["source_rank"] == 1, "source rank must be 1")
        for cell in asset["cells"]:
            cid = cell["cell_id"]
            _require(cid not in cell_ids, f"duplicate cell id {cid}")
            cell_ids.add(cid)
            _require(cell["exposure_arm"] in EXPOSURE_ARMS, "invalid exposure arm")
        sem = asset["semantic_fixture"]
        for req_key in ("gold_atoms", "final_atoms", "fixed_classification_cases"):
            for item in sem.get(req_key, []):
                rid = item.get("id")
                if rid:
                    full = f"{asset['asset_id']}:{rid}"
                    _require(full not in requirement_ids, f"duplicate semantic identity {full}")
                    requirement_ids.add(full)
    _require(asset_ids == set(ASSET_IDS), "asset catalog mismatch")
    _require(cell_ids == set(CELL_IDS), "cell catalog mismatch")


def load_pack(path: Path | str = DEFAULT_FIXTURES) -> dict[str, Any]:
    pack = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_pack(pack)
    return pack


def find_asset(pack: dict[str, Any], asset_id: str) -> dict[str, Any]:
    for asset in pack["assets"]:
        if asset["asset_id"] == asset_id:
            return asset
    raise EvidenceExposure2DError(f"asset {asset_id} not found")


def find_cell(asset: dict[str, Any], cell_id: str) -> dict[str, Any]:
    for cell in asset["cells"]:
        if cell["cell_id"] == cell_id:
            return cell
    raise EvidenceExposure2DError(f"cell {cell_id} not found in asset {asset['asset_id']}")


def build_cell_request(pack: dict[str, Any], cell_id: str) -> dict[str, Any]:
    for asset in pack["assets"]:
        for cell in asset["cells"]:
            if cell["cell_id"] == cell_id:
                source = asset["source"]
                verify_system = load_verify_system()
                exposed_context = expose_source_context(
                    source_kind=source["source_kind"],
                    source_text=source["source_text"],
                    source_url=source["source_url"],
                    exposure_arm=cell["exposure_arm"],
                )
                user_message = build_verifier_user_message(asset["draft_text"], exposed_context)
                messages = [
                    {"role": "system", "content": verify_system},
                    {"role": "user", "content": user_message},
                ]
                return {
                    "cell_id": cell_id,
                    "asset_id": asset["asset_id"],
                    "call_order": CELL_IDS.index(cell_id) + 1,
                    "source_id": source["source_id"],
                    "source_sha256": source["source_sha256"],
                    "complete_source_text": source["source_text"],
                    "draft_text": asset["draft_text"],
                    "draft_sha256": asset["draft_sha256"],
                    "exposure_arm": cell["exposure_arm"],
                    "exposed_verifier_context": exposed_context,
                    "exposed_context_sha256": sha256_text(exposed_context),
                    "verify_system_sha256": sha256_text(verify_system),
                    "messages": messages,
                    "model_config": pack["model_config"],
                    "max_attempts": MAX_ATTEMPTS_PER_CELL,
                    "provider_retries_disabled": PROVIDER_RETRIES_DISABLED,
                    "expected_corrected_semantic_pass": asset["expected_corrected_semantic_pass"][cell_id],
                    "telemetry_contract": [
                        "cell_id",
                        "execution_git_sha",
                        "clean_state_attestation",
                        "source_sha256",
                        "complete_source_text",
                        "exposure_arm",
                        "exposed_verifier_context",
                        "exposed_context_sha256",
                        "verify_system_message",
                        "verify_user_message",
                        "prompt_hashes",
                        "draft_sha256",
                        "raw_provider_response",
                        "provider_response_id",
                        "requested_model_alias",
                        "returned_model_identity",
                        "finish_reason",
                        "parsed_claims",
                        "parsed_statuses",
                        "parse_errors",
                        "evidence_bindings",
                        "semantic_oracle_inputs_outputs",
                        "shadow_runtime_decision",
                        "input_tokens",
                        "output_tokens",
                        "cache_tokens",
                        "price_schedule_id",
                        "calculated_cost_usd",
                        "actual_cost_usd",
                        "latency_ms",
                        "timestamps",
                        "artifact_sha256",
                    ],
                }
    raise EvidenceExposure2DError(f"cell {cell_id} not found")


def build_all_requests(pack: dict[str, Any]) -> list[dict[str, Any]]:
    return [build_cell_request(pack, cell_id) for cell_id in CELL_IDS]


def load_price_schedule(path: Path | str = DEFAULT_PRICE_SCHEDULE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def estimate_cell_cost_usd(request: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    input_tokens = sum(count_tokens(message["content"]) for message in request["messages"])
    input_chars = sum(len(message["content"]) for message in request["messages"])
    output_tokens = schedule["max_output_tokens_per_cell"]
    input_cost = input_tokens * schedule["input_cost_per_million_tokens_usd"] / 1_000_000
    output_cost = output_tokens * schedule["output_cost_per_million_tokens_usd"] / 1_000_000
    char_input_cost = input_chars * schedule["input_cost_per_million_tokens_usd"] / 1_000_000
    return {
        "cell_id": request["cell_id"],
        "input_tokens": input_tokens,
        "input_characters": input_chars,
        "output_tokens_bound": output_tokens,
        "estimated_max_cost_usd": round(input_cost + output_cost, 6),
        "char_bound_max_cost_usd": round(char_input_cost + output_cost, 6),
    }


def preflight_budget(
    pack: dict[str, Any],
    *,
    schedule_path: Path | str = DEFAULT_PRICE_SCHEDULE,
    ceiling_usd: float = HARD_SPEND_CEILING_USD,
) -> dict[str, Any]:
    schedule = load_price_schedule(schedule_path)
    validate_price_schedule(schedule)
    requests = build_all_requests(pack)
    estimates = [estimate_cell_cost_usd(request, schedule) for request in requests]
    estimated_total = round(sum(item["estimated_max_cost_usd"] for item in estimates), 6)
    char_bound_total = round(sum(item["char_bound_max_cost_usd"] for item in estimates), 6)
    estimated_authorized = estimated_total <= ceiling_usd
    char_bound_authorized = char_bound_total <= ceiling_usd
    authorized = estimated_authorized and char_bound_authorized
    return {
        "schedule_id": schedule["schedule_id"],
        "model_alias": schedule["model_alias"],
        "input_price_per_million_usd": schedule["input_cost_per_million_tokens_usd"],
        "output_price_per_million_usd": schedule["output_cost_per_million_tokens_usd"],
        "cell_estimates": estimates,
        "estimated_max_spend_usd": estimated_total,
        "char_bound_max_spend_usd": char_bound_total,
        "conservative_max_spend_usd": estimated_total,
        "estimated_budget_authorized": estimated_authorized,
        "char_bound_budget_authorized": char_bound_authorized,
        "hard_ceiling_usd": ceiling_usd,
        "budget_authorized": authorized,
        "provider_execution_authorized_flag": provider_execution_authorized(),
    }


def verify_paired_isolation(pack: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for asset_id in PAIRED_ASSETS:
        asset = find_asset(pack, asset_id)
        prefix_cell = next(cell for cell in asset["cells"] if cell["exposure_arm"] == "prefix")
        complete_cell = next(cell for cell in asset["cells"] if cell["exposure_arm"] == "complete")
        prefix_req = build_cell_request(pack, prefix_cell["cell_id"])
        complete_req = build_cell_request(pack, complete_cell["cell_id"])
        same = (
            prefix_req["draft_text"] == complete_req["draft_text"]
            and prefix_req["draft_sha256"] == complete_req["draft_sha256"]
            and prefix_req["source_sha256"] == complete_req["source_sha256"]
            and prefix_req["complete_source_text"] == complete_req["complete_source_text"]
            and prefix_req["model_config"] == complete_req["model_config"]
            and prefix_req["verify_system_sha256"] == complete_req["verify_system_sha256"]
            and prefix_req["messages"][0] == complete_req["messages"][0]
            and prefix_req["max_attempts"] == complete_req["max_attempts"]
            and prefix_req["provider_retries_disabled"] == complete_req["provider_retries_disabled"]
            and prefix_req["exposure_arm"] != complete_req["exposure_arm"]
            and prefix_req["exposed_context_sha256"] != complete_req["exposed_context_sha256"]
            and _user_messages_differ_only_by_context(prefix_req, complete_req)
        )
        results[asset_id] = {
            "isolated": same,
            "prefix_context_sha256": prefix_req["exposed_context_sha256"],
            "complete_context_sha256": complete_req["exposed_context_sha256"],
        }
    return results


def validate_execution_preflight(pack: dict[str, Any]) -> dict[str, Any]:
    validate_pack(pack)
    validate_stage2d_semantic_fixtures(pack)
    budget = preflight_budget(pack)
    paired = verify_paired_isolation(pack)
    paired_ok = all(item["isolated"] for item in paired.values())
    approved_execution_config_hash = build_approved_execution_config_hash(pack)
    return {
        "budget": budget,
        "paired_isolation": paired,
        "paired_isolation_ok": paired_ok,
        "approved_execution_config_hash": approved_execution_config_hash,
        "execution_preflight_ok": paired_ok and budget["budget_authorized"],
    }


def get_execution_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_clean_state_attestation(pack: dict[str, Any]) -> str:
    payload = {
        "pack_id": pack["pack_id"],
        "evaluator_id": pack["evaluator_id"],
        "schema_version": pack["schema_version"],
        "cell_ids": list(CELL_IDS),
        "verify_system_sha256": sha256_text(load_verify_system()),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=True))


def enrich_cell_artifact(
    request: dict[str, Any],
    artifact: dict[str, Any],
    schedule: dict[str, Any],
    *,
    git_sha: str,
    clean_attestation: str,
) -> dict[str, Any]:
    enriched = dict(artifact)
    enriched["execution_git_sha"] = git_sha
    enriched["clean_state_attestation"] = clean_attestation
    enriched["source_sha256"] = request["source_sha256"]
    enriched["complete_source_text"] = request["complete_source_text"]
    enriched["exposed_verifier_context"] = request["exposed_verifier_context"]
    enriched["exposed_context_sha256"] = request["exposed_context_sha256"]
    enriched["draft_sha256"] = request["draft_sha256"]
    enriched["verify_system_message"] = request["messages"][0]["content"]
    enriched["verify_user_message"] = request["messages"][1]["content"]
    enriched["prompt_hashes"] = {
        "verify_system_sha256": request["verify_system_sha256"],
        "exposed_context_sha256": request["exposed_context_sha256"],
        "draft_sha256": request["draft_sha256"],
    }
    enriched["price_schedule_id"] = schedule["schedule_id"]
    input_tokens = enriched.get("input_tokens") or 0
    output_tokens = enriched.get("output_tokens") or schedule["max_output_tokens_per_cell"]
    enriched["calculated_cost_usd"] = round(
        input_tokens * schedule["input_cost_per_million_tokens_usd"] / 1_000_000
        + output_tokens * schedule["output_cost_per_million_tokens_usd"] / 1_000_000,
        6,
    )
    enriched["timestamps"] = {"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    payload = {key: value for key, value in enriched.items() if key != "artifact_sha256"}
    enriched["artifact_sha256"] = sha256_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    )
    return enriched


def _provider_failure_reason(exc: Exception) -> str:
    if isinstance(exc, ExecutionAuthorization2DError):
        return "execution_authorization_error"
    if isinstance(exc, SemanticValidation2DError):
        return "semantic_validation_error"
    if isinstance(exc, SemanticMapping2DError):
        return "semantic_mapping_error"
    if isinstance(exc, APITimeoutError):
        return "provider_timeout"
    if isinstance(exc, APIConnectionError):
        return "provider_connection_error"
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) in {301, 302, 303, 307, 308}:
        return "provider_redirect_error"
    if isinstance(exc, (APIStatusError, RateLimitError, AuthenticationError, BadRequestError)):
        return "provider_api_error"
    if isinstance(exc, EvidenceExposure2DError):
        return "semantic_mapping_error"
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return "provider_parse_error"
    return "provider_execution_error"


def _invoke_stage2d_provider_create(
    create: Callable[..., Any],
    *,
    request: dict[str, Any],
) -> Any:
    return create(
        model=request["model_config"]["model_alias"],
        messages=request["messages"],
        temperature=request["model_config"]["temperature"],
        max_tokens=request["model_config"]["max_tokens"],
    )


def execute_cell_once(
    pack: dict[str, Any],
    request: dict[str, Any],
    *,
    client: OpenAI | None = None,
    run_state: RunExecutionState | None = None,
    call_hook: Callable[..., Any] | None = None,
    execution_auth: Stage2DExecutionAuthorization | None = None,
) -> dict[str, Any]:
    """Execute exactly one provider attempt for a Stage 2D cell. Never retries."""
    _assert_transport_authorized(pack, request, execution_auth)
    asset = find_asset(pack, request["asset_id"])
    started = time.time()
    state = run_state or RunExecutionState()
    requested_alias = request["model_config"]["model_alias"]
    artifact: dict[str, Any] = {
        "cell_id": request["cell_id"],
        "asset_id": request["asset_id"],
        "requested_model_alias": requested_alias,
        "provider_path": STAGE2D_PROVIDER_PATH,
        "max_attempts": MAX_ATTEMPTS_PER_CELL,
        "provider_retries_disabled": PROVIDER_RETRIES_DISABLED,
        "approved_execution_config_hash": request.get("approved_execution_config_hash"),
        "executed_max_tokens": request["model_config"]["max_tokens"],
        "disposition": "INVALID",
    }

    try:
        if call_hook is not None:
            create = call_hook
        elif client is not None:
            create = client.chat.completions.create
        else:
            create = build_stage2d_client().chat.completions.create
        response = _invoke_stage2d_provider_create(create, request=request)
        raw = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        returned_model = response.model
        model_error = state.check_returned_model_identity(returned_model)
        if model_error:
            artifact.update(
                {
                    "invalid_reason": model_error,
                    "returned_model_identity": returned_model,
                    "finish_reason": finish_reason,
                    "raw_provider_response": raw,
                }
            )
            state.cell_artifacts.append(artifact)
            return artifact

        grounding_report = parse_verifier_output(raw)
        semantic_fixture = map_provider_to_semantic_fixture(asset, grounding_report)
        semantic_result = evaluate_semantic_oracle(
            semantic_fixture,
            template_fixture=asset["semantic_fixture"],
        )
        shadow = shadow_runtime_acceptance(grounding_report)
        scored = score_cell_result(
            cell_id=request["cell_id"],
            asset=asset,
            grounding_report=grounding_report,
            semantic_result=semantic_result,
            shadow=shadow,
        )
        usage = getattr(response, "usage", None)
        artifact.update(
            {
                "disposition": scored["disposition"],
                "invalid_reason": scored.get("invalid_reason"),
                "qualification_fail_reasons": scored.get("qualification_fail_reasons", []),
                "raw_provider_response": raw,
                "provider_response_id": getattr(response, "id", None),
                "returned_model_identity": returned_model,
                "finish_reason": finish_reason,
                "parsed_claims": grounding_report,
                "parsed_statuses": [row.get("status") for row in grounding_report],
                "semantic_oracle_inputs_outputs": semantic_result,
                "shadow_runtime_decision": shadow,
                "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "latency_ms": round((time.time() - started) * 1000, 3),
                "expected_corrected_semantic_pass": asset["expected_corrected_semantic_pass"][request["cell_id"]],
            }
        )
    except Exception as exc:
        artifact.update(
            {
                "invalid_reason": _provider_failure_reason(exc),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "latency_ms": round((time.time() - started) * 1000, 3),
            }
        )

    state.cell_artifacts.append(artifact)
    return artifact


def score_cell_result(
    *,
    cell_id: str,
    asset: dict[str, Any],
    grounding_report: list[dict[str, Any]] | None,
    semantic_result: dict[str, Any] | None,
    shadow: dict[str, Any] | None,
    invalid_reason: str | None = None,
) -> dict[str, Any]:
    if invalid_reason:
        return {"cell_id": cell_id, "disposition": "INVALID", "invalid_reason": invalid_reason}

    expected_pass = asset["expected_corrected_semantic_pass"][cell_id]
    semantic_pass = bool(semantic_result and semantic_result["oracle"]["semantic_pass"])
    false_verification = int(
        semantic_result["metrics"]["material_false_verification_rate.v2"]["numerator"]
    )
    automatic_false_pass = semantic_pass is False and bool(shadow and shadow.get("accepted"))
    qualification_fail_reasons: list[str] = []

    if false_verification > 0:
        qualification_fail_reasons.append("material_false_verification")
    if automatic_false_pass:
        qualification_fail_reasons.append("automatic_semantic_false_pass")
    if semantic_pass != expected_pass:
        qualification_fail_reasons.append("semantic_expectation_mismatch")
    if cell_id in HIGH_RISK_PREFIX_CELLS and (false_verification > 0 or automatic_false_pass):
        qualification_fail_reasons.append("high_risk_prefix_unsafe_outcome")

    disposition = "PASS" if not qualification_fail_reasons else "FAIL"
    return {
        "cell_id": cell_id,
        "disposition": disposition,
        "invalid_reason": None,
        "qualification_fail_reasons": qualification_fail_reasons,
        "semantic_pass": semantic_pass,
        "expected_semantic_pass": expected_pass,
        "false_verification_numerator": false_verification,
        "automatic_false_pass": automatic_false_pass,
        "grounding_report": grounding_report,
        "shadow_runtime_decision": shadow,
    }


def validate_run_cell_catalog(cell_results: list[dict[str, Any]]) -> dict[str, Any]:
    cell_ids = [result["cell_id"] for result in cell_results]
    seen: set[str] = set()
    duplicate_cells: list[str] = []
    for cell_id in cell_ids:
        if cell_id in seen and cell_id not in duplicate_cells:
            duplicate_cells.append(cell_id)
        seen.add(cell_id)
    required = set(CELL_IDS)
    present = set(cell_ids)
    missing_cells = [cell_id for cell_id in CELL_IDS if cell_id not in present]
    extra_cells = sorted(present - required)
    catalog_valid = (
        len(cell_results) == 10
        and len(cell_ids) == 10
        and not duplicate_cells
        and not missing_cells
        and not extra_cells
    )
    return {
        "catalog_valid": catalog_valid,
        "cell_count": len(cell_results),
        "duplicate_cells": duplicate_cells,
        "missing_cells": missing_cells,
        "extra_cells": extra_cells,
        "attempted_cells": cell_ids,
    }


def score_run_results(
    pack: dict[str, Any],
    cell_results: list[dict[str, Any]],
    *,
    run_state: RunExecutionState | None = None,
    attempted_cells: list[str] | None = None,
    unattempted_cells: list[str] | None = None,
) -> dict[str, Any]:
    catalog = validate_run_cell_catalog(cell_results)
    invalid_reasons: list[str] = []
    if run_state and run_state.model_identity_invalid:
        invalid_reasons.append("model_identity_invalid")
    if run_state and run_state.run_aborted and run_state.abort_reason:
        invalid_reasons.append(run_state.abort_reason)
    if not catalog["catalog_valid"]:
        if catalog["cell_count"] != 10:
            invalid_reasons.append("incomplete_cell_count")
        if catalog["missing_cells"]:
            invalid_reasons.append("missing_cells")
        if catalog["duplicate_cells"]:
            invalid_reasons.append("duplicate_cells")
        if catalog["extra_cells"]:
            invalid_reasons.append("extra_cells")

    material_false_verification = 0
    automatic_false_pass = 0
    invalid_cells: list[str] = []
    failed_cells: list[str] = []

    for result in cell_results:
        cell_id = result["cell_id"]
        if cell_id not in CELL_IDS:
            continue
        if result.get("disposition") == "INVALID" or result.get("invalid_reason"):
            invalid_cells.append(cell_id)
            continue
        asset = find_asset(pack, result["asset_id"])
        semantic = result.get("semantic_oracle_inputs_outputs")
        shadow = result.get("shadow_runtime_decision")
        if semantic:
            material_false_verification += int(
                semantic["metrics"]["material_false_verification_rate.v2"]["numerator"]
            )
            if not semantic["oracle"]["semantic_pass"] and shadow and shadow.get("accepted"):
                automatic_false_pass += 1
        scored = score_cell_result(
            cell_id=cell_id,
            asset=asset,
            grounding_report=result.get("parsed_claims"),
            semantic_result=semantic,
            shadow=shadow,
        )
        if scored["disposition"] == "FAIL":
            failed_cells.append(cell_id)

    if invalid_reasons or invalid_cells or not catalog["catalog_valid"]:
        overall = "INVALID"
    elif failed_cells or material_false_verification > 0 or automatic_false_pass > 0:
        overall = "FAIL"
    else:
        overall = "PASS"

    return {
        "overall_disposition": overall,
        "catalog": catalog,
        "invalid_reasons": invalid_reasons,
        "invalid_cells": invalid_cells,
        "failed_cells": failed_cells,
        "material_false_verification_rate.v2_numerator": material_false_verification,
        "automatic_semantic_false_pass_rate.v2_numerator": automatic_false_pass,
        "cell_results": cell_results,
        "attempted_cells": attempted_cells or catalog["attempted_cells"],
        "unattempted_cells": unattempted_cells or [],
    }


def score_mocked_provider_output(
    pack: dict[str, Any],
    *,
    cell_id: str,
    raw_provider_response: str,
    returned_model_identity: str = "deepseek-chat",
) -> dict[str, Any]:
    """Deterministic scorer for mocked provider outputs without network I/O."""
    request = build_cell_request(pack, cell_id)
    asset = find_asset(pack, request["asset_id"])
    state = RunExecutionState()
    try:
        model_error = state.check_returned_model_identity(returned_model_identity)
        if model_error:
            return {
                "cell_id": cell_id,
                "disposition": "INVALID",
                "invalid_reason": model_error,
            }
        grounding_report = parse_verifier_output(raw_provider_response)
        semantic_fixture = map_provider_to_semantic_fixture(asset, grounding_report)
        semantic_result = evaluate_semantic_oracle(
            semantic_fixture,
            template_fixture=asset["semantic_fixture"],
        )
        shadow = shadow_runtime_acceptance(grounding_report)
        scored = score_cell_result(
            cell_id=cell_id,
            asset=asset,
            grounding_report=grounding_report,
            semantic_result=semantic_result,
            shadow=shadow,
        )
        return {
            **scored,
            "parsed_claims": grounding_report,
            "semantic_oracle_inputs_outputs": semantic_result,
            "shadow_runtime_decision": shadow,
            "returned_model_identity": returned_model_identity,
        }
    except Exception as exc:
        return {
            "cell_id": cell_id,
            "disposition": "INVALID",
            "invalid_reason": _provider_failure_reason(exc),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def execute_stage2d_run(
    pack: dict[str, Any],
    *,
    client: OpenAI | None = None,
    call_hook: Callable[..., Any] | None = None,
    schedule_path: Path | str = DEFAULT_PRICE_SCHEDULE,
) -> dict[str, Any]:
    """Execute exactly one Stage 2D qualification run over the frozen ten-cell catalog."""
    if not provider_execution_authorized():
        raise EvidenceExposure2DError(
            "provider execution disabled; set EVIDENCE_EXPOSURE_2D_EXECUTE=1 after independent preflight review"
        )
    execution_auth = issue_stage2d_execution_authorization(pack, schedule_path=schedule_path)
    schedule = load_price_schedule(schedule_path)
    validate_price_schedule(schedule)
    git_sha = get_execution_git_sha()
    clean_attestation = build_clean_state_attestation(pack)
    run_state = RunExecutionState()
    cell_results: list[dict[str, Any]] = []
    attempted_cells: list[str] = []

    for cell_id in CELL_IDS:
        if run_state.should_stop_run():
            break
        request = build_cell_request(pack, cell_id)
        request["approved_execution_config_hash"] = execution_auth.approved_execution_config_hash
        artifact = execute_cell_once(
            pack,
            request,
            client=client,
            run_state=run_state,
            call_hook=call_hook,
            execution_auth=execution_auth,
        )
        artifact = enrich_cell_artifact(
            request,
            artifact,
            schedule,
            git_sha=git_sha,
            clean_attestation=clean_attestation,
        )
        cell_results.append(artifact)
        attempted_cells.append(cell_id)
        if artifact.get("disposition") == "INVALID" or artifact.get("invalid_reason"):
            run_state.run_aborted = True
            run_state.abort_reason = artifact.get("invalid_reason") or "cell_invalid"
            break
        if run_state.model_identity_invalid:
            break

    unattempted_cells = [cell_id for cell_id in CELL_IDS if cell_id not in attempted_cells]
    report = score_run_results(
        pack,
        cell_results,
        run_state=run_state,
        attempted_cells=attempted_cells,
        unattempted_cells=unattempted_cells,
    )
    report["network_calls"] = len(attempted_cells)
    report["frozen_returned_model_identity"] = run_state.frozen_returned_model_identity
    return report


def execute_provider_if_authorized(
    pack: dict[str, Any],
    cell_id: str,
    *,
    client: OpenAI | None = None,
    run_state: RunExecutionState | None = None,
    call_hook: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if not provider_execution_authorized():
        raise EvidenceExposure2DError(
            "provider execution disabled; set EVIDENCE_EXPOSURE_2D_EXECUTE=1 after independent preflight review"
        )
    execution_auth = issue_stage2d_execution_authorization(pack)
    request = build_cell_request(pack, cell_id)
    request["approved_execution_config_hash"] = execution_auth.approved_execution_config_hash
    return execute_cell_once(
        pack,
        request,
        client=client,
        run_state=run_state,
        call_hook=call_hook,
        execution_auth=execution_auth,
    )


def write_frozen_fixtures(path: Path | str = DEFAULT_FIXTURES) -> None:
    pack = build_frozen_pack()
    Path(path).write_text(json.dumps(pack, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run_preflight(pack: dict[str, Any]) -> dict[str, Any]:
    requests = build_all_requests(pack)
    budget = preflight_budget(pack)
    paired = verify_paired_isolation(pack)
    paired_ok = all(item["isolated"] for item in paired.values())
    semantic_checks = {}
    for asset in pack["assets"]:
        result = evaluate_semantic_oracle(asset["semantic_fixture"])
        semantic_checks[asset["asset_id"]] = {
            "semantic_pass": result["oracle"]["semantic_pass"],
            "expected_by_cell": asset["expected_corrected_semantic_pass"],
        }
    return {
        "pack_id": pack["pack_id"],
        "stage": STAGE,
        "cell_count": len(requests),
        "asset_count": len(pack["assets"]),
        "requests_built": [request["cell_id"] for request in requests],
        "budget": budget,
        "paired_isolation": paired,
        "paired_isolation_ok": paired_ok,
        "semantic_template_checks": semantic_checks,
        "provider_execution_default_disabled": not provider_execution_authorized(),
        "provider_client_config": build_stage2d_client_config(),
        "preflight_ready": budget["budget_authorized"] and paired_ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2D evidence exposure preflight")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--write-fixtures", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.write_fixtures:
        write_frozen_fixtures(args.fixtures)
        print(f"wrote {args.fixtures}")
        return 0
    pack = load_pack(args.fixtures)
    report = run_preflight(pack)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(f"preflight_ready={report['preflight_ready']} cells={report['cell_count']}")
    return 0 if report["preflight_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
