"""Verifier semantic-contract 3-cell prompt-only preflight harness. No provider calls by default."""
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

import httpx
import tiktoken
from openai import APIConnectionError, APITimeoutError, OpenAI
from openai import APIStatusError, AuthenticationError, BadRequestError, RateLimitError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FIXTURES = REPO_ROOT / "evals" / "fixtures" / "verifier_semantic_contract_3cell.json"
DEFAULT_PRICE_SCHEDULE = REPO_ROOT / "evals" / "verifier_semantic_contract_3cell_price_schedule.json"
STAGE2D_FIXTURES = REPO_ROOT / "evals" / "fixtures" / "evidence_exposure_2d.json"
ORIGINAL_VERIFY_SYSTEM_PATH = REPO_ROOT / "prompts" / "verify_system.md"
CANDIDATE_VERIFY_SYSTEM_PATH = REPO_ROOT / "prompts" / "verify_system_semantic_contract_candidate.md"
STAGE2D_REFERENCE_ARTIFACT = (
    REPO_ROOT / "outputs" / "evidence_exposure_2d" / "stage2d_run_20260905T074337Z.json"
)
STAGE2D_REFERENCE_SHA256 = "7243f2075a06aba3d48c69085e085b7b7b57bf59c7b926b6eafb29991e914ea0"

PACK_ID = "verifier_semantic_contract_3cell"
EVALUATOR_ID = "verifier_semantic_contract_3cell_preflight"
SCHEMA_VERSION = 1
STAGE = "3cell-prompt-contract"

HARD_SPEND_CEILING_USD = 0.02
MAX_OUTPUT_TOKENS = 4000
MAX_PROVIDER_REQUESTS = 3
VERIFY_TEMPERATURE = 0.1
MAX_ATTEMPTS_PER_CELL = 1
PROVIDER_RETRIES_DISABLED = True
PROVIDER_PATH = "verifier_semantic_contract_3cell_direct_openai"

CELL_IDS = ("C1", "C2", "C3")
ASSET_IDS = ("P6", "P7", "P1")

CELL_SOURCE_MAP: dict[str, dict[str, str]] = {
    "C1": {"asset_id": "P6", "source_cell_id": "P6-COMPLETE", "exposure_arm": "complete"},
    "C2": {"asset_id": "P7", "source_cell_id": "P7-COMPLETE", "exposure_arm": "complete"},
    "C3": {"asset_id": "P1", "source_cell_id": "P1-COMPLETE", "exposure_arm": "complete"},
}

EXPECTED_PROVIDER_CLASSIFICATION: dict[str, str] = {
    "C1": "weak",
    "C2": "weak",
    "C3": "verified",
}

EXPECTED_SEMANTIC_P0_PASS: dict[str, bool] = {
    "C1": False,
    "C2": False,
    "C3": True,
}

ORIGINAL_STATUS_PARAGRAPH = (
    'A claim is "verified" if any source supports its substance, INCLUDING through paraphrase '
    "or synthesis — the claim need not match source wording, only source meaning. A claim is "
    '"weak" only if support is partial or indirect. A claim is "unverified" only if NO source '
    "addresses its substance."
)

CANDIDATE_STATUS_BLOCK = """## Semantic-status contract

Evaluate each claim exactly as asserted against all verifier-visible evidence considered together.

* **verified:** The evidence fully entails every truth-relevant part of the claim, including all material subjects, actions, objects, quantities, units, versions, scopes, conditions, qualifiers, time limits, and exceptions. Any material contradiction prevents verified. Conditional or qualified evidence cannot verify a stronger unconditional claim.

* **weak:** Meaningful positive support exists, but full entailment is absent because a material part is unsupported, indirect, ambiguous, conditional, qualified, or contradicted. Support plus conflicting evidence is weak.

* **unverified:** No meaningful positive support exists; the evidence is absent, unrelated, only topical, unavailable, invalidly bound, or contradictory without meaningful positive support.

Reconcile all relevant evidence before assigning the status. Do not use verified merely because one passage supports part of the claim."""

CANDIDATE_PROMPT_SHA256 = "adc0200c5e600cc7de98a42c0241dd13df0c049c48a00148dbc0fffa7b741b57"
ORIGINAL_PRODUCTION_PROMPT_SHA256 = "e7ac8744409d63a879d4df85b7e4240b1d447a36d91506731f6c7ba00217809a"


class VerifierSemanticContractError(ValueError):
    """3-cell prompt-contract pack or preflight asset is not evaluable."""


class PromptDiffError(VerifierSemanticContractError):
    """Candidate verifier prompt differs outside the semantic-status paragraph."""


class ExecutionAuthorizationError(VerifierSemanticContractError):
    """3-cell execution authorization missing or invalid at the transport boundary."""


class SemanticMappingError(VerifierSemanticContractError):
    """Live provider output could not be mapped into semantic fixture inputs."""


class SemanticValidationError(VerifierSemanticContractError):
    """Constructed semantic fixture failed validation before metric computation."""


@dataclass(frozen=True)
class ExecutionAuthorization:
    authorization_token: str
    approved_execution_config_hash: str
    execution_git_sha: str
    budget_authorized: bool
    provider_execution_authorized: bool


@dataclass
class RunExecutionState:
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifierSemanticContractError(message)


def provider_execution_authorized() -> bool:
    return os.getenv("VERIFIER_SEMANTIC_CONTRACT_EXECUTE", "").strip() == "1"


def load_original_verify_system() -> str:
    return ORIGINAL_VERIFY_SYSTEM_PATH.read_text(encoding="utf-8")


def load_candidate_verify_system() -> str:
    text = CANDIDATE_VERIFY_SYSTEM_PATH.read_text(encoding="utf-8")
    _require(sha256_text(text) == CANDIDATE_PROMPT_SHA256, "candidate prompt hash drift")
    return text


def validate_candidate_prompt_hash() -> None:
    candidate = CANDIDATE_VERIFY_SYSTEM_PATH.read_text(encoding="utf-8")
    _require(sha256_text(candidate) == CANDIDATE_PROMPT_SHA256, "candidate prompt hash mismatch")
    _require(
        sha256_text(load_original_verify_system()) == ORIGINAL_PRODUCTION_PROMPT_SHA256,
        "production prompt hash drift",
    )


def build_candidate_verify_system_from_original(original: str | None = None) -> str:
    original = original or load_original_verify_system()
    _require(ORIGINAL_STATUS_PARAGRAPH in original, "original status paragraph not found")
    return original.replace(ORIGINAL_STATUS_PARAGRAPH, CANDIDATE_STATUS_BLOCK, 1)


def validate_prompt_diff_only_status() -> dict[str, Any]:
    original = load_original_verify_system()
    candidate = load_candidate_verify_system()
    expected_candidate = build_candidate_verify_system_from_original(original)
    if candidate != expected_candidate:
        raise PromptDiffError(
            "candidate prompt differs outside the semantic-status paragraph replacement"
        )
    original_without_status = original.replace(ORIGINAL_STATUS_PARAGRAPH, "", 1)
    candidate_without_status = candidate.replace(CANDIDATE_STATUS_BLOCK, "", 1)
    if original_without_status != candidate_without_status:
        raise PromptDiffError("non-status prompt regions differ after status-block removal")
    return {
        "valid": True,
        "original_verify_system_path": str(ORIGINAL_VERIFY_SYSTEM_PATH.relative_to(REPO_ROOT)),
        "candidate_verify_system_path": str(CANDIDATE_VERIFY_SYSTEM_PATH.relative_to(REPO_ROOT)),
        "original_prompt_sha256": sha256_text(original),
        "candidate_prompt_sha256": sha256_text(candidate),
        "only_status_paragraph_changed": True,
    }


def load_stage2d_pack() -> dict[str, Any]:
    return json.loads(STAGE2D_FIXTURES.read_text(encoding="utf-8"))


def find_stage2d_asset(pack: dict[str, Any], asset_id: str) -> dict[str, Any]:
    for asset in pack["assets"]:
        if asset["asset_id"] == asset_id:
            return asset
    raise VerifierSemanticContractError(f"Stage 2D asset {asset_id} not found")


def expose_source_context(*, source_kind: str, source_text: str, source_url: str, exposure_arm: str) -> str:
    from scripts.evidence_exposure_2d_preflight import expose_source_context as _expose

    return _expose(
        source_kind=source_kind,
        source_text=source_text,
        source_url=source_url,
        exposure_arm=exposure_arm,
    )


def build_verifier_user_message(draft_text: str, source_context: str) -> str:
    from scripts.evidence_exposure_2d_preflight import build_verifier_user_message as _build

    return _build(draft_text, source_context)


def canonicalize_semantic_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    from scripts.evidence_exposure_2d_preflight import canonicalize_semantic_fixture as _canon

    return _canon(fixture)


def validate_stage2d_semantic_fixture(live_fixture: dict[str, Any], template_fixture: dict[str, Any]) -> None:
    from scripts.evidence_exposure_2d_preflight import validate_stage2d_semantic_fixture as _validate

    _validate(live_fixture, template_fixture)


def evaluate_semantic_oracle(
    semantic_fixture: dict[str, Any],
    *,
    template_fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from scripts.evidence_exposure_2d_preflight import evaluate_semantic_oracle as _evaluate

    return _evaluate(semantic_fixture, template_fixture=template_fixture)


def parse_verifier_output(raw: str) -> list[dict[str, Any]]:
    from scripts.evidence_exposure_2d_preflight import parse_verifier_output as _parse

    return _parse(raw)


def map_provider_to_semantic_fixture(
    asset: dict[str, Any],
    grounding_report: list[dict[str, Any]],
) -> dict[str, Any]:
    from scripts.evidence_exposure_2d_preflight import map_provider_to_semantic_fixture as _map

    return _map(asset, grounding_report)


def shadow_runtime_acceptance(
    grounding_report: list[dict[str, Any]],
    *,
    verification_status: str = "completed",
    uvr_threshold: float = 0.15,
) -> dict[str, Any]:
    from scripts.evidence_exposure_2d_preflight import shadow_runtime_acceptance as _shadow

    return _shadow(
        grounding_report,
        verification_status=verification_status,
        uvr_threshold=uvr_threshold,
    )


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def build_frozen_pack() -> dict[str, Any]:
    prompt_identity = validate_prompt_diff_only_status()
    stage2d = load_stage2d_pack()
    assets: list[dict[str, Any]] = []
    for cell_id in CELL_IDS:
        mapping = CELL_SOURCE_MAP[cell_id]
        source_asset = find_stage2d_asset(stage2d, mapping["asset_id"])
        assets.append(
            {
                "asset_id": mapping["asset_id"],
                "cell_id": cell_id,
                "source_cell_id": mapping["source_cell_id"],
                "title": source_asset["title"],
                "source": source_asset["source"],
                "draft_text": source_asset["draft_text"],
                "draft_sha256": source_asset["draft_sha256"],
                "exposure_arm": mapping["exposure_arm"],
                "expected_provider_classification": EXPECTED_PROVIDER_CLASSIFICATION[cell_id],
                "expected_corrected_semantic_pass": EXPECTED_SEMANTIC_P0_PASS[cell_id],
                "semantic_fixture": source_asset["semantic_fixture"],
            }
        )
    pack = {
        "pack_id": PACK_ID,
        "evaluator_id": EVALUATOR_ID,
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "stage2d_reference_artifact": {
            "path": str(STAGE2D_REFERENCE_ARTIFACT.relative_to(REPO_ROOT)),
            "sha256": STAGE2D_REFERENCE_SHA256,
        },
        "model_config": stage2d["model_config"],
        "prompt_identity": {
            "original_verify_system_path": prompt_identity["original_verify_system_path"],
            "original_prompt_sha256": prompt_identity["original_prompt_sha256"],
            "candidate_verify_system_path": prompt_identity["candidate_verify_system_path"],
            "candidate_prompt_sha256": prompt_identity["candidate_prompt_sha256"],
            "only_status_paragraph_changed": True,
        },
        "qualification_gates": {
            "material_false_verification_rate.v2_numerator": 0,
        },
        "diagnostic_gates": {
            "automatic_semantic_false_pass_rate.v2_numerator": "diagnostic_only_not_qualification",
        },
        "max_provider_requests": MAX_PROVIDER_REQUESTS,
        "assets": assets,
    }
    validate_pack(pack)
    for asset in pack["assets"]:
        asset["semantic_fixture"] = canonicalize_semantic_fixture(asset["semantic_fixture"])
    return pack


def load_pack(path: Path | str = DEFAULT_FIXTURES) -> dict[str, Any]:
    pack = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_pack(pack)
    return pack


def find_asset(pack: dict[str, Any], asset_id: str) -> dict[str, Any]:
    for asset in pack["assets"]:
        if asset["asset_id"] == asset_id:
            return asset
    raise VerifierSemanticContractError(f"asset {asset_id} not found")


def find_asset_by_cell(pack: dict[str, Any], cell_id: str) -> dict[str, Any]:
    for asset in pack["assets"]:
        if asset["cell_id"] == cell_id:
            return asset
    raise VerifierSemanticContractError(f"cell {cell_id} not found")


def validate_frozen_model_config(pack: dict[str, Any]) -> None:
    model_config = pack["model_config"]
    _require(model_config["temperature"] == VERIFY_TEMPERATURE, "temperature drift")
    _require(model_config["max_tokens"] == MAX_OUTPUT_TOKENS, "max_tokens drift")
    _require(model_config["model_alias"] == "deepseek-chat", "model alias drift")


def validate_stage2d_frozen_identity(pack: dict[str, Any]) -> dict[str, Any]:
    """Prove 3-cell frozen inputs match Stage 2D COMPLETE cells except verifier prompt."""
    from scripts.evidence_exposure_2d_preflight import build_cell_request as build_stage2d_request

    stage2d = load_stage2d_pack()
    results: dict[str, Any] = {}
    for cell_id in CELL_IDS:
        mapping = CELL_SOURCE_MAP[cell_id]
        asset = find_asset_by_cell(pack, cell_id)
        req_3cell = build_cell_request(pack, cell_id)
        req_2d = build_stage2d_request(stage2d, mapping["source_cell_id"])
        same_frozen = (
            req_3cell["draft_text"] == req_2d["draft_text"]
            and req_3cell["draft_sha256"] == req_2d["draft_sha256"]
            and req_3cell["source_sha256"] == req_2d["source_sha256"]
            and req_3cell["complete_source_text"] == req_2d["complete_source_text"]
            and req_3cell["exposed_context_sha256"] == req_2d["exposed_context_sha256"]
            and req_3cell["exposure_arm"] == req_2d["exposure_arm"]
            and req_3cell["model_config"] == req_2d["model_config"]
            and req_3cell["verify_system_sha256"] != req_2d["verify_system_sha256"]
            and asset["draft_text"] == req_2d["draft_text"]
        )
        results[cell_id] = {
            "source_cell_id": mapping["source_cell_id"],
            "asset_id": mapping["asset_id"],
            "frozen_identity_ok": same_frozen,
            "draft_sha256": req_3cell["draft_sha256"],
            "source_sha256": req_3cell["source_sha256"],
            "exposed_context_sha256": req_3cell["exposed_context_sha256"],
            "stage2d_verify_system_sha256": req_2d["verify_system_sha256"],
            "candidate_verify_system_sha256": req_3cell["verify_system_sha256"],
        }
    return {
        "cells": results,
        "all_frozen_identity_ok": all(item["frozen_identity_ok"] for item in results.values()),
    }


def validate_pack(pack: dict[str, Any]) -> None:
    _require(pack["pack_id"] == PACK_ID, "pack_id mismatch")
    _require(pack["evaluator_id"] == EVALUATOR_ID, "evaluator_id mismatch")
    _require(pack["schema_version"] == SCHEMA_VERSION, "schema_version mismatch")
    _require(pack.get("stage") == STAGE, "stage mismatch")
    validate_frozen_model_config(pack)
    assets = pack["assets"]
    _require(len(assets) == 3, "asset count must be exactly 3")
    cell_ids = [asset["cell_id"] for asset in assets]
    _require(cell_ids == list(CELL_IDS), "cell catalog mismatch")
    asset_ids = [asset["asset_id"] for asset in assets]
    _require(asset_ids == list(ASSET_IDS), "asset order mismatch")
    prompt_identity = pack.get("prompt_identity", {})
    _require(prompt_identity.get("only_status_paragraph_changed") is True, "prompt identity invalid")
    for cell_id in CELL_IDS:
        asset = find_asset_by_cell(pack, cell_id)
        _require(
            asset["expected_provider_classification"] == EXPECTED_PROVIDER_CLASSIFICATION[cell_id],
            f"expected provider classification mismatch for {cell_id}",
        )
        _require(
            asset["exposure_arm"] == "complete",
            f"{cell_id} must use COMPLETE exposure only",
        )
        mapping = CELL_SOURCE_MAP[cell_id]
        _require(asset["source_cell_id"] == mapping["source_cell_id"], "source cell identity mismatch")


def validate_stage2d_semantic_fixtures(pack: dict[str, Any]) -> None:
    for asset in pack["assets"]:
        template = canonicalize_semantic_fixture(asset["semantic_fixture"])
        validate_stage2d_semantic_fixture(template, template)


def build_cell_request(pack: dict[str, Any], cell_id: str) -> dict[str, Any]:
    asset = find_asset_by_cell(pack, cell_id)
    source = asset["source"]
    verify_system = load_candidate_verify_system()
    exposed_context = expose_source_context(
        source_kind=source["source_kind"],
        source_text=source["source_text"],
        source_url=source["source_url"],
        exposure_arm=asset["exposure_arm"],
    )
    user_message = build_verifier_user_message(asset["draft_text"], exposed_context)
    messages = [
        {"role": "system", "content": verify_system},
        {"role": "user", "content": user_message},
    ]
    return {
        "cell_id": cell_id,
        "asset_id": asset["asset_id"],
        "source_cell_id": asset["source_cell_id"],
        "call_order": CELL_IDS.index(cell_id) + 1,
        "source_id": source["source_id"],
        "source_sha256": source["source_sha256"],
        "complete_source_text": source["source_text"],
        "draft_text": asset["draft_text"],
        "draft_sha256": asset["draft_sha256"],
        "exposure_arm": asset["exposure_arm"],
        "exposed_verifier_context": exposed_context,
        "exposed_context_sha256": sha256_text(exposed_context),
        "verify_system_sha256": sha256_text(verify_system),
        "messages": messages,
        "model_config": pack["model_config"],
        "max_attempts": MAX_ATTEMPTS_PER_CELL,
        "provider_retries_disabled": PROVIDER_RETRIES_DISABLED,
        "expected_provider_classification": asset["expected_provider_classification"],
        "expected_corrected_semantic_pass": asset["expected_corrected_semantic_pass"],
    }


def build_all_requests(pack: dict[str, Any]) -> list[dict[str, Any]]:
    return [build_cell_request(pack, cell_id) for cell_id in CELL_IDS]


def load_price_schedule(path: Path | str = DEFAULT_PRICE_SCHEDULE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        "cell_estimates": estimates,
        "estimated_max_spend_usd": estimated_total,
        "char_bound_max_spend_usd": char_bound_total,
        "conservative_max_spend_usd": char_bound_total,
        "estimated_budget_authorized": estimated_authorized,
        "char_bound_budget_authorized": char_bound_authorized,
        "hard_ceiling_usd": ceiling_usd,
        "budget_authorized": authorized,
        "max_provider_requests": MAX_PROVIDER_REQUESTS,
        "provider_execution_authorized_flag": provider_execution_authorized(),
    }


def build_approved_execution_config_hash(pack: dict[str, Any]) -> str:
    validate_frozen_model_config(pack)
    validate_candidate_prompt_hash()
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


def get_execution_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def issue_execution_authorization(
    pack: dict[str, Any],
    *,
    schedule_path: Path | str = DEFAULT_PRICE_SCHEDULE,
) -> ExecutionAuthorization:
    if not provider_execution_authorized():
        raise ExecutionAuthorizationError(
            "provider execution disabled; set VERIFIER_SEMANTIC_CONTRACT_EXECUTE=1 "
            "after independent preflight review"
        )
    validate_pack(pack)
    validate_stage2d_semantic_fixtures(pack)
    validate_prompt_diff_only_status()
    validate_candidate_prompt_hash()
    frozen = validate_stage2d_frozen_identity(pack)
    if not frozen["all_frozen_identity_ok"]:
        raise ExecutionAuthorizationError("Stage 2D frozen identity check failed")
    budget = preflight_budget(pack, schedule_path=schedule_path)
    if not budget["budget_authorized"]:
        raise ExecutionAuthorizationError("budget not authorized")
    git_sha = get_execution_git_sha()
    approved_hash = build_approved_execution_config_hash(pack)
    token_payload = {
        "approved_execution_config_hash": approved_hash,
        "execution_git_sha": git_sha,
        "budget_authorized": budget["budget_authorized"],
        "provider_execution_authorized": True,
        "cell_ids": list(CELL_IDS),
    }
    return ExecutionAuthorization(
        authorization_token=sha256_text(json.dumps(token_payload, sort_keys=True, ensure_ascii=True)),
        approved_execution_config_hash=approved_hash,
        execution_git_sha=git_sha,
        budget_authorized=budget["budget_authorized"],
        provider_execution_authorized=True,
    )


def build_provider_client_config() -> dict[str, Any]:
    from scripts.evidence_exposure_2d_preflight import build_stage2d_client_config

    config = build_stage2d_client_config()
    config["provider_path"] = PROVIDER_PATH
    return config


def build_3cell_http_client(*, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    from scripts.evidence_exposure_2d_preflight import build_stage2d_http_client

    return build_stage2d_http_client(transport=transport)


def build_3cell_client(
    *,
    api_key: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> OpenAI:
    """Construct a dedicated retry-free OpenAI-compatible client for 3-cell runs."""
    from scripts.evidence_exposure_2d_preflight import build_stage2d_client

    return build_stage2d_client(api_key=api_key, transport=transport)


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
    _require(
        request["verify_system_sha256"] == CANDIDATE_PROMPT_SHA256,
        "candidate prompt hash mismatch on request",
    )
    _require(
        request["messages"][0]["content"] == load_candidate_verify_system(),
        "system message must be candidate verifier prompt",
    )
    _require(
        request["messages"][0]["content"] != load_original_verify_system(),
        "production verifier prompt must not be sent",
    )
    _require(
        rebuilt["messages"][1]["content"] == request["messages"][1]["content"],
        "user message drift from frozen Stage 2D COMPLETE context",
    )


def _assert_transport_authorized(
    pack: dict[str, Any],
    request: dict[str, Any],
    execution_auth: ExecutionAuthorization | None,
) -> None:
    if execution_auth is None:
        raise ExecutionAuthorizationError("missing 3-cell execution authorization")
    if not execution_auth.provider_execution_authorized:
        raise ExecutionAuthorizationError("provider execution not authorized")
    if not execution_auth.budget_authorized:
        raise ExecutionAuthorizationError("budget not authorized")
    approved_hash = request.get("approved_execution_config_hash")
    if approved_hash != execution_auth.approved_execution_config_hash:
        raise ExecutionAuthorizationError("request execution config hash mismatch")
    validate_cell_request_artifact(pack, request)
    _require(
        request["model_config"]["max_tokens"] == MAX_OUTPUT_TOKENS,
        "executed max_tokens differs from approved budget configuration",
    )
    rebuilt_hash = build_approved_execution_config_hash(pack)
    if rebuilt_hash != execution_auth.approved_execution_config_hash:
        raise ExecutionAuthorizationError("execution config drift from approved preflight hash")
    if request["cell_id"] not in CELL_IDS:
        raise ExecutionAuthorizationError(f"cell {request['cell_id']!r} not in frozen catalog")


def _provider_failure_reason(exc: Exception) -> str:
    if isinstance(exc, ExecutionAuthorizationError):
        return "execution_authorization_error"
    if isinstance(exc, SemanticValidationError):
        return "semantic_validation_error"
    if isinstance(exc, SemanticMappingError):
        return "semantic_mapping_error"
    from scripts.evidence_exposure_2d_preflight import SemanticMapping2DError, SemanticValidation2DError

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
    if isinstance(exc, VerifierSemanticContractError):
        return "semantic_mapping_error"
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return "provider_parse_error"
    return "provider_execution_error"


def _invoke_3cell_provider_create(
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


def build_clean_state_attestation(pack: dict[str, Any]) -> str:
    payload = {
        "pack_id": pack["pack_id"],
        "evaluator_id": pack["evaluator_id"],
        "schema_version": pack["schema_version"],
        "cell_ids": list(CELL_IDS),
        "candidate_prompt_sha256": CANDIDATE_PROMPT_SHA256,
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


def execute_cell_once(
    pack: dict[str, Any],
    request: dict[str, Any],
    *,
    client: OpenAI | None = None,
    run_state: RunExecutionState | None = None,
    call_hook: Callable[..., Any] | None = None,
    execution_auth: ExecutionAuthorization | None = None,
) -> dict[str, Any]:
    """Execute exactly one provider attempt for a 3-cell catalog cell. Never retries."""
    _assert_transport_authorized(pack, request, execution_auth)
    asset = find_asset_by_cell(pack, request["cell_id"])
    started = time.time()
    state = run_state or RunExecutionState()
    requested_alias = request["model_config"]["model_alias"]
    artifact: dict[str, Any] = {
        "cell_id": request["cell_id"],
        "asset_id": request["asset_id"],
        "source_cell_id": request["source_cell_id"],
        "requested_model_alias": requested_alias,
        "provider_path": PROVIDER_PATH,
        "max_attempts": MAX_ATTEMPTS_PER_CELL,
        "provider_retries_disabled": PROVIDER_RETRIES_DISABLED,
        "approved_execution_config_hash": request.get("approved_execution_config_hash"),
        "executed_max_tokens": request["model_config"]["max_tokens"],
        "candidate_prompt_sha256": request["verify_system_sha256"],
        "disposition": "INVALID",
    }

    try:
        if call_hook is not None:
            create = call_hook
        elif client is not None:
            create = client.chat.completions.create
        else:
            create = build_3cell_client().chat.completions.create
        response = _invoke_3cell_provider_create(create, request=request)
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
                "expected_corrected_semantic_pass": asset["expected_corrected_semantic_pass"],
                "provider_status": scored.get("provider_status"),
                "expected_provider_status": scored.get("expected_provider_status"),
                "automatic_false_pass": scored.get("automatic_false_pass"),
                "automatic_false_pass_diagnostic_only": True,
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


def execute_verifier_semantic_contract_run(
    pack: dict[str, Any],
    *,
    client: OpenAI | None = None,
    call_hook: Callable[..., Any] | None = None,
    schedule_path: Path | str = DEFAULT_PRICE_SCHEDULE,
) -> dict[str, Any]:
    """Execute exactly one 3-cell qualification run over the frozen catalog."""
    if not provider_execution_authorized():
        raise VerifierSemanticContractError(
            "provider execution disabled; set VERIFIER_SEMANTIC_CONTRACT_EXECUTE=1 "
            "after independent provider-authorization review"
        )
    execution_auth = issue_execution_authorization(pack, schedule_path=schedule_path)
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
    )
    report["network_calls"] = len(attempted_cells)
    report["frozen_returned_model_identity"] = run_state.frozen_returned_model_identity
    report["attempted_cells"] = attempted_cells
    report["unattempted_cells"] = unattempted_cells
    report["execution_authorization_token"] = execution_auth.authorization_token
    report["approved_execution_config_hash"] = execution_auth.approved_execution_config_hash
    return report


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
        len(cell_results) == 3
        and len(cell_ids) == 3
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

    expected_pass = asset["expected_corrected_semantic_pass"]
    expected_provider_status = asset["expected_provider_classification"]
    semantic_pass = bool(semantic_result and semantic_result["oracle"]["semantic_pass"])
    false_verification = int(
        semantic_result["metrics"]["material_false_verification_rate.v2"]["numerator"]
    )
    automatic_false_pass = semantic_pass is False and bool(shadow and shadow.get("accepted"))

    draft_claim = asset["semantic_fixture"]["candidates"][0]["text"]
    provider_status = None
    if grounding_report:
        matches = [row for row in grounding_report if (row.get("claim") or "").strip() == draft_claim.strip()]
        if len(matches) == 1:
            provider_status = matches[0].get("status")

    qualification_fail_reasons: list[str] = []
    if false_verification > 0:
        qualification_fail_reasons.append("material_false_verification")
    if provider_status != expected_provider_status:
        qualification_fail_reasons.append("provider_classification_mismatch")
    if semantic_pass != expected_pass:
        qualification_fail_reasons.append("semantic_expectation_mismatch")

    disposition = "PASS" if not qualification_fail_reasons else "FAIL"
    return {
        "cell_id": cell_id,
        "disposition": disposition,
        "invalid_reason": None,
        "qualification_fail_reasons": qualification_fail_reasons,
        "semantic_pass": semantic_pass,
        "expected_semantic_pass": expected_pass,
        "provider_status": provider_status,
        "expected_provider_status": expected_provider_status,
        "false_verification_numerator": false_verification,
        "automatic_false_pass": automatic_false_pass,
        "automatic_false_pass_diagnostic_only": True,
        "grounding_report": grounding_report,
        "shadow_runtime_decision": shadow,
        "corrected_semantic_p0_disposition": "PASS" if semantic_pass else "FAIL",
    }


def score_run_results(
    pack: dict[str, Any],
    cell_results: list[dict[str, Any]],
    *,
    run_state: RunExecutionState | None = None,
) -> dict[str, Any]:
    catalog = validate_run_cell_catalog(cell_results)
    invalid_reasons: list[str] = []
    if run_state and run_state.model_identity_invalid:
        invalid_reasons.append("model_identity_invalid")
    if run_state and run_state.run_aborted and run_state.abort_reason:
        invalid_reasons.append(run_state.abort_reason)
    if not catalog["catalog_valid"]:
        if catalog["cell_count"] != 3:
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
        asset = find_asset_by_cell(pack, cell_id)
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
    elif failed_cells or material_false_verification > 0:
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
        "automatic_false_pass_diagnostic_only": True,
        "cell_results": cell_results,
    }


def score_mocked_provider_output(
    pack: dict[str, Any],
    *,
    cell_id: str,
    raw_provider_response: str,
    returned_model_identity: str = "deepseek-chat",
) -> dict[str, Any]:
    asset = find_asset_by_cell(pack, cell_id)
    state = RunExecutionState()
    try:
        model_error = state.check_returned_model_identity(returned_model_identity)
        if model_error:
            return {"cell_id": cell_id, "disposition": "INVALID", "invalid_reason": model_error}
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
            "invalid_reason": type(exc).__name__,
            "error_message": str(exc),
        }


def run_preflight(pack: dict[str, Any]) -> dict[str, Any]:
    prompt_identity = validate_prompt_diff_only_status()
    requests = build_all_requests(pack)
    budget = preflight_budget(pack)
    frozen_identity = validate_stage2d_frozen_identity(pack)
    validate_stage2d_semantic_fixtures(pack)
    return {
        "pack_id": pack["pack_id"],
        "stage": STAGE,
        "cell_count": len(requests),
        "asset_count": len(pack["assets"]),
        "requests_built": [request["cell_id"] for request in requests],
        "prompt_identity": prompt_identity,
        "budget": budget,
        "stage2d_frozen_identity": frozen_identity,
        "expected_provider_classifications": EXPECTED_PROVIDER_CLASSIFICATION,
        "expected_semantic_p0_pass": EXPECTED_SEMANTIC_P0_PASS,
        "qualification_gates": pack.get("qualification_gates"),
        "diagnostic_gates": pack.get("diagnostic_gates"),
        "provider_execution_default_disabled": not provider_execution_authorized(),
        "provider_client_config": build_provider_client_config(),
        "transport_runner": "execute_verifier_semantic_contract_run",
        "transport_ready": (
            budget["budget_authorized"]
            and frozen_identity["all_frozen_identity_ok"]
            and prompt_identity["valid"]
        ),
        "preflight_ready": (
            budget["budget_authorized"]
            and frozen_identity["all_frozen_identity_ok"]
            and prompt_identity["valid"]
        ),
    }


def write_frozen_fixtures(path: Path | str = DEFAULT_FIXTURES) -> None:
    pack = build_frozen_pack()
    Path(path).write_text(json.dumps(pack, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verifier semantic-contract 3-cell preflight")
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
