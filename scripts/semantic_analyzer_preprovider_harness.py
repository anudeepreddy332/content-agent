"""Offline semantic-analyzer pre-provider qualification harness. No provider calls by default."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.semantic_analyzer.quote_binding import (  # noqa: E402
    ALLOWED_EXTERNAL_BLOCKER_FIELDS,
    ALLOWED_EXTERNAL_OBSERVATION_FIELDS,
    ALLOWED_QUOTE_FIELDS,
    QuoteBindingError,
    convert_quote_observations_to_canonical,
)
from agent.semantic_analyzer.response_contract import (  # noqa: E402
    FORBIDDEN_ANALYZER_FIELDS,
    ResponseContractError as _ResponseContractError,
    parse_analyzer_response as _parse_analyzer_response,
)
from agent.semantic_analyzer.status_engine import (  # noqa: E402
    ALLOWED_BLOCKER_FIELDS,
    ALLOWED_OBSERVATION_FIELDS,
    ALLOWED_SPAN_FIELDS,
    BLOCKER_KINDS,
    adjudicate_hybrid_verifier_observations,
    build_minimal_envelope,
    extract_span_text,
    sha256_utf8,
)

DEFAULT_FIXTURES = REPO_ROOT / "evals" / "fixtures" / "hybrid_verifier_status_offline.json"
DEFAULT_IDENTITY = REPO_ROOT / "evals" / "fixtures" / "semantic_analyzer_preprovider_harness_identity.json"
HARNESS_MODULE = REPO_ROOT / "scripts" / "semantic_analyzer_preprovider_harness.py"

PACK_ID = "hybrid_verifier_status_offline"
HARNESS_ID = "semantic_analyzer_preprovider_harness"
SCHEMA_VERSION = 1
STAGE = "semantic-analyzer-preprovider"

REQUIRED_ENGINE_BASELINE_SHA = "1aa4acc7e0ccdb4cb769b8617667d6a505cc2671"
FAILED_HARNESS_CHECKPOINT_SHA = "5f325cb871ed9889a1058a510ca3983306b0c7dc"
EXPECTED_FIXTURE_SHA256 = "72c1bd4dd8a3d39acec01300dd1a9b03021634cfbc49433f0881ee2c2b796d1d"

CASE_ORDER = ("P6", "P7", "P1")
CASE_KEYS = {
    "P6": "P6-COMPLETE",
    "P7": "P7-COMPLETE",
    "P1": "P1-COMPLETE",
}
MAX_PROVIDER_REQUESTS = 3
MAX_ATTEMPTS_PER_CASE = 1

ARTIFACT_DIGEST_FIELDS = (
    "harness_id",
    "stage",
    "pack_id",
    "provider_calls",
    "identity_hashes",
    "identity_validation",
    "fixture_truth_source",
    "attempt_ledger",
    "case_order",
    "case_results",
    "pass_count",
    "fail_count",
    "invalid_count",
    "not_run_count",
    "overall_disposition",
    "provider_execution_authorized",
)

CaseDisposition = Literal["PASS", "FAIL", "INVALID", "NOT_RUN"]
OverallDisposition = Literal["PASS", "FAIL", "INVALID"]

FROZEN_ANALYZER_RESPONSE_CONTRACT = {
    "top_level_fields": ["observations"],
    "observation_fields": sorted(ALLOWED_EXTERNAL_OBSERVATION_FIELDS),
    "quote_fields": sorted(ALLOWED_QUOTE_FIELDS),
    "blocker_fields": sorted(ALLOWED_EXTERNAL_BLOCKER_FIELDS),
    "canonical_observation_fields": sorted(ALLOWED_OBSERVATION_FIELDS),
    "canonical_span_fields": sorted(ALLOWED_SPAN_FIELDS),
    "canonical_blocker_fields": sorted(ALLOWED_BLOCKER_FIELDS),
    "blocker_kinds": sorted(BLOCKER_KINDS),
    "forbidden_analyzer_fields": sorted(FORBIDDEN_ANALYZER_FIELDS),
}
ANALYZER_SCHEMA_SHA256 = sha256_utf8(
    json.dumps(FROZEN_ANALYZER_RESPONSE_CONTRACT, sort_keys=True, ensure_ascii=True)
)

FROZEN_ANALYZER_PROMPT_CONTRACT = (
    "Semantic analyzer supplies observations only: claim_id, support_quotes, "
    "full_entailment, blockers with evidence_quotes. Exact verbatim quotes only; "
    "Python binds quotes to canonical spans. Blocker kinds: contradiction, limitation. "
    "No offsets, final status, materiality, publication decision, or routing confidence."
)
ANALYZER_PROMPT_SHA256 = sha256_utf8(FROZEN_ANALYZER_PROMPT_CONTRACT)


class SemanticAnalyzerHarnessError(ValueError):
    """Pre-provider harness pack or qualification asset is not evaluable."""


class ResponseContractError(SemanticAnalyzerHarnessError, _ResponseContractError):
    """Analyzer response violates the frozen JSON contract."""


def parse_analyzer_response(raw: str) -> dict[str, Any]:
    """Harness-facing wrapper; production logic lives in agent.semantic_analyzer.response_contract."""
    try:
        return _parse_analyzer_response(raw)
    except _ResponseContractError as exc:
        raise ResponseContractError(str(exc)) from exc


class TrustedIngressError(SemanticAnalyzerHarnessError):
    """Draft/claim/evidence ingress validation failed."""


class AttemptGovernanceError(SemanticAnalyzerHarnessError):
    """Attempt ledger rejected an unauthorized or duplicate attempt."""


class ArtifactIntegrityError(SemanticAnalyzerHarnessError):
    """Stored qualification artifact failed integrity checks."""


class ExecutionAuthorizationError(SemanticAnalyzerHarnessError):
    """Provider execution authorization missing or invalid."""


@dataclass(frozen=True)
class SpanExpectation:
    evidence_id: str
    start: int
    end: int


@dataclass(frozen=True)
class BlockerExpectation:
    kind: str
    span: SpanExpectation


@dataclass(frozen=True)
class GoldOracleSpec:
    claim_id: str
    support: SpanExpectation
    full_entailment: bool
    blockers: tuple[BlockerExpectation, ...]
    expected_semantic_status: str
    require_nonempty_limitation_span: bool = False


GOLD_ORACLE: dict[str, GoldOracleSpec] = {
    "P6": GoldOracleSpec(
        claim_id="P6.claim.1",
        support=SpanExpectation("SRC-P6-W03", 400, 458),
        full_entailment=False,
        blockers=(BlockerExpectation("contradiction", SpanExpectation("SRC-P6-W03", 1500, 1562)),),
        expected_semantic_status="weak",
    ),
    "P7": GoldOracleSpec(
        claim_id="P7.claim.1",
        support=SpanExpectation("SRC-P7-K03", 0, 60),
        full_entailment=False,
        blockers=(BlockerExpectation("limitation", SpanExpectation("SRC-P7-K03", 2000, 2048)),),
        expected_semantic_status="weak",
        require_nonempty_limitation_span=True,
    ),
    "P1": GoldOracleSpec(
        claim_id="P1.claim.1",
        support=SpanExpectation("SRC-P1-W02", 1500, 1563),
        full_entailment=True,
        blockers=(),
        expected_semantic_status="verified",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticAnalyzerHarnessError(message)


def provider_execution_authorized() -> bool:
    return os.getenv("SEMANTIC_ANALYZER_EXECUTE", "").strip() == "1"


def get_implementation_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _git_is_ancestor(ancestor_sha: str, descendant_sha: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def load_harness_identity(path: Path | str = DEFAULT_IDENTITY) -> dict[str, Any]:
    identity_path = Path(path)
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    if payload.get("required_engine_baseline_sha") != REQUIRED_ENGINE_BASELINE_SHA:
        raise SemanticAnalyzerHarnessError("harness identity engine baseline mismatch")
    if payload.get("failed_harness_checkpoint_sha") != FAILED_HARNESS_CHECKPOINT_SHA:
        raise SemanticAnalyzerHarnessError("harness identity failed checkpoint mismatch")
    if payload.get("fixture_sha256") != EXPECTED_FIXTURE_SHA256:
        raise SemanticAnalyzerHarnessError("harness identity fixture hash mismatch")
    return payload


def validate_harness_identities(
    *,
    harness_implementation_sha: str | None = None,
) -> dict[str, Any]:
    """Separate harness implementation SHA from immutable engine baseline SHA."""
    identity = load_harness_identity()
    harness_sha = harness_implementation_sha or get_implementation_git_sha()
    invalid_reasons: list[str] = []

    if harness_sha == REQUIRED_ENGINE_BASELINE_SHA:
        invalid_reasons.append("harness_sha_equals_engine_baseline_only")
    if not HARNESS_MODULE.is_file():
        invalid_reasons.append("harness_module_missing")
    if not _git_is_ancestor(REQUIRED_ENGINE_BASELINE_SHA, harness_sha):
        invalid_reasons.append("engine_baseline_not_ancestor_of_harness")
    if not _git_is_ancestor(FAILED_HARNESS_CHECKPOINT_SHA, harness_sha):
        invalid_reasons.append("failed_harness_checkpoint_not_ancestor_of_harness")

    return {
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "harness_implementation_sha": harness_sha,
        "required_engine_baseline_sha": identity["required_engine_baseline_sha"],
        "failed_harness_checkpoint_sha": identity["failed_harness_checkpoint_sha"],
        "failed_harness_checkpoint_parent_sha": identity["failed_harness_checkpoint_parent_sha"],
    }


def sha256_text(text: str) -> str:
    return sha256_utf8(text)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_fixture_pack(path: Path | str = DEFAULT_FIXTURES) -> dict[str, Any]:
    fixture_path = Path(path)
    raw_bytes = fixture_path.read_bytes()
    actual_hash = sha256_bytes(raw_bytes)
    if actual_hash != EXPECTED_FIXTURE_SHA256:
        raise SemanticAnalyzerHarnessError(
            f"fixture hash mismatch: expected {EXPECTED_FIXTURE_SHA256}, got {actual_hash}"
        )
    pack = json.loads(raw_bytes.decode("utf-8"))
    validate_fixture_pack(pack)
    return pack


def load_verified_fixture_pack() -> dict[str, Any]:
    """Load qualification truth only from canonical fixture bytes on disk."""
    resolved = DEFAULT_FIXTURES.resolve()
    if not resolved.is_file():
        raise SemanticAnalyzerHarnessError("canonical qualification fixture missing")
    return load_fixture_pack(resolved)


def validate_caller_pack_against_verified(caller_pack: dict[str, Any]) -> dict[str, Any]:
    """Caller-supplied packs are untrusted working copies; never qualification truth."""
    verified_pack = load_verified_fixture_pack()
    caller_canonical = json.dumps(caller_pack, sort_keys=True, ensure_ascii=True)
    verified_canonical = json.dumps(verified_pack, sort_keys=True, ensure_ascii=True)
    matches = caller_canonical == verified_canonical
    return {
        "matches_verified_fixture": matches,
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "caller_is_authoritative": False,
    }


def get_verified_frozen_case_truth(case_id: str) -> dict[str, Any]:
    """Evaluator-owned frozen truth built only from verified canonical fixture bytes."""
    if case_id not in CASE_ORDER:
        raise SemanticAnalyzerHarnessError(f"unknown qualification case {case_id!r}")
    verified_pack = load_verified_fixture_pack()
    return build_frozen_case_truth(verified_pack, case_id)


def validate_fixture_pack(pack: dict[str, Any]) -> None:
    _require(pack.get("pack_id") == PACK_ID, "fixture pack_id mismatch")
    _require(pack.get("schema_version") == SCHEMA_VERSION, "fixture schema_version mismatch")
    cases = pack.get("cases")
    _require(isinstance(cases, dict), "fixture cases missing")
    for case_id in CASE_KEYS.values():
        _require(case_id in cases, f"missing fixture case {case_id}")
    for case in cases.values():
        _require(case["source_sha256"] == sha256_text(case["source_text"]), "stale source hash in fixture")
        obs = case["observation"]
        fixture_forbidden = {
            field
            for field in FROZEN_ANALYZER_RESPONSE_CONTRACT["forbidden_analyzer_fields"]
            if field not in {"support_spans", "evidence_spans", "start", "end"}
        }
        _require(
            fixture_forbidden.isdisjoint(obs.keys()),
            "fixture observation contains forbidden analyzer fields",
        )


def build_manifest_hash(evidence_manifest: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "evidence_id": entry["evidence_id"],
            "request_id": entry["request_id"],
            "source_sha256": entry["source_sha256"],
            "verifier_visible": entry.get("verifier_visible", True),
        }
        for entry in sorted(evidence_manifest, key=lambda row: row["evidence_id"])
    ]
    return sha256_text(json.dumps(canonical, sort_keys=True, ensure_ascii=True))


def build_frozen_case_truth(pack: dict[str, Any], case_id: str) -> dict[str, Any]:
    """Evaluator-owned immutable qualification input for one case."""
    bundle = build_case_bundle(pack, case_id)
    return {
        "case_id": bundle["case_id"],
        "case_key": bundle["case_key"],
        "request_id": bundle["request_id"],
        "schema_version": bundle["schema_version"],
        "draft_text": bundle["draft_text"],
        "draft_sha256": bundle["draft_sha256"],
        "claims": copy_claims_for_transport(bundle["claims"]),
        "evidence_manifest": copy_evidence_for_transport(bundle["evidence_manifest"]),
        "manifest_hash": bundle["manifest_hash"],
        "exposure_identity": dict(bundle["exposure_identity"]),
    }


def build_case_bundle(pack: dict[str, Any], case_id: str) -> dict[str, Any]:
    case_key = CASE_KEYS[case_id]
    case = pack["cases"][case_key]
    claim_span_end = len(case["draft_text"])
    claims = [
        {
            "claim_id": case["claim_id"],
            "claim_text": case["draft_text"],
            "claim_span": [0, claim_span_end],
        }
    ]
    evidence_manifest = [
        {
            "evidence_id": case["source_id"],
            "request_id": case["request_id"],
            "source_text": case["source_text"],
            "source_sha256": case["source_sha256"],
            "verifier_visible": True,
            "exposure_arm": "complete",
        }
    ]
    return {
        "case_id": case_id,
        "case_key": case_key,
        "request_id": case["request_id"],
        "schema_version": SCHEMA_VERSION,
        "draft_text": case["draft_text"],
        "draft_sha256": sha256_text(case["draft_text"]),
        "claims": claims,
        "evidence_manifest": evidence_manifest,
        "manifest_hash": build_manifest_hash(evidence_manifest),
        "exposure_identity": {
            "request_id": case["request_id"],
            "exposure_arm": "complete",
            "evidence_ids": [case["source_id"]],
        },
    }


def build_analyzer_input(bundle: dict[str, Any]) -> dict[str, Any]:
    """Future provider input — no observations, gold spans, or expected labels."""
    return {
        "request_id": bundle["request_id"],
        "schema_version": bundle["schema_version"],
        "draft_text": bundle["draft_text"],
        "draft_sha256": bundle["draft_sha256"],
        "claims": copy_claims_for_transport(bundle["claims"]),
        "evidence_manifest": copy_evidence_for_transport(bundle["evidence_manifest"]),
    }


def copy_claims_for_transport(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "claim_span": list(claim["claim_span"]),
        }
        for claim in claims
    ]


def copy_evidence_for_transport(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": entry["evidence_id"],
            "request_id": entry["request_id"],
            "source_text": entry["source_text"],
            "source_sha256": entry["source_sha256"],
            "verifier_visible": entry.get("verifier_visible", True),
            "exposure_arm": entry.get("exposure_arm", "complete"),
        }
        for entry in manifest
    ]


def validate_against_frozen_truth(
    bundle: dict[str, Any],
    frozen_truth: dict[str, Any],
) -> dict[str, Any]:
    """Pin decision-critical inputs to evaluator-owned fixture truth."""
    invalid_reasons: list[str] = []

    for truth_field in (
        "case_id",
        "request_id",
        "schema_version",
        "draft_text",
        "draft_sha256",
        "manifest_hash",
    ):
        if bundle.get(truth_field) != frozen_truth.get(truth_field):
            invalid_reasons.append(f"frozen_{truth_field}_mismatch")

    if copy_claims_for_transport(bundle.get("claims") or []) != frozen_truth["claims"]:
        invalid_reasons.append("frozen_claims_mismatch")

    if copy_evidence_for_transport(bundle.get("evidence_manifest") or []) != frozen_truth["evidence_manifest"]:
        invalid_reasons.append("frozen_evidence_manifest_mismatch")

    if dict(bundle.get("exposure_identity") or {}) != frozen_truth["exposure_identity"]:
        invalid_reasons.append("frozen_exposure_identity_mismatch")

    return {
        "valid": not invalid_reasons,
        "invalid_reasons": sorted(set(invalid_reasons)),
    }


def validate_trusted_ingress(
    bundle: dict[str, Any],
    *,
    frozen_truth: dict[str, Any] | None = None,
    caller_draft_sha256: str | None = None,
    caller_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Deterministic ingress validation before provider call and before accepting results."""
    invalid_reasons: list[str] = []

    if frozen_truth is not None:
        frozen_result = validate_against_frozen_truth(bundle, frozen_truth)
        invalid_reasons.extend(frozen_result["invalid_reasons"])
    draft_text = bundle.get("draft_text")
    if not isinstance(draft_text, str) or not draft_text:
        invalid_reasons.append("missing_draft_text")
        draft_text = ""

    recomputed_draft_hash = sha256_text(draft_text)
    declared_draft_hash = bundle.get("draft_sha256")
    if declared_draft_hash != recomputed_draft_hash:
        invalid_reasons.append("draft_sha256_mismatch")
    if caller_draft_sha256 is not None:
        if frozen_truth is not None and caller_draft_sha256 != frozen_truth["draft_sha256"]:
            invalid_reasons.append("caller_draft_sha256_overrides_frozen_truth")
        elif caller_draft_sha256 != recomputed_draft_hash:
            invalid_reasons.append("caller_draft_sha256_mismatch")

    claims = bundle.get("claims")
    if not isinstance(claims, list) or not claims:
        invalid_reasons.append("missing_claim_roster")
        claims = []

    claim_ids: list[str] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            invalid_reasons.append(f"claim_{index}_malformed")
            continue
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            invalid_reasons.append(f"claim_{index}_invalid_id")
            continue
        if claim_id in claim_ids:
            invalid_reasons.append("duplicate_claim_id")
        claim_ids.append(claim_id)
        claim_text = claim.get("claim_text")
        claim_span = claim.get("claim_span")
        if not isinstance(claim_text, str):
            invalid_reasons.append(f"claim_{claim_id}_invalid_text_type")
            continue
        if (
            not isinstance(claim_span, list)
            or len(claim_span) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in claim_span)
        ):
            invalid_reasons.append(f"claim_{claim_id}_invalid_span")
            continue
        start, end = claim_span
        if start < 0 or end <= start or end > len(draft_text):
            invalid_reasons.append(f"claim_{claim_id}_out_of_bounds_span")
            continue
        if draft_text[start:end] != claim_text:
            invalid_reasons.append(f"claim_{claim_id}_text_draft_slice_mismatch")

    manifest = bundle.get("evidence_manifest")
    if not isinstance(manifest, list) or not manifest:
        invalid_reasons.append("missing_evidence_manifest")
        manifest = []

    evidence_ids: list[str] = []
    request_id = bundle.get("request_id")
    for index, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            invalid_reasons.append(f"evidence_{index}_malformed")
            continue
        evidence_id = entry.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            invalid_reasons.append(f"evidence_{index}_invalid_id")
            continue
        if evidence_id in evidence_ids:
            invalid_reasons.append("duplicate_evidence_id")
        evidence_ids.append(evidence_id)
        source_text = entry.get("source_text")
        if not isinstance(source_text, str):
            invalid_reasons.append(f"evidence_{evidence_id}_invalid_source_text")
            continue
        declared_source_hash = entry.get("source_sha256")
        recomputed_source_hash = sha256_text(source_text)
        if declared_source_hash != recomputed_source_hash:
            invalid_reasons.append(f"evidence_{evidence_id}_source_sha256_mismatch")
        if entry.get("request_id") != request_id:
            invalid_reasons.append(f"evidence_{evidence_id}_request_ownership_mismatch")
        if not entry.get("verifier_visible", True):
            invalid_reasons.append(f"evidence_{evidence_id}_invisible")

    recomputed_manifest_hash = build_manifest_hash(manifest) if manifest else ""
    declared_manifest_hash = bundle.get("manifest_hash")
    if declared_manifest_hash and declared_manifest_hash != recomputed_manifest_hash:
        invalid_reasons.append("manifest_hash_mismatch")
    if caller_manifest_hash is not None:
        if frozen_truth is not None and caller_manifest_hash != frozen_truth["manifest_hash"]:
            invalid_reasons.append("caller_manifest_hash_overrides_frozen_truth")
        elif caller_manifest_hash != recomputed_manifest_hash:
            invalid_reasons.append("caller_manifest_hash_mismatch")

    exposure = bundle.get("exposure_identity") or {}
    if exposure.get("request_id") != request_id:
        invalid_reasons.append("exposure_request_identity_mismatch")
    expected_evidence_ids = sorted(exposure.get("evidence_ids") or [])
    if expected_evidence_ids and sorted(evidence_ids) != expected_evidence_ids:
        invalid_reasons.append("exposure_evidence_identity_mismatch")

    valid = not invalid_reasons
    return {
        "valid": valid,
        "invalid_reasons": sorted(set(invalid_reasons)),
        "recomputed_draft_sha256": recomputed_draft_hash,
        "recomputed_manifest_hash": recomputed_manifest_hash,
    }


def span_observation_to_quote_observation(
    *,
    source_text: str,
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Build a quote-based analyzer observation from canonical span gold (offline helpers only)."""
    support_quotes = [
        {
            "evidence_id": span["evidence_id"],
            "quote": extract_span_text(source_text, span["start"], span["end"]),
        }
        for span in observation["support_spans"]
    ]
    blockers = []
    for blocker in observation["blockers"]:
        blockers.append(
            {
                "kind": blocker["kind"],
                "explanation": blocker["explanation"],
                "evidence_quotes": [
                    {
                        "evidence_id": span["evidence_id"],
                        "quote": extract_span_text(source_text, span["start"], span["end"]),
                    }
                    for span in blocker["evidence_spans"]
                ],
            }
        )
    return {
        "claim_id": observation["claim_id"],
        "support_quotes": support_quotes,
        "full_entailment": observation["full_entailment"],
        "blockers": blockers,
    }


def _span_key(span: dict[str, Any]) -> tuple[str, int, int]:
    return (span["evidence_id"], span["start"], span["end"])


def _extract_span_text_from_bundle(
    bundle: dict[str, Any],
    span: dict[str, Any],
) -> str | None:
    evidence_by_id = {entry["evidence_id"]: entry for entry in bundle["evidence_manifest"]}
    entry = evidence_by_id.get(span["evidence_id"])
    if entry is None:
        return None
    return extract_span_text(entry["source_text"], span["start"], span["end"])


def evaluate_semantic_oracle(
    *,
    case_id: str,
    bundle: dict[str, Any],
    observations: list[dict[str, Any]],
    adjudication: dict[str, Any],
) -> dict[str, Any]:
    """Evaluator-only gold oracle — never included in analyzer input."""
    spec = GOLD_ORACLE[case_id]
    invalid_reasons: list[str] = []
    fail_reasons: list[str] = []

    matching = [row for row in observations if row.get("claim_id") == spec.claim_id]
    if len(matching) != 1:
        return {
            "valid": False,
            "disposition": "INVALID",
            "invalid_reasons": ["oracle_claim_row_count_mismatch"],
            "fail_reasons": [],
        }
    observation = matching[0]

    if observation["full_entailment"] is not spec.full_entailment:
        fail_reasons.append("full_entailment_mismatch")

    support_keys = {_span_key(span) for span in observation["support_spans"]}
    expected_support = (spec.support.evidence_id, spec.support.start, spec.support.end)
    if expected_support not in support_keys:
        fail_reasons.append("support_span_mismatch")
    if len(support_keys) != 1:
        fail_reasons.append("support_span_count_mismatch")

    blockers = observation["blockers"]
    if len(blockers) != len(spec.blockers):
        fail_reasons.append("blocker_count_mismatch")

    if spec.require_nonempty_limitation_span:
        for blocker in blockers:
            if blocker["kind"] == "limitation" and not blocker["evidence_spans"]:
                invalid_reasons.append("empty_limitation_evidence_span")

    for expected_blocker in spec.blockers:
        matched = [
            blocker
            for blocker in blockers
            if blocker["kind"] == expected_blocker.kind
            and {_span_key(span) for span in blocker["evidence_spans"]}
            == {(expected_blocker.span.evidence_id, expected_blocker.span.start, expected_blocker.span.end)}
        ]
        if not matched:
            fail_reasons.append(f"missing_required_{expected_blocker.kind}_blocker")

    expected_support_text = _extract_span_text_from_bundle(
        bundle,
        {"evidence_id": spec.support.evidence_id, "start": spec.support.start, "end": spec.support.end},
    )
    actual_support_texts = [
        _extract_span_text_from_bundle(bundle, span) for span in observation["support_spans"]
    ]
    if expected_support_text and any(text != expected_support_text for text in actual_support_texts if text):
        fail_reasons.append("support_text_mismatch")

    for expected_blocker in spec.blockers:
        expected_blocker_text = _extract_span_text_from_bundle(
            bundle,
            {
                "evidence_id": expected_blocker.span.evidence_id,
                "start": expected_blocker.span.start,
                "end": expected_blocker.span.end,
            },
        )
        for blocker in blockers:
            if blocker["kind"] != expected_blocker.kind:
                continue
            for span in blocker["evidence_spans"]:
                span_text = _extract_span_text_from_bundle(bundle, span)
                if span_text and span_text != expected_blocker_text:
                    fail_reasons.append(f"{expected_blocker.kind}_text_mismatch")

    claim_result = adjudication["semantic_status_by_claim"].get(spec.claim_id)
    if adjudication["validity"] != "VALID":
        invalid_reasons.append("adjudication_invalid")
    elif claim_result != spec.expected_semantic_status:
        fail_reasons.append("derived_semantic_status_mismatch")

    if invalid_reasons:
        disposition: CaseDisposition = "INVALID"
    elif fail_reasons:
        disposition = "FAIL"
    else:
        disposition = "PASS"

    extracted = {
        "support_span_texts": [
            {
                "span": _span_key(span),
                "text": _extract_span_text_from_bundle(bundle, span),
            }
            for span in observation["support_spans"]
        ],
        "blocker_span_texts": [
            {
                "kind": blocker["kind"],
                "spans": [
                    {
                        "span": _span_key(span),
                        "text": _extract_span_text_from_bundle(bundle, span),
                    }
                    for span in blocker["evidence_spans"]
                ],
            }
            for blocker in blockers
        ],
    }

    return {
        "valid": disposition == "PASS",
        "disposition": disposition,
        "invalid_reasons": sorted(set(invalid_reasons)),
        "fail_reasons": sorted(set(fail_reasons)),
        "expected_semantic_status": spec.expected_semantic_status,
        "derived_semantic_status": claim_result,
        "extracted_span_text": extracted,
    }


def build_envelope_from_bundle(
    bundle: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    return build_minimal_envelope(
        request_id=bundle["request_id"],
        schema_version=bundle["schema_version"],
        draft_text=bundle["draft_text"],
        claims=bundle["claims"],
        evidence_manifest=bundle["evidence_manifest"],
        observations=observations,
    )


def adjudicate_with_validated_boundary(envelope: dict[str, Any]) -> dict[str, Any]:
    """Validated hybrid adjudication path — not raw derive_claim_status()."""
    result = adjudicate_hybrid_verifier_observations(envelope)
    return {
        "request_id": result.request_id,
        "validity": result.validity,
        "invalid_reasons": result.invalid_reasons,
        "claim_results": [
            {
                "claim_id": row.claim_id,
                "validity": row.validity,
                "semantic_status": row.semantic_status,
                "invalid_reasons": row.invalid_reasons,
                "consistency_diagnostics": row.consistency_diagnostics,
                "reason_codes": row.reason_codes,
            }
            for row in result.claim_results
        ],
        "semantic_status_by_claim": result.semantic_status_by_claim,
    }


@dataclass
class AttemptRecord:
    case_id: str
    attempt_index: int
    authorized: bool
    recorded_at: str
    disposition: CaseDisposition | None = None


@dataclass
class AttemptLedger:
    run_id: str
    records: list[AttemptRecord] = field(default_factory=list)
    stopped: bool = False
    stop_reason: str | None = None

    @property
    def attempt_count(self) -> int:
        return len(self.records)

    def authorize_attempt(self, case_id: str) -> None:
        if self.stopped:
            raise AttemptGovernanceError(f"run_stopped:{self.stop_reason}")
        if case_id not in CASE_ORDER:
            raise AttemptGovernanceError(f"unknown_case:{case_id}")
        expected_index = len(self.records)
        if expected_index >= MAX_PROVIDER_REQUESTS:
            raise AttemptGovernanceError("fourth_attempt_denied")
        if any(record.case_id == case_id for record in self.records):
            raise AttemptGovernanceError(f"duplicate_retry:{case_id}")
        expected_case = CASE_ORDER[expected_index]
        if case_id != expected_case:
            raise AttemptGovernanceError(
                f"out_of_order_attempt:expected_{expected_case}_got_{case_id}"
            )

    def record_attempt(
        self,
        *,
        case_id: str,
        disposition: CaseDisposition,
    ) -> AttemptRecord:
        self.authorize_attempt(case_id)
        record = AttemptRecord(
            case_id=case_id,
            attempt_index=len(self.records) + 1,
            authorized=True,
            recorded_at=datetime.now(UTC).isoformat(),
            disposition=disposition,
        )
        self.records.append(record)
        if disposition in {"FAIL", "INVALID"}:
            self.stopped = True
            self.stop_reason = disposition.lower()
        return record

    def remaining_cases(self) -> list[str]:
        attempted = {record.case_id for record in self.records}
        remaining: list[str] = []
        for case_id in CASE_ORDER:
            if case_id not in attempted:
                remaining.append(case_id)
        return remaining

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "attempt_count": self.attempt_count,
            "max_provider_requests": MAX_PROVIDER_REQUESTS,
            "case_order": list(CASE_ORDER),
            "records": [
                {
                    "case_id": record.case_id,
                    "attempt_index": record.attempt_index,
                    "authorized": record.authorized,
                    "recorded_at": record.recorded_at,
                    "disposition": record.disposition,
                }
                for record in self.records
            ],
        }

    def derive_state_from_history(self) -> None:
        self.stopped = False
        self.stop_reason = None
        for index, record in enumerate(self.records):
            if record.case_id != CASE_ORDER[index]:
                raise AttemptGovernanceError(
                    f"ledger_order_violation:expected_{CASE_ORDER[index]}_got_{record.case_id}"
                )
            if record.attempt_index != index + 1:
                raise AttemptGovernanceError("ledger_attempt_index_violation")
            if record.disposition in {"FAIL", "INVALID"}:
                self.stopped = True
                self.stop_reason = record.disposition.lower()
                if index + 1 < len(self.records):
                    raise AttemptGovernanceError("ledger_attempts_after_terminal_disposition")
                break

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AttemptLedger:
        records: list[AttemptRecord] = []
        seen_cases: set[str] = set()
        for row in payload.get("records", []):
            case_id = row["case_id"]
            if case_id in seen_cases:
                raise AttemptGovernanceError(f"ledger_duplicate_case:{case_id}")
            seen_cases.add(case_id)
            records.append(
                AttemptRecord(
                    case_id=case_id,
                    attempt_index=row["attempt_index"],
                    authorized=row.get("authorized", True),
                    recorded_at=row.get("recorded_at", ""),
                    disposition=row.get("disposition"),
                )
            )
        if len(records) > MAX_PROVIDER_REQUESTS:
            raise AttemptGovernanceError("ledger_exceeds_max_attempts")

        ledger = cls(run_id=payload["run_id"])
        ledger.records = records
        ledger.derive_state_from_history()

        if payload.get("stopped") is False and ledger.stopped:
            raise AttemptGovernanceError("ledger_stopped_flag_contradicts_history")
        if payload.get("stop_reason") not in (None, ledger.stop_reason) and ledger.stopped:
            raise AttemptGovernanceError("ledger_stop_reason_contradicts_history")
        return ledger

    def reject_replay_attempt(self, case_id: str) -> None:
        if any(record.case_id == case_id for record in self.records):
            raise AttemptGovernanceError(f"replay_retry_denied:{case_id}")


def validate_attempt_ledger(payload: dict[str, Any]) -> dict[str, Any]:
    invalid_reasons: list[str] = []
    try:
        AttemptLedger.from_dict(payload)
    except AttemptGovernanceError as exc:
        invalid_reasons.append(str(exc))
    return {"valid": not invalid_reasons, "invalid_reasons": invalid_reasons}


def qualify_case_response(
    *,
    case_id: str,
    bundle: dict[str, Any],
    raw_response: str,
    frozen_truth: dict[str, Any],
    ingress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Qualify one analyzer response against trusted ingress, contract, oracle, and adjudication."""
    case_result: dict[str, Any] = {
        "case_id": case_id,
        "request_id": bundle.get("request_id"),
    }

    if frozen_truth is None:
        case_result.update(
            {
                "disposition": "INVALID",
                "invalid_reasons": ["missing_frozen_truth"],
                "response_contract": None,
                "adjudication": None,
                "semantic_oracle": None,
            }
        )
        return case_result

    ingress_result = ingress or validate_trusted_ingress(bundle, frozen_truth=frozen_truth)
    case_result["trusted_ingress"] = ingress_result
    if not ingress_result["valid"]:
        case_result.update(
            {
                "disposition": "INVALID",
                "invalid_reasons": ingress_result["invalid_reasons"],
                "response_contract": None,
                "adjudication": None,
                "semantic_oracle": None,
            }
        )
        return case_result

    try:
        quote_response_contract = parse_analyzer_response(raw_response)
    except ResponseContractError as exc:
        case_result.update(
            {
                "disposition": "INVALID",
                "invalid_reasons": [str(exc)],
                "quote_response_contract": None,
                "response_contract": None,
                "adjudication": None,
                "semantic_oracle": None,
            }
        )
        return case_result

    case_result["quote_response_contract"] = quote_response_contract
    try:
        canonical_observations = convert_quote_observations_to_canonical(
            quote_response_contract["observations"],
            bundle["evidence_manifest"],
        )
    except QuoteBindingError as exc:
        case_result.update(
            {
                "disposition": "INVALID",
                "invalid_reasons": [f"quote_binding:{exc.reason}"],
                "response_contract": None,
                "adjudication": None,
                "semantic_oracle": None,
            }
        )
        return case_result

    response_contract = {"observations": canonical_observations}
    case_result["response_contract"] = response_contract
    envelope = build_envelope_from_bundle(bundle, response_contract["observations"])
    post_ingress = validate_trusted_ingress(
        {
            **bundle,
            "observations": response_contract["observations"],
        },
        frozen_truth=frozen_truth,
    )
    case_result["post_response_ingress"] = post_ingress
    if not post_ingress["valid"]:
        case_result.update(
            {
                "disposition": "INVALID",
                "invalid_reasons": post_ingress["invalid_reasons"],
                "adjudication": None,
                "semantic_oracle": None,
            }
        )
        return case_result

    adjudication = adjudicate_with_validated_boundary(envelope)
    case_result["adjudication"] = adjudication
    if adjudication["validity"] == "INVALID":
        case_result.update(
            {
                "disposition": "INVALID",
                "invalid_reasons": adjudication["invalid_reasons"],
                "semantic_oracle": None,
            }
        )
        return case_result

    oracle = evaluate_semantic_oracle(
        case_id=case_id,
        bundle=bundle,
        observations=response_contract["observations"],
        adjudication=adjudication,
    )
    case_result["semantic_oracle"] = oracle
    case_result["disposition"] = oracle["disposition"]
    case_result["invalid_reasons"] = oracle.get("invalid_reasons", [])
    case_result["fail_reasons"] = oracle.get("fail_reasons", [])
    return case_result


def build_run_identity_hashes(
    *,
    harness_implementation_sha: str,
    fixture_sha256: str = EXPECTED_FIXTURE_SHA256,
) -> dict[str, str]:
    identity = load_harness_identity()
    return {
        "harness_implementation_sha": harness_implementation_sha,
        "required_engine_baseline_sha": identity["required_engine_baseline_sha"],
        "failed_harness_checkpoint_sha": identity["failed_harness_checkpoint_sha"],
        "failed_harness_checkpoint_parent_sha": identity["failed_harness_checkpoint_parent_sha"],
        "fixture_sha256": fixture_sha256,
        "schema_sha256": ANALYZER_SCHEMA_SHA256,
        "prompt_sha256": ANALYZER_PROMPT_SHA256,
    }


def build_artifact_digest_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    """Decision-critical artifact fields covered by the digest (excludes digest/integrity/readiness)."""
    return {field: artifact[field] for field in ARTIFACT_DIGEST_FIELDS if field in artifact}


def compute_artifact_digest(artifact: dict[str, Any]) -> str:
    payload = build_artifact_digest_payload(artifact)
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=True))


def verify_artifact_integrity(artifact: dict[str, Any]) -> dict[str, Any]:
    """Reject stale or corrupted PASS artifacts."""
    invalid_reasons: list[str] = []
    identity = artifact.get("identity_hashes") or {}
    current_sha = get_implementation_git_sha()
    identity_validation = validate_harness_identities(harness_implementation_sha=current_sha)
    if not identity_validation["valid"]:
        invalid_reasons.extend(
            f"identity_{reason}" for reason in identity_validation["invalid_reasons"]
        )

    for hash_field, expected in (
        ("harness_implementation_sha", current_sha),
        ("required_engine_baseline_sha", REQUIRED_ENGINE_BASELINE_SHA),
        ("failed_harness_checkpoint_sha", FAILED_HARNESS_CHECKPOINT_SHA),
        ("fixture_sha256", EXPECTED_FIXTURE_SHA256),
        ("schema_sha256", ANALYZER_SCHEMA_SHA256),
        ("prompt_sha256", ANALYZER_PROMPT_SHA256),
    ):
        actual = identity.get(hash_field)
        if actual != expected:
            invalid_reasons.append(f"stale_{hash_field}")

    stored_digest = artifact.get("artifact_digest")
    recomputed = compute_artifact_digest(artifact)
    if not isinstance(stored_digest, str) or not stored_digest.strip():
        invalid_reasons.append("missing_artifact_digest")
    elif stored_digest != recomputed:
        invalid_reasons.append("corrupt_artifact_digest")

    if artifact.get("overall_disposition") == "PASS":
        case_results = artifact.get("case_results") or []
        if len(case_results) != 3:
            invalid_reasons.append("incomplete_pass_case_count")
        if any(row.get("disposition") != "PASS" for row in case_results):
            invalid_reasons.append("pass_with_non_pass_case")
        if artifact.get("invalid_case_count", 0) != 0:
            invalid_reasons.append("pass_with_invalid_cases")

    ledger_payload = artifact.get("attempt_ledger")
    if isinstance(ledger_payload, dict):
        ledger_validation = validate_attempt_ledger(ledger_payload)
        if not ledger_validation["valid"]:
            invalid_reasons.extend(
                f"ledger_{reason}" for reason in ledger_validation["invalid_reasons"]
            )

    return {
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "recomputed_artifact_digest": recomputed,
    }


def compute_qualification_ready(artifact: dict[str, Any]) -> bool:
    integrity = artifact.get("artifact_integrity") or {}
    identity = artifact.get("identity_validation") or {}
    ledger = artifact.get("attempt_ledger_validation") or {}
    digest = artifact.get("artifact_digest")
    digest_present = isinstance(digest, str) and bool(digest.strip())
    return (
        artifact.get("overall_disposition") == "PASS"
        and artifact.get("pass_count") == 3
        and artifact.get("invalid_count") == 0
        and artifact.get("fail_count") == 0
        and digest_present
        and integrity.get("valid") is True
        and identity.get("valid") is True
        and ledger.get("valid") is True
    )


def run_offline_qualification(
    pack: dict[str, Any] | None = None,
    *,
    response_provider: dict[str, str] | None = None,
    run_id: str = "offline-mock-run",
    harness_implementation_sha: str | None = None,
) -> dict[str, Any]:
    """Execute the frozen P6→P7→P1 mock qualification run. No provider calls."""
    if provider_execution_authorized():
        raise ExecutionAuthorizationError(
            "provider execution flag set; offline harness requires mock-only qualification"
        )

    verified_pack = load_verified_fixture_pack()
    caller_pack_validation = (
        validate_caller_pack_against_verified(pack) if pack is not None else None
    )

    harness_implementation_sha = harness_implementation_sha or get_implementation_git_sha()
    identity_validation = validate_harness_identities(
        harness_implementation_sha=harness_implementation_sha
    )
    if not identity_validation["valid"]:
        raise SemanticAnalyzerHarnessError(
            "harness identity validation failed: "
            + ", ".join(identity_validation["invalid_reasons"])
        )

    ledger = AttemptLedger(run_id=run_id)
    case_results: list[dict[str, Any]] = []
    response_provider = response_provider or {}

    for case_id in CASE_ORDER:
        if ledger.stopped:
            break
        bundle = build_case_bundle(verified_pack, case_id)
        frozen_truth = get_verified_frozen_case_truth(case_id)
        analyzer_input = build_analyzer_input(bundle)
        _require(
            "observation" not in json.dumps(analyzer_input),
            "analyzer input leaked gold observations",
        )
        _require(
            "expected_status" not in json.dumps(analyzer_input),
            "analyzer input leaked expected labels",
        )

        if case_id not in response_provider:
            ledger.record_attempt(case_id=case_id, disposition="INVALID")
            case_results.append(
                {
                    "case_id": case_id,
                    "disposition": "INVALID",
                    "invalid_reasons": ["missing_mock_response"],
                }
            )
            break

        ledger.authorize_attempt(case_id)
        raw_response = response_provider[case_id]
        case_result = qualify_case_response(
            case_id=case_id,
            bundle=bundle,
            raw_response=raw_response,
            frozen_truth=frozen_truth,
        )
        ledger.record_attempt(case_id=case_id, disposition=case_result["disposition"])
        case_result["analyzer_input"] = analyzer_input
        case_result["raw_response"] = raw_response
        case_results.append(case_result)
        if case_result["disposition"] in {"FAIL", "INVALID"}:
            break

    attempted = {row["case_id"] for row in case_results}
    for case_id in CASE_ORDER:
        if case_id not in attempted:
            case_results.append({"case_id": case_id, "disposition": "NOT_RUN"})

    pass_count = sum(1 for row in case_results if row.get("disposition") == "PASS")
    invalid_count = sum(1 for row in case_results if row.get("disposition") == "INVALID")
    fail_count = sum(1 for row in case_results if row.get("disposition") == "FAIL")
    not_run_count = sum(1 for row in case_results if row.get("disposition") == "NOT_RUN")

    if invalid_count > 0 or fail_count > 0 or pass_count != 3:
        overall: OverallDisposition = "INVALID" if invalid_count > 0 else "FAIL"
    else:
        overall = "PASS"

    artifact: dict[str, Any] = {
        "harness_id": HARNESS_ID,
        "stage": STAGE,
        "pack_id": PACK_ID,
        "provider_calls": 0,
        "identity_hashes": build_run_identity_hashes(
            harness_implementation_sha=harness_implementation_sha
        ),
        "identity_validation": identity_validation,
        "fixture_truth_source": {
            "path": str(DEFAULT_FIXTURES.relative_to(REPO_ROOT)),
            "fixture_sha256": EXPECTED_FIXTURE_SHA256,
            "caller_pack_matches_verified": (
                caller_pack_validation["matches_verified_fixture"]
                if caller_pack_validation is not None
                else None
            ),
            "caller_pack_authoritative": False,
        },
        "attempt_ledger": ledger.to_dict(),
        "case_order": list(CASE_ORDER),
        "case_results": case_results,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "invalid_count": invalid_count,
        "not_run_count": not_run_count,
        "overall_disposition": overall,
        "provider_execution_authorized": False,
    }
    artifact["artifact_digest"] = compute_artifact_digest(artifact)
    integrity = verify_artifact_integrity(artifact)
    artifact["artifact_integrity"] = integrity
    ledger_validation = validate_attempt_ledger(artifact["attempt_ledger"])
    artifact["attempt_ledger_validation"] = ledger_validation
    artifact["qualification_ready"] = compute_qualification_ready(artifact)
    return artifact


def build_gold_mock_responses(pack: dict[str, Any] | None = None) -> dict[str, str]:
    verified_pack = load_verified_fixture_pack()
    if pack is not None:
        validation = validate_caller_pack_against_verified(pack)
        if not validation["matches_verified_fixture"]:
            raise SemanticAnalyzerHarnessError("caller pack does not match verified fixture")
    responses: dict[str, str] = {}
    for case_id in CASE_ORDER:
        case = verified_pack["cases"][CASE_KEYS[case_id]]
        quote_observation = span_observation_to_quote_observation(
            source_text=case["source_text"],
            observation=case["observation"],
        )
        responses[case_id] = json.dumps({"observations": [quote_observation]})
    return responses


def run_preflight(pack: dict[str, Any] | None = None) -> dict[str, Any]:
    pack = pack or load_fixture_pack()
    harness_sha = get_implementation_git_sha()
    identity_validation = validate_harness_identities(harness_implementation_sha=harness_sha)
    return {
        "harness_id": HARNESS_ID,
        "stage": STAGE,
        "pack_id": PACK_ID,
        "case_order": list(CASE_ORDER),
        "identity_hashes": build_run_identity_hashes(harness_implementation_sha=harness_sha),
        "identity_validation": identity_validation,
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "provider_execution_default_disabled": not provider_execution_authorized(),
        "max_provider_requests": MAX_PROVIDER_REQUESTS,
        "max_attempts_per_case": MAX_ATTEMPTS_PER_CASE,
        "analyzer_response_contract": FROZEN_ANALYZER_RESPONSE_CONTRACT,
        "trusted_ingress_controls": [
            "frozen_fixture_truth_pinning",
            "draft_sha256_recomputation",
            "claim_text_draft_slice_binding",
            "claim_span_bounds",
            "unique_claim_ids",
            "evidence_manifest_exactness",
            "unique_evidence_ids",
            "source_sha256_recomputation",
            "request_ownership",
            "exposure_identity",
            "evidence_visibility",
            "manifest_hash_recomputation",
        ],
        "invalid_dominance_path": "adjudicate_hybrid_verifier_observations",
        "semantic_oracle_cases": list(GOLD_ORACLE.keys()),
        "preflight_ready": identity_validation["valid"],
    }
