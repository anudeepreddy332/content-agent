"""Offline semantic-analyzer pre-provider qualification harness. No provider calls by default."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hybrid_verifier_status_engine import (  # noqa: E402
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

PACK_ID = "hybrid_verifier_status_offline"
HARNESS_ID = "semantic_analyzer_preprovider_harness"
SCHEMA_VERSION = 1
STAGE = "semantic-analyzer-preprovider"

REQUIRED_ENGINE_BASELINE_SHA = "1aa4acc7e0ccdb4cb769b8617667d6a505cc2671"
PARENT_IMPLEMENTATION_SHA = "b4074fdedff502d6d10e40653943a04ed999dcb8"
EXPECTED_FIXTURE_SHA256 = "72c1bd4dd8a3d39acec01300dd1a9b03021634cfbc49433f0881ee2c2b796d1d"

CASE_ORDER = ("P6", "P7", "P1")
CASE_KEYS = {
    "P6": "P6-COMPLETE",
    "P7": "P7-COMPLETE",
    "P1": "P1-COMPLETE",
}
MAX_PROVIDER_REQUESTS = 3
MAX_ATTEMPTS_PER_CASE = 1

CaseDisposition = Literal["PASS", "FAIL", "INVALID", "NOT_RUN"]
OverallDisposition = Literal["PASS", "FAIL", "INVALID"]

FROZEN_ANALYZER_RESPONSE_CONTRACT = {
    "top_level_fields": ["observations"],
    "observation_fields": sorted(ALLOWED_OBSERVATION_FIELDS),
    "span_fields": sorted(ALLOWED_SPAN_FIELDS),
    "blocker_fields": sorted(ALLOWED_BLOCKER_FIELDS),
    "blocker_kinds": sorted(BLOCKER_KINDS),
    "forbidden_analyzer_fields": [
        "status",
        "materiality",
        "confidence",
        "publication_decision",
        "routing",
    ],
}
ANALYZER_SCHEMA_SHA256 = sha256_utf8(
    json.dumps(FROZEN_ANALYZER_RESPONSE_CONTRACT, sort_keys=True, ensure_ascii=True)
)

FROZEN_ANALYZER_PROMPT_CONTRACT = (
    "Semantic analyzer supplies observations only: claim_id, support_spans, "
    "full_entailment, blockers. Blocker kinds: contradiction, limitation. "
    "No final status, materiality, publication decision, or routing confidence."
)
ANALYZER_PROMPT_SHA256 = sha256_utf8(FROZEN_ANALYZER_PROMPT_CONTRACT)


class SemanticAnalyzerHarnessError(ValueError):
    """Pre-provider harness pack or qualification asset is not evaluable."""


class ResponseContractError(SemanticAnalyzerHarnessError):
    """Analyzer response violates the frozen JSON contract."""


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
        forbidden = set(FROZEN_ANALYZER_RESPONSE_CONTRACT["forbidden_analyzer_fields"])
        _require(forbidden.isdisjoint(obs.keys()), "fixture observation contains forbidden analyzer fields")


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


def validate_trusted_ingress(
    bundle: dict[str, Any],
    *,
    caller_draft_sha256: str | None = None,
    caller_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Deterministic ingress validation before provider call and before accepting results."""
    invalid_reasons: list[str] = []
    draft_text = bundle.get("draft_text")
    if not isinstance(draft_text, str) or not draft_text:
        invalid_reasons.append("missing_draft_text")
        draft_text = ""

    recomputed_draft_hash = sha256_text(draft_text)
    declared_draft_hash = bundle.get("draft_sha256")
    if declared_draft_hash != recomputed_draft_hash:
        invalid_reasons.append("draft_sha256_mismatch")
    if caller_draft_sha256 is not None and caller_draft_sha256 != recomputed_draft_hash:
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
    if caller_manifest_hash is not None and caller_manifest_hash != recomputed_manifest_hash:
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


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*)\n?```\s*$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _json_load_no_duplicate_keys(raw: str) -> Any:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ResponseContractError("duplicate_json_keys")
        return dict(pairs)

    return json.loads(raw, object_pairs_hook=hook)


def parse_analyzer_response(raw: str) -> dict[str, Any]:
    """Parse and structurally validate the frozen analyzer response contract."""
    if not isinstance(raw, str) or not raw.strip():
        raise ResponseContractError("empty_response")

    text = _strip_markdown_fences(raw)
    if "```" in raw and text == raw.strip():
        raise ResponseContractError("markdown_fence_parse_failed")

    try:
        parsed = _json_load_no_duplicate_keys(text)
    except json.JSONDecodeError as exc:
        if exc.msg == "Extra data":
            raise ResponseContractError("trailing_prose") from exc
        raise ResponseContractError(f"malformed_json:{exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ResponseContractError("top_level_not_object")

    extra_top = set(parsed) - {"observations"}
    if extra_top:
        raise ResponseContractError("unknown_top_level_field")
    if "observations" not in parsed:
        raise ResponseContractError("missing_observations")

    observations = parsed["observations"]
    if not isinstance(observations, list):
        raise ResponseContractError("observations_not_array")

    validated_rows: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        prefix = f"observation_{index}"
        if observation is None:
            raise ResponseContractError(f"{prefix}_null")
        if not isinstance(observation, dict):
            raise ResponseContractError(f"{prefix}_not_object")
        extra = set(observation) - ALLOWED_OBSERVATION_FIELDS
        if extra:
            raise ResponseContractError(f"{prefix}_unknown_field")
        for required in ALLOWED_OBSERVATION_FIELDS:
            if required not in observation:
                raise ResponseContractError(f"{prefix}_missing_{required}")
            if observation[required] is None:
                raise ResponseContractError(f"{prefix}_null_{required}")

        claim_id = observation["claim_id"]
        if not isinstance(claim_id, str) or not claim_id:
            raise ResponseContractError(f"{prefix}_invalid_claim_id")

        full_entailment = observation["full_entailment"]
        if not isinstance(full_entailment, bool):
            raise ResponseContractError(f"{prefix}_invalid_full_entailment_type")

        support_spans = observation["support_spans"]
        blockers = observation["blockers"]
        if not isinstance(support_spans, list):
            raise ResponseContractError(f"{prefix}_malformed_support_spans")
        if not isinstance(blockers, list):
            raise ResponseContractError(f"{prefix}_malformed_blockers")

        for span_index, span in enumerate(support_spans):
            span_prefix = f"{prefix}_support_span_{span_index}"
            _validate_response_span(span, prefix=span_prefix)

        for blocker_index, blocker in enumerate(blockers):
            blocker_prefix = f"{prefix}_blocker_{blocker_index}"
            if not isinstance(blocker, dict):
                raise ResponseContractError(f"{blocker_prefix}_not_object")
            extra_blocker = set(blocker) - ALLOWED_BLOCKER_FIELDS
            if extra_blocker:
                raise ResponseContractError(f"{blocker_prefix}_unknown_field")
            for required in ALLOWED_BLOCKER_FIELDS:
                if required not in blocker:
                    raise ResponseContractError(f"{blocker_prefix}_missing_{required}")
                if blocker[required] is None:
                    raise ResponseContractError(f"{blocker_prefix}_null_{required}")
            kind = blocker["kind"]
            if kind not in BLOCKER_KINDS:
                raise ResponseContractError(f"{blocker_prefix}_unknown_kind")
            if not isinstance(blocker["explanation"], str) or not blocker["explanation"].strip():
                raise ResponseContractError(f"{blocker_prefix}_missing_explanation")
            evidence_spans = blocker["evidence_spans"]
            if not isinstance(evidence_spans, list):
                raise ResponseContractError(f"{blocker_prefix}_malformed_evidence_spans")
            for span_index, span in enumerate(evidence_spans):
                _validate_response_span(
                    span,
                    prefix=f"{blocker_prefix}_span_{span_index}",
                )

        validated_rows.append(observation)

    return {"observations": validated_rows}


def _validate_response_span(span: Any, *, prefix: str) -> None:
    if not isinstance(span, dict):
        raise ResponseContractError(f"{prefix}_not_object")
    extra = set(span) - ALLOWED_SPAN_FIELDS
    if extra:
        raise ResponseContractError(f"{prefix}_unknown_field")
    for required in ALLOWED_SPAN_FIELDS:
        if required not in span:
            raise ResponseContractError(f"{prefix}_missing_{required}")
        if span[required] is None:
            raise ResponseContractError(f"{prefix}_null_{required}")
    evidence_id = span["evidence_id"]
    start = span["start"]
    end = span["end"]
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ResponseContractError(f"{prefix}_invalid_evidence_id")
    if not isinstance(start, int) or isinstance(start, bool):
        raise ResponseContractError(f"{prefix}_invalid_start_type")
    if not isinstance(end, int) or isinstance(end, bool):
        raise ResponseContractError(f"{prefix}_invalid_end_type")


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AttemptLedger:
        ledger = cls(run_id=payload["run_id"], stopped=payload.get("stopped", False))
        ledger.stop_reason = payload.get("stop_reason")
        for row in payload.get("records", []):
            ledger.records.append(
                AttemptRecord(
                    case_id=row["case_id"],
                    attempt_index=row["attempt_index"],
                    authorized=row.get("authorized", True),
                    recorded_at=row.get("recorded_at", ""),
                    disposition=row.get("disposition"),
                )
            )
        return ledger

    def reject_replay_attempt(self, case_id: str) -> None:
        if any(record.case_id == case_id for record in self.records):
            raise AttemptGovernanceError(f"replay_retry_denied:{case_id}")


def qualify_case_response(
    *,
    case_id: str,
    bundle: dict[str, Any],
    raw_response: str,
    ingress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Qualify one analyzer response against trusted ingress, contract, oracle, and adjudication."""
    case_result: dict[str, Any] = {
        "case_id": case_id,
        "request_id": bundle["request_id"],
    }

    ingress_result = ingress or validate_trusted_ingress(bundle)
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
        response_contract = parse_analyzer_response(raw_response)
    except ResponseContractError as exc:
        case_result.update(
            {
                "disposition": "INVALID",
                "invalid_reasons": [str(exc)],
                "response_contract": None,
                "adjudication": None,
                "semantic_oracle": None,
            }
        )
        return case_result

    case_result["response_contract"] = response_contract
    envelope = build_envelope_from_bundle(bundle, response_contract["observations"])
    post_ingress = validate_trusted_ingress(
        {
            **bundle,
            "observations": response_contract["observations"],
        }
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
    implementation_sha: str,
    fixture_sha256: str = EXPECTED_FIXTURE_SHA256,
) -> dict[str, str]:
    return {
        "implementation_sha": implementation_sha,
        "parent_implementation_sha": PARENT_IMPLEMENTATION_SHA,
        "required_engine_baseline_sha": REQUIRED_ENGINE_BASELINE_SHA,
        "fixture_sha256": fixture_sha256,
        "schema_sha256": ANALYZER_SCHEMA_SHA256,
        "prompt_sha256": ANALYZER_PROMPT_SHA256,
    }


def compute_artifact_digest(payload: dict[str, Any]) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=True))


def verify_artifact_integrity(artifact: dict[str, Any]) -> dict[str, Any]:
    """Reject stale or corrupted PASS artifacts."""
    invalid_reasons: list[str] = []
    identity = artifact.get("identity_hashes") or {}
    current_sha = get_implementation_git_sha()

    for hash_field, expected in (
        ("implementation_sha", current_sha),
        ("required_engine_baseline_sha", REQUIRED_ENGINE_BASELINE_SHA),
        ("fixture_sha256", EXPECTED_FIXTURE_SHA256),
        ("schema_sha256", ANALYZER_SCHEMA_SHA256),
        ("prompt_sha256", ANALYZER_PROMPT_SHA256),
    ):
        actual = identity.get(hash_field)
        if actual != expected:
            invalid_reasons.append(f"stale_{hash_field}")

    stored_digest = artifact.get("artifact_digest")
    payload = {key: value for key, value in artifact.items() if key != "artifact_digest"}
    recomputed = compute_artifact_digest(payload)
    if stored_digest and stored_digest != recomputed:
        invalid_reasons.append("corrupt_artifact_digest")

    if artifact.get("overall_disposition") == "PASS":
        case_results = artifact.get("case_results") or []
        if len(case_results) != 3:
            invalid_reasons.append("incomplete_pass_case_count")
        if any(row.get("disposition") != "PASS" for row in case_results):
            invalid_reasons.append("pass_with_non_pass_case")
        if artifact.get("invalid_case_count", 0) != 0:
            invalid_reasons.append("pass_with_invalid_cases")

    return {
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "recomputed_artifact_digest": recomputed,
    }


def run_offline_qualification(
    pack: dict[str, Any],
    *,
    response_provider: dict[str, str] | None = None,
    run_id: str = "offline-mock-run",
    implementation_sha: str | None = None,
) -> dict[str, Any]:
    """Execute the frozen P6→P7→P1 mock qualification run. No provider calls."""
    if provider_execution_authorized():
        raise ExecutionAuthorizationError(
            "provider execution flag set; offline harness requires mock-only qualification"
        )

    implementation_sha = implementation_sha or get_implementation_git_sha()
    if implementation_sha != REQUIRED_ENGINE_BASELINE_SHA:
        raise SemanticAnalyzerHarnessError(
            f"implementation SHA {implementation_sha} != required baseline {REQUIRED_ENGINE_BASELINE_SHA}"
        )

    ledger = AttemptLedger(run_id=run_id)
    case_results: list[dict[str, Any]] = []
    response_provider = response_provider or {}

    for case_id in CASE_ORDER:
        if ledger.stopped:
            break
        bundle = build_case_bundle(pack, case_id)
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
        "identity_hashes": build_run_identity_hashes(implementation_sha=implementation_sha),
        "attempt_ledger": ledger.to_dict(),
        "case_order": list(CASE_ORDER),
        "case_results": case_results,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "invalid_count": invalid_count,
        "not_run_count": not_run_count,
        "overall_disposition": overall,
        "qualification_ready": overall == "PASS" and invalid_count == 0 and pass_count == 3,
        "provider_execution_authorized": False,
    }
    artifact["artifact_digest"] = compute_artifact_digest(
        {key: value for key, value in artifact.items() if key != "artifact_digest"}
    )
    integrity = verify_artifact_integrity(artifact)
    artifact["artifact_integrity"] = integrity
    return artifact


def build_gold_mock_responses(pack: dict[str, Any]) -> dict[str, str]:
    responses: dict[str, str] = {}
    for case_id in CASE_ORDER:
        case = pack["cases"][CASE_KEYS[case_id]]
        responses[case_id] = json.dumps({"observations": [case["observation"]]})
    return responses


def run_preflight(pack: dict[str, Any] | None = None) -> dict[str, Any]:
    pack = pack or load_fixture_pack()
    implementation_sha = get_implementation_git_sha()
    return {
        "harness_id": HARNESS_ID,
        "stage": STAGE,
        "pack_id": PACK_ID,
        "case_order": list(CASE_ORDER),
        "identity_hashes": build_run_identity_hashes(implementation_sha=implementation_sha),
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "provider_execution_default_disabled": not provider_execution_authorized(),
        "max_provider_requests": MAX_PROVIDER_REQUESTS,
        "max_attempts_per_case": MAX_ATTEMPTS_PER_CASE,
        "analyzer_response_contract": FROZEN_ANALYZER_RESPONSE_CONTRACT,
        "trusted_ingress_controls": [
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
        "preflight_ready": implementation_sha == REQUIRED_ENGINE_BASELINE_SHA,
    }
