"""Offline Call-A pre-provider qualification harness. No provider calls by default."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.claim_inventory import (  # noqa: E402
    ANCHOR_AMBIGUOUS,
    ANCHOR_BOUND,
    ANCHOR_BOUND_MULTIPLE,
    ANCHOR_FAILED,
    ANCHOR_UNRESOLVED,
    ClaimExtractorRow,
    ClaimInventoryError,
    build_claim_inventory,
    canonical_claim_text,
    inventory_critical_failures,
    parse_claim_inventory_rows,
)
from agent.nodes import CLAIM_INVENTORY_SYSTEM, _claim_inventory_user_message  # noqa: E402

DEFAULT_FIXTURE = REPO_ROOT / "evals" / "fixtures" / "claim_inventory_golden_v1.json"
DEFAULT_IDENTITY = REPO_ROOT / "evals" / "fixtures" / "call_a_preprovider_harness_identity.json"
HARNESS_MODULE = REPO_ROOT / "scripts" / "call_a_preprovider_harness.py"
PROMPT_PATH = REPO_ROOT / "prompts" / "claim_inventory_system.md"
CLAIM_INVENTORY_CODE = REPO_ROOT / "agent" / "claim_inventory.py"

HARNESS_ID = "call_a_preprovider_harness"
STAGE = "call-a-preprovider"
PACK_ID = "claim_inventory_golden_v1"

QUALIFIED_HARNESS_HEAD = "190360086fe8da3b001ca82dcee890edef9d971f"
EXPECTED_FIXTURE_SHA256 = "112b15ca3b79e16c38d0d9ba94063d60268a4a120414242115a55c5227dae006"

CASE_ORDER = (
    "G01-simple-factual",
    "G02-multi-atomic-one-sentence",
    "G03-repeated-identical-claim",
    "G04-near-duplicate-claims",
    "G05-factual-definition",
    "G06-editorial-nonfactual",
    "G07-code-factual-assertion",
    "G08-cross-section-claims",
    "G09-new-claim-after-revision",
    "G10-removed-claim",
    "G11-material-central-claim",
    "G12-incidental-factual",
    "G13-required-by-brief-material",
    "G14-false-nonmaterial-trap",
    "G15-anchor-ambiguity",
    "G16-anchor-failure",
)

MAX_PROVIDER_REQUESTS = len(CASE_ORDER)
MAX_ATTEMPTS_PER_CASE = 1

FIXTURE_EVAL_STATE = {
    "topic": "fixture-evaluation-topic",
    "card_id": "fixture-evaluation-card",
    "series_context": "",
}

CRITICAL_CHALLENGE_TAGS = frozenset({
    "required_by_brief",
    "material_central",
    "false_nonmaterial_trap",
    "multi_atomic",
    "repeated_claim",
    "near_duplicate",
    "factual_definition",
    "editorial",
    "code_factual",
    "anchor_ambiguity",
    "anchor_failure",
    "revision_new",
})

CaseDisposition = Literal["PASS", "FAIL", "INVALID", "NOT_RUN"]
OverallDisposition = Literal["PASS", "FAIL", "INVALID"]


class CallAHarnessError(ValueError):
    """Call-A qualification harness asset or evaluation error."""


class ResponseContractError(CallAHarnessError):
    """Call-A provider output violates the frozen response contract."""


@dataclass(frozen=True)
class CriticalGoldClaim:
    case_id: str
    claim_text: str
    designation: str


CRITICAL_GOLD_CLAIMS: tuple[CriticalGoldClaim, ...] = (
    CriticalGoldClaim("G13-required-by-brief-material", "The article includes a failure-modes analysis.", "required_by_brief"),
    CriticalGoldClaim("G11-material-central-claim", "Quorum requires seven acknowledgements before commit.", "material_central"),
    CriticalGoldClaim("G14-false-nonmaterial-trap", "The scheduler guarantees at-most-once delivery.", "false_nonmaterial_trap"),
    CriticalGoldClaim("G02-multi-atomic-one-sentence", "System A improves latency.", "multi_atomic"),
    CriticalGoldClaim("G02-multi-atomic-one-sentence", "System A reduces storage cost.", "multi_atomic"),
    CriticalGoldClaim("G03-repeated-identical-claim", "T3 caches aggressively.", "repeated_claim"),
    CriticalGoldClaim("G04-near-duplicate-claims", "Batch size 32 trains stably.", "near_duplicate"),
    CriticalGoldClaim("G04-near-duplicate-claims", "Batch size 32 usually trains stably.", "near_duplicate"),
    CriticalGoldClaim(
        "G05-factual-definition",
        "A transformer is a neural architecture that uses self-attention to model token interactions.",
        "factual_definition",
    ),
    CriticalGoldClaim("G06-editorial-nonfactual", "Dropout randomly zeroes activations during training.", "editorial_factual"),
    CriticalGoldClaim("G07-code-factual-assertion", "Binary search runs in O(log n) time.", "code_factual"),
    CriticalGoldClaim("G09-new-claim-after-revision", "Early stopping also helps.", "revision_new"),
    CriticalGoldClaim("G15-anchor-ambiguity", "Alpha mode permits export.", "anchor_ambiguity"),
    CriticalGoldClaim("G15-anchor-ambiguity", "Alpha mode logs the event.", "anchor_ambiguity"),
    CriticalGoldClaim("G16-anchor-failure", "Momentum always accelerates convergence.", "anchor_failure"),
)

CRITICAL_FALSE_NONMATERIAL_TRAPS: frozenset[tuple[str, str]] = frozenset({
    (c.case_id, canonical_claim_text(c.claim_text))
    for c in CRITICAL_GOLD_CLAIMS
    if c.designation == "false_nonmaterial_trap"
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CallAHarnessError(message)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def response_contract_sha256() -> str:
    schema = {"ClaimExtractorRow": ClaimExtractorRow.model_json_schema()}
    return sha256_text(canonical_json_dumps(schema))


def call_inventory_code_sha256() -> str:
    return sha256_bytes(CLAIM_INVENTORY_CODE.read_bytes())


def prompt_sha256() -> str:
    return sha256_bytes(PROMPT_PATH.read_bytes())


def get_implementation_git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def load_fixture_pack(path: Path | str = DEFAULT_FIXTURE) -> dict[str, Any]:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    observed = sha256_bytes(fixture_path.read_bytes())
    if observed != EXPECTED_FIXTURE_SHA256:
        raise CallAHarnessError(
            f"fixture sha256 mismatch: expected {EXPECTED_FIXTURE_SHA256}, got {observed}"
        )
    return payload


def validate_fixture_structure(pack: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on ambiguous or structurally invalid fixtures before provider auth."""
    invalid_reasons: list[str] = []
    cases = pack.get("cases")
    if not isinstance(cases, list) or not cases:
        invalid_reasons.append("missing_cases")
        return {"valid": False, "invalid_reasons": invalid_reasons}

    case_ids = [case.get("case_id") for case in cases]
    if len(set(case_ids)) != len(case_ids):
        invalid_reasons.append("duplicate_case_ids")
    if tuple(case_ids) != CASE_ORDER:
        invalid_reasons.append("case_order_mismatch")

    tags_seen: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str):
            invalid_reasons.append("missing_case_id")
            continue
        if not isinstance(case.get("draft_markdown"), str):
            invalid_reasons.append(f"{case_id}:missing_draft_markdown")
        if not isinstance(case.get("gold_claims"), list) or not case["gold_claims"]:
            invalid_reasons.append(f"{case_id}:missing_gold_claims")
        for tag in case.get("tags") or []:
            tags_seen.add(tag)
        for gold in case.get("gold_claims") or []:
            if not isinstance(gold.get("claim_text"), str) or not gold["claim_text"].strip():
                invalid_reasons.append(f"{case_id}:invalid_gold_claim_text")

    if not CRITICAL_CHALLENGE_TAGS <= tags_seen:
        invalid_reasons.append(
            "missing_critical_tags:" + ",".join(sorted(CRITICAL_CHALLENGE_TAGS - tags_seen))
        )

    for critical in CRITICAL_GOLD_CLAIMS:
        case = _case_by_id(pack, critical.case_id)
        gold_texts = {canonical_claim_text(g["claim_text"]) for g in case["gold_claims"]}
        if canonical_claim_text(critical.claim_text) not in gold_texts:
            invalid_reasons.append(
                f"critical_gold_missing:{critical.case_id}:{critical.claim_text[:40]}"
            )

    return {"valid": not invalid_reasons, "invalid_reasons": invalid_reasons}


def _case_by_id(pack: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in pack["cases"]:
        if case["case_id"] == case_id:
            return case
    raise CallAHarnessError(f"unknown case_id: {case_id}")


def build_fixture_eval_state(case: dict[str, Any]) -> dict[str, Any]:
    return {
        **FIXTURE_EVAL_STATE,
        "brief_requirements": case.get("brief_requirements") or [],
    }


def build_case_user_message(case: dict[str, Any]) -> str:
    return _claim_inventory_user_message(build_fixture_eval_state(case), case["draft_markdown"])


def build_case_request(case: dict[str, Any]) -> dict[str, Any]:
    user_message = build_case_user_message(case)
    from config import DEEPSEEK_MODEL, LLM_TIMEOUT_S

    request_body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": CLAIM_INVENTORY_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
    }
    return {
        "case_id": case["case_id"],
        "tags": case.get("tags") or [],
        "draft_markdown": case["draft_markdown"],
        "brief_requirements": case.get("brief_requirements") or [],
        "gold_claims": case["gold_claims"],
        "messages": request_body["messages"],
        "request_body": request_body,
        "request_body_sha256": sha256_text(canonical_json_dumps(request_body)),
        "prompt_sha256": prompt_sha256(),
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "model_config": {
            "provider": "deepseek",
            "requested_model": DEEPSEEK_MODEL,
            "accepted_returned_model": DEEPSEEK_MODEL,
            "temperature": 0.1,
            "max_tokens": 4000,
            "response_format": None,
            "timeout_s": LLM_TIMEOUT_S,
            "production_transport_retries": 3,
            "qualification_transport_retries": 0,
            "max_attempts_per_case": MAX_ATTEMPTS_PER_CASE,
        },
    }


def build_all_case_requests(pack: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    pack = pack or load_fixture_pack()
    by_id = {case["case_id"]: case for case in pack["cases"]}
    return [build_case_request(by_id[case_id]) for case_id in CASE_ORDER]


def build_request_identities(pack: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        req["case_id"]: {
            "request_body_sha256": req["request_body_sha256"],
            "prompt_sha256": req["prompt_sha256"],
            "fixture_sha256": req["fixture_sha256"],
        }
        for req in build_all_case_requests(pack)
    }


def build_gold_mock_response(case: dict[str, Any]) -> str:
    """Simulated provider output from frozen fixture call_a_response (harness only)."""
    return json.dumps(case["call_a_response"], ensure_ascii=True)


def merged_model_materiality(rows: list[dict]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for row in rows:
        key = canonical_claim_text(row["claim_text"])
        material = row["material"]
        if key not in merged:
            merged[key] = material
        elif material is True:
            merged[key] = True
        elif material == "unknown" and merged[key] is not True:
            merged[key] = "unknown"
    return merged


def _occurrences_key(claim: dict) -> list[list[int]]:
    return [[o["start"], o["end"]] for o in claim.get("occurrences", [])]


def _material_key(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def evaluate_parsed_case(
    *,
    case: dict[str, Any],
    raw_response: str,
    contract_error: str | None = None,
) -> dict[str, Any]:
    """Run production parse/build pipeline and compare to independent gold."""
    case_id = case["case_id"]
    result: dict[str, Any] = {
        "case_id": case_id,
        "tags": case.get("tags") or [],
        "contract_error": contract_error,
        "raw_provider_response": raw_response,
    }
    if contract_error:
        result["disposition"] = "INVALID"
        result["invalid_reasons"] = [contract_error]
        return result

    try:
        parsed_rows = parse_claim_inventory_rows(raw_response)
    except (ClaimInventoryError, json.JSONDecodeError, ValueError) as exc:
        result["disposition"] = "INVALID"
        result["invalid_reasons"] = [f"contract_error:{exc}"]
        result["contract_error"] = str(exc)
        return result

    inventory = build_claim_inventory(
        run_id=f"call-a-qual-{case_id}",
        iteration=1,
        draft_markdown=case["draft_markdown"],
        raw_claims=parsed_rows,
        brief_requirements=case.get("brief_requirements") or [],
    )
    model_material = merged_model_materiality(parsed_rows)
    produced = {canonical_claim_text(c["claim_text"]): c for c in inventory["claims"]}
    gold = {canonical_claim_text(g["claim_text"]): g for g in case["gold_claims"]}

    matched = sorted(set(produced) & set(gold))
    missing = sorted(set(gold) - set(produced))
    extra = sorted(set(produced) - set(gold))

    silently_dropped: list[str] = []
    produced_texts = {c["claim_text"].strip() for c in inventory["claims"]}
    for row in parsed_rows:
        if row["claim_text"].strip() not in produced_texts:
            silently_dropped.append(row["claim_text"])

    comparisons: list[dict[str, Any]] = []
    split_errors: list[dict[str, Any]] = []
    merge_errors: list[dict[str, Any]] = []
    claim_type_errors: list[dict[str, Any]] = []
    anchor_reports: list[dict[str, Any]] = []

    for key in matched:
        produced_claim = produced[key]
        gold_claim = gold[key]
        raw_mat = model_material.get(key)
        final_mat = produced_claim["material"]
        comparisons.append({
            "claim_text": gold_claim["claim_text"],
            "gold_type": gold_claim["claim_type"],
            "produced_type": produced_claim["claim_type"],
            "gold_material": gold_claim["material"],
            "raw_model_material": raw_mat,
            "final_policy_material": final_mat,
            "materiality_override": produced_claim.get("materiality_override"),
            "gold_anchor_validity": gold_claim["anchor_validity"],
            "produced_anchor_validity": produced_claim["anchor_validity"],
            "occurrences_match": _occurrences_key(produced_claim) == gold_claim["occurrences"],
        })
        if produced_claim["claim_type"] != gold_claim["claim_type"]:
            claim_type_errors.append({
                "claim_text": gold_claim["claim_text"],
                "gold_type": gold_claim["claim_type"],
                "produced_type": produced_claim["claim_type"],
            })

    for key in sorted(set(produced) | set(gold)):
        if key in gold and key in produced:
            continue
        if key in gold:
            continue
        claim = produced[key]
        anchor_reports.append({
            "claim_text": claim["claim_text"],
            "anchor_validity": claim["anchor_validity"],
            "occurrences": _occurrences_key(claim),
            "invented": True,
        })

    for claim in inventory["claims"]:
        anchor_reports.append({
            "claim_text": claim["claim_text"],
            "anchor_validity": claim["anchor_validity"],
            "occurrences": _occurrences_key(claim),
            "invented": canonical_claim_text(claim["claim_text"]) not in gold,
        })

    gold_texts = set(gold)
    produced_texts_canon = set(produced)
    for key in gold_texts:
        contained = [p for p in produced_texts_canon if p != key and p in key]
        if len(contained) >= 2:
            split_errors.append({"gold": key, "parts": sorted(contained)})
    for key in produced_texts_canon:
        contained = [g for g in gold_texts if g != key and g in key]
        if len(contained) >= 2:
            merge_errors.append({"produced": key, "parts": sorted(contained)})

    false_nonmaterial_raw: list[str] = []
    false_nonmaterial_final: list[str] = []
    for comp in comparisons:
        if comp["gold_material"] is True:
            if comp["raw_model_material"] is False:
                false_nonmaterial_raw.append(comp["claim_text"])
            if comp["final_policy_material"] is False:
                false_nonmaterial_final.append(comp["claim_text"])

    critical_misses = [
        {
            "case_id": c.case_id,
            "claim_text": c.claim_text,
            "designation": c.designation,
        }
        for c in CRITICAL_GOLD_CLAIMS
        if c.case_id == case_id and canonical_claim_text(c.claim_text) in missing
    ]

    critical_false_nonmaterial_final = [
        comp["claim_text"]
        for comp in comparisons
        if (case_id, canonical_claim_text(comp["claim_text"])) in CRITICAL_FALSE_NONMATERIAL_TRAPS
        and comp["final_policy_material"] is False
    ]

    unresolved_anchors = [
        c["claim_text"]
        for c in inventory["claims"]
        if c.get("anchor_validity") in ANCHOR_UNRESOLVED
    ]
    critical_failures = inventory_critical_failures(inventory)

    gate_failures: list[str] = []
    if silently_dropped:
        gate_failures.append("silently_dropped_emitted_claims")
    if critical_misses:
        gate_failures.append("critical_gold_miss")
    if critical_false_nonmaterial_final:
        gate_failures.append("critical_false_nonmaterial")
    disposition: CaseDisposition = "PASS" if not gate_failures else "FAIL"
    result.update({
        "disposition": disposition,
        "parsed_rows": parsed_rows,
        "inventory": inventory,
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "silently_dropped": silently_dropped,
        "comparisons": comparisons,
        "split_errors": split_errors,
        "merge_errors": merge_errors,
        "claim_type_errors": claim_type_errors,
        "anchor_reports": anchor_reports,
        "false_nonmaterial_raw": false_nonmaterial_raw,
        "false_nonmaterial_final": false_nonmaterial_final,
        "critical_misses": critical_misses,
        "critical_false_nonmaterial_final": critical_false_nonmaterial_final,
        "unresolved_anchors": unresolved_anchors,
        "inventory_critical_failures": [
            {"claim_id": c["claim_id"], "claim_text": c["claim_text"], "anchor_validity": c["anchor_validity"]}
            for c in critical_failures
        ],
        "gate_failures": gate_failures,
    })
    return result


def compute_aggregate_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    gold_factual = produced_factual = matched_factual = 0
    gold_material = material_correct_raw = material_correct_final = 0
    confusion_raw: dict[str, dict[str, int]] = {
        g: {"true": 0, "false": 0, "unknown": 0} for g in ("true", "false", "unknown")
    }
    confusion_final: dict[str, dict[str, int]] = {
        g: {"true": 0, "false": 0, "unknown": 0} for g in ("true", "false", "unknown")
    }
    override_count = 0
    split_errors: list[dict[str, Any]] = []
    merge_errors: list[dict[str, Any]] = []
    anchor_bind_success = anchor_multiple = anchor_failed = anchor_ambiguous = 0
    claim_type_errors: list[dict[str, Any]] = []
    missed_gold: list[dict[str, str]] = []
    invented_claims: list[dict[str, str]] = []
    critical_failures: list[dict[str, Any]] = []

    for result in case_results:
        if result.get("disposition") == "INVALID":
            continue
        case_id = result["case_id"]
        inventory = result["inventory"]
        for gold in _case_by_id(load_fixture_pack(), case_id)["gold_claims"]:
            if gold["claim_type"] == "factual":
                gold_factual += 1
        for claim in inventory["claims"]:
            if claim["claim_type"] == "factual":
                produced_factual += 1
            validity = claim.get("anchor_validity")
            if validity == ANCHOR_BOUND:
                anchor_bind_success += 1
            elif validity == ANCHOR_BOUND_MULTIPLE:
                anchor_multiple += 1
            elif validity == ANCHOR_FAILED:
                anchor_failed += 1
            elif validity == ANCHOR_AMBIGUOUS:
                anchor_ambiguous += 1

        for comp in result.get("comparisons") or []:
            if comp["gold_type"] == "factual":
                matched_factual += 1
            gm = _material_key(comp["gold_material"])
            confusion_raw[gm][_material_key(comp["raw_model_material"])] += 1
            confusion_final[gm][_material_key(comp["final_policy_material"])] += 1
            if comp["gold_material"] is True:
                gold_material += 1
                if comp["raw_model_material"] is True:
                    material_correct_raw += 1
                if comp["final_policy_material"] is True:
                    material_correct_final += 1
            if comp.get("materiality_override"):
                override_count += 1

        for text in result.get("missing") or []:
            missed_gold.append({"case_id": case_id, "claim_text": text})
        for text in result.get("extra") or []:
            invented_claims.append({"case_id": case_id, "claim_text": text})
        split_errors.extend({"case_id": case_id, **row} for row in result.get("split_errors") or [])
        merge_errors.extend({"case_id": case_id, **row} for row in result.get("merge_errors") or [])
        claim_type_errors.extend(
            {"case_id": case_id, **row} for row in result.get("claim_type_errors") or []
        )
        for miss in result.get("critical_misses") or []:
            critical_failures.append({**miss, "kind": "critical_gold_miss"})
        for text in result.get("critical_false_nonmaterial_final") or []:
            critical_failures.append({
                "case_id": case_id,
                "claim_text": text,
                "kind": "critical_false_nonmaterial",
            })
        if result.get("silently_dropped"):
            critical_failures.append({
                "case_id": case_id,
                "kind": "silently_dropped",
                "claims": result["silently_dropped"],
            })

    return {
        "factual_true_positives": matched_factual,
        "factual_false_negatives": gold_factual - matched_factual,
        "factual_false_positives": produced_factual - matched_factual,
        "factual_recall": (matched_factual / gold_factual) if gold_factual else None,
        "factual_precision": (matched_factual / produced_factual) if produced_factual else None,
        "factual_recall_denominator": gold_factual,
        "factual_precision_denominator": produced_factual,
        "material_claim_recall_raw": (material_correct_raw / gold_material) if gold_material else None,
        "material_claim_recall_final": (material_correct_final / gold_material) if gold_material else None,
        "material_claim_recall_denominator": gold_material,
        "materiality_confusion_raw": confusion_raw,
        "materiality_confusion_final": confusion_final,
        "required_material_override_count": override_count,
        "missed_gold_claims": missed_gold,
        "invented_claims": invented_claims,
        "split_errors": split_errors,
        "merge_errors": merge_errors,
        "claim_type_errors": claim_type_errors,
        "anchor_bind_success": anchor_bind_success,
        "anchor_bound_multiple": anchor_multiple,
        "anchor_failed": anchor_failed,
        "anchor_ambiguous": anchor_ambiguous,
        "critical_failures": critical_failures,
    }


def apply_overall_gate(
    *,
    case_results: list[dict[str, Any]],
    provider_invalid_count: int,
    identity_drift: bool,
) -> dict[str, Any]:
    gate_reasons: list[str] = []
    if provider_invalid_count:
        gate_reasons.append("provider_or_contract_failures")
    if identity_drift:
        gate_reasons.append("provider_identity_drift")
    if any(result.get("silently_dropped") for result in case_results):
        gate_reasons.append("silently_dropped_emitted_claims")
    if any(result.get("critical_misses") for result in case_results):
        gate_reasons.append("critical_gold_miss")
    if any(result.get("critical_false_nonmaterial_final") for result in case_results):
        gate_reasons.append("critical_false_nonmaterial")
    overall: OverallDisposition
    if provider_invalid_count or identity_drift:
        overall = "INVALID"
    elif gate_reasons:
        overall = "FAIL"
    else:
        overall = "PASS"

    return {
        "overall_disposition": overall,
        "gate_reasons": sorted(set(gate_reasons)),
        "pass_conditions": {
            "A_no_provider_contract_failures": provider_invalid_count == 0,
            "B_no_silently_dropped_claims": not any(r.get("silently_dropped") for r in case_results),
            "C_no_critical_gold_misses": not any(r.get("critical_misses") for r in case_results),
            "D_no_critical_false_nonmaterial": not any(
                r.get("critical_false_nonmaterial_final") for r in case_results
            ),
            "E_unresolved_anchors_fail_closed": True,
            "F_no_identity_or_fixture_drift": not identity_drift,
        },
    }


def _git_is_ancestor(ancestor_sha: str, descendant_sha: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def validate_harness_identities(*, harness_implementation_sha: str | None = None) -> dict[str, Any]:
    harness_sha = harness_implementation_sha or get_implementation_git_sha()
    invalid_reasons: list[str] = []
    if harness_sha != QUALIFIED_HARNESS_HEAD and not _git_is_ancestor(QUALIFIED_HARNESS_HEAD, harness_sha):
        invalid_reasons.append("harness_head_not_qualified_ancestor")
    if not HARNESS_MODULE.is_file():
        invalid_reasons.append("harness_module_missing")
    if sha256_bytes(PROMPT_PATH.read_bytes()) != prompt_sha256():
        invalid_reasons.append("prompt_hash_unstable")
    pack = load_fixture_pack()
    fixture_validation = validate_fixture_structure(pack)
    if not fixture_validation["valid"]:
        invalid_reasons.extend(fixture_validation["invalid_reasons"])
    return {
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "harness_implementation_sha": harness_sha,
        "qualified_harness_head": QUALIFIED_HARNESS_HEAD,
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "prompt_sha256": prompt_sha256(),
        "response_contract_sha256": response_contract_sha256(),
        "claim_inventory_code_sha256": call_inventory_code_sha256(),
    }


def fixture_population_report(pack: dict[str, Any] | None = None) -> dict[str, Any]:
    pack = pack or load_fixture_pack()
    gold_claims = sum(len(c["gold_claims"]) for c in pack["cases"])
    gold_material = sum(
        1 for c in pack["cases"] for g in c["gold_claims"] if g["material"] is True
    )
    gold_factual = sum(
        1 for c in pack["cases"] for g in c["gold_claims"] if g["claim_type"] == "factual"
    )
    return {
        "fixture_path": str(DEFAULT_FIXTURE.relative_to(REPO_ROOT)),
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "fixture_count": len(pack["cases"]),
        "gold_claim_count": gold_claims,
        "gold_factual_claim_count": gold_factual,
        "gold_material_claim_count": gold_material,
        "critical_challenge_cases": [
            {"case_id": c.case_id, "claim_text": c.claim_text, "designation": c.designation}
            for c in CRITICAL_GOLD_CLAIMS
        ],
        "critical_challenge_tags": sorted(CRITICAL_CHALLENGE_TAGS),
    }
