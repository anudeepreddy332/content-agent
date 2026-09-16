"""Provider-eval oracle v2 for live Call-A qualification rescoring.

HARNESS GOLD (claim_inventory_golden_v1) != PROVIDER EVAL GOLD (call_a_provider_eval_gold_v2).

This module scores immutable raw provider responses using reviewer-approved,
case-specific matching rules. No LLM judge. No open-ended semantic similarity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from agent.claim_inventory import (
    ANCHOR_AMBIGUOUS,
    ANCHOR_FAILED,
    build_claim_inventory,
    canonical_claim_text,
    parse_claim_inventory_rows,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ORACLE_PATH = REPO_ROOT / "evals/fixtures/call_a_provider_eval_gold_v2.json"
HARNESS_PATH = REPO_ROOT / "evals/fixtures/claim_inventory_golden_v1.json"
EXPECTED_ORACLE_SHA256 = "4591914638478bd1531da2929b9cff3f8f70e2654f194fbdb1d566f914848314"

MatchRule = Literal[
    "exact",
    "semantic_equivalent",
    "atomic_decomposition",
    "inventory_descriptive_only",
]


class ProviderEvalV2Error(ValueError):
    """Provider-eval v2 oracle or scoring error."""


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def load_oracle(path: Path | str = ORACLE_PATH) -> dict[str, Any]:
    oracle_path = Path(path)
    payload = json.loads(oracle_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "call_a_provider_eval_gold_v2":
        raise ProviderEvalV2Error("invalid provider eval oracle schema")
    observed = sha256_bytes(oracle_path.read_bytes())
    if EXPECTED_ORACLE_SHA256 and observed != EXPECTED_ORACLE_SHA256:
        raise ProviderEvalV2Error(f"oracle sha256 mismatch: {observed}")
    harness_sha = sha256_bytes(HARNESS_PATH.read_bytes())
    if payload.get("harness_fixture_sha256") != harness_sha:
        raise ProviderEvalV2Error("harness fixture sha256 drift vs oracle reference")
    return payload


def load_harness_pack() -> dict[str, Any]:
    return json.loads(HARNESS_PATH.read_text(encoding="utf-8"))


def _canon(text: str) -> str:
    return canonical_claim_text(text)


def _case_eval(oracle: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in oracle["cases"]:
        if case["case_id"] == case_id:
            return case
    raise ProviderEvalV2Error(f"unknown provider eval case: {case_id}")


def _case_harness(harness: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in harness["cases"]:
        if case["case_id"] == case_id:
            return case
    raise ProviderEvalV2Error(f"unknown harness case: {case_id}")


def _producer_materiality(
    produced: dict[str, dict],
    model_material: dict[str, Any],
    producer_key: str,
) -> Any:
    if producer_key in produced:
        return produced[producer_key]["material"]
    return model_material.get(producer_key)


def _match_eval_claim(
    eval_claim: dict[str, Any],
    produced_keys: set[str],
    produced: dict[str, dict],
    model_material: dict[str, Any],
) -> dict[str, Any]:
    rule: MatchRule = eval_claim.get("match_rule", "exact")
    result: dict[str, Any] = {
        "eval_claim_text": eval_claim["claim_text"],
        "claim_type": eval_claim["claim_type"],
        "material": eval_claim["material"],
        "critical": eval_claim.get("critical", False),
        "match_rule": rule,
        "covered": False,
        "matched_producer_keys": [],
    }

    if rule == "exact":
        key = _canon(eval_claim["claim_text"])
        if key in produced_keys:
            result["covered"] = True
            result["matched_producer_keys"] = [key]
    elif rule == "semantic_equivalent":
        allowed = {_canon(eval_claim["claim_text"])}
        for alt in eval_claim.get("equivalent_producer_claims") or []:
            allowed.add(_canon(alt))
        matched = sorted(produced_keys & allowed)
        if matched:
            result["covered"] = True
            result["matched_producer_keys"] = matched
    elif rule == "atomic_decomposition":
        parts = [_canon(p) for p in eval_claim["decomposition_parts"]]
        if all(p in produced_keys for p in parts):
            result["covered"] = True
            result["matched_producer_keys"] = parts
    elif rule == "inventory_descriptive_only":
        result["covered"] = True  # never fails factual critical gate
        result["inventory_descriptive_only"] = True
        key = _canon(eval_claim["claim_text"])
        editorial_match = key if key in produced_keys else None
        for pk in produced_keys:
            claim = produced[pk]
            if claim.get("claim_type") == "editorial":
                editorial_match = editorial_match or pk
        result["editorial_inventory_observed"] = editorial_match is not None
        if editorial_match:
            result["matched_producer_keys"] = [editorial_match]
    else:
        raise ProviderEvalV2Error(f"unsupported match_rule: {rule}")

    if result["matched_producer_keys"]:
        pk = result["matched_producer_keys"][0]
        result["raw_model_material"] = _producer_materiality(produced, model_material, pk)
        if pk in produced:
            result["final_policy_material"] = produced[pk]["material"]
            result["produced_type"] = produced[pk]["claim_type"]
            result["produced_anchor_validity"] = produced[pk]["anchor_validity"]
            if produced[pk]["claim_type"] != eval_claim["claim_type"]:
                result["claim_type_mismatch"] = True
    return result


def evaluate_provider_eval_v2_case(
    *,
    case_eval: dict[str, Any],
    harness_case: dict[str, Any],
    inventory: dict[str, Any],
    parsed_rows: list[dict],
    raw_response: str,
) -> dict[str, Any]:
    """Score one case against provider-eval v2 oracle."""
    from scripts.call_a_preprovider_harness import merged_model_materiality

    case_id = case_eval["case_id"]
    produced = {_canon(c["claim_text"]): c for c in inventory["claims"]}
    produced_keys = set(produced)
    model_material = merged_model_materiality(parsed_rows)

    claim_matches = [
        _match_eval_claim(ec, produced_keys, produced, model_material)
        for ec in case_eval["provider_eval_claims"]
    ]

    used_producer: set[str] = set()
    for match in claim_matches:
        used_producer.update(match.get("matched_producer_keys") or [])

    invented = sorted(
        key for key in produced_keys
        if key not in used_producer
    )
    invented_claim_texts = [produced[k]["claim_text"] for k in invented]

    factual_eval = [m for m in claim_matches if m["claim_type"] == "factual"]
    factual_covered = sum(1 for m in factual_eval if m["covered"])
    factual_total = len(factual_eval)

    critical_misses = [
        {
            "case_id": case_id,
            "claim_text": m["eval_claim_text"],
            "match_rule": m["match_rule"],
        }
        for m in claim_matches
        if m.get("critical") and not m["covered"] and m["match_rule"] != "inventory_descriptive_only"
    ]

    false_nonmaterial_raw: list[str] = []
    false_nonmaterial_final: list[str] = []
    unknown_materiality: list[str] = []
    for m in claim_matches:
        if m["claim_type"] != "factual" or not m["covered"]:
            continue
        gold_mat = m["material"]
        raw_mat = m.get("raw_model_material")
        final_mat = m.get("final_policy_material")
        if gold_mat is True and raw_mat is False:
            false_nonmaterial_raw.append(m["eval_claim_text"])
        if gold_mat is True and final_mat is False:
            false_nonmaterial_final.append(m["eval_claim_text"])
        if raw_mat == "unknown" or final_mat == "unknown":
            unknown_materiality.append(m["eval_claim_text"])

    critical_false_nonmaterial: list[str] = []
    if case_id == "G14-false-nonmaterial-trap":
        for m in claim_matches:
            if m.get("final_policy_material") is False:
                critical_false_nonmaterial.append(m["eval_claim_text"])

    silently_dropped: list[str] = []
    produced_texts = {c["claim_text"].strip() for c in inventory["claims"]}
    for row in parsed_rows:
        if row["claim_text"].strip() not in produced_texts:
            silently_dropped.append(row["claim_text"])

    gate_failures: list[str] = []
    if silently_dropped:
        gate_failures.append("silently_dropped_emitted_claims")
    if critical_misses:
        gate_failures.append("critical_gold_miss")
    if critical_false_nonmaterial:
        gate_failures.append("critical_false_nonmaterial")

    disposition = "PASS" if not gate_failures else "FAIL"

    return {
        "case_id": case_id,
        "disposition": disposition,
        "oracle_version": 2,
        "claim_matches": claim_matches,
        "invented_producer_claims": invented_claim_texts,
        "critical_misses": critical_misses,
        "critical_false_nonmaterial_final": critical_false_nonmaterial,
        "false_nonmaterial_raw": false_nonmaterial_raw,
        "false_nonmaterial_final": false_nonmaterial_final,
        "unknown_materiality": unknown_materiality,
        "silently_dropped": silently_dropped,
        "gate_failures": gate_failures,
        "factual_eval_total": factual_total,
        "factual_eval_covered": factual_covered,
        "inventory": inventory,
        "parsed_rows": parsed_rows,
        "raw_provider_response": raw_response,
        "harness_only_trap": case_eval.get("harness_only_trap"),
    }


def compute_provider_eval_v2_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    factual_tp = factual_fn = 0
    produced_factual = 0
    gold_material = material_correct_raw = material_correct_final = 0
    false_nonmaterial_raw = false_nonmaterial_final = 0
    unknown_materiality: list[dict[str, str]] = []
    split_errors: list[dict[str, Any]] = []
    merge_errors: list[dict[str, Any]] = []
    invented: list[dict[str, str]] = []
    critical_misses: list[dict[str, Any]] = []
    anchor_bind = anchor_multiple = anchor_failed = anchor_ambiguous = 0

    for result in results:
        case_id = result["case_id"]
        inventory = result["inventory"]
        for claim in inventory["claims"]:
            if claim["claim_type"] == "factual":
                produced_factual += 1
            v = claim.get("anchor_validity")
            if v == "bound":
                anchor_bind += 1
            elif v == "bound_multiple":
                anchor_multiple += 1
            elif v == ANCHOR_FAILED:
                anchor_failed += 1
            elif v == ANCHOR_AMBIGUOUS:
                anchor_ambiguous += 1

        factual_tp += result["factual_eval_covered"]
        factual_fn += result["factual_eval_total"] - result["factual_eval_covered"]

        for text in result["invented_producer_claims"]:
            invented.append({"case_id": case_id, "claim_text": text})

        for m in result["claim_matches"]:
            if m["claim_type"] == "factual" and m["covered"]:
                if m["material"] is True:
                    gold_material += 1
                    if m.get("raw_model_material") is True:
                        material_correct_raw += 1
                    if m.get("final_policy_material") is True:
                        material_correct_final += 1
                if m.get("raw_model_material") == "unknown" or m.get("final_policy_material") == "unknown":
                    unknown_materiality.append({"case_id": case_id, "claim_text": m["eval_claim_text"]})

        false_nonmaterial_raw += len(result["false_nonmaterial_raw"])
        false_nonmaterial_final += len(result["false_nonmaterial_final"])
        critical_misses.extend(result["critical_misses"])

    factual_fp = max(produced_factual - factual_tp, 0)

    return {
        "factual_true_positives": factual_tp,
        "factual_false_negatives": factual_fn,
        "factual_false_positives": factual_fp,
        "factual_recall": (factual_tp / (factual_tp + factual_fn)) if (factual_tp + factual_fn) else None,
        "factual_precision": (factual_tp / (factual_tp + factual_fp)) if (factual_tp + factual_fp) else None,
        "factual_recall_denominator": factual_tp + factual_fn,
        "factual_precision_denominator": factual_tp + factual_fp,
        "material_claim_recall_raw": (material_correct_raw / gold_material) if gold_material else None,
        "material_claim_recall_final": (material_correct_final / gold_material) if gold_material else None,
        "material_claim_recall_denominator": gold_material,
        "raw_false_nonmaterial_count": false_nonmaterial_raw,
        "final_false_nonmaterial_count": false_nonmaterial_final,
        "unknown_materiality": unknown_materiality,
        "split_errors": split_errors,
        "merge_errors": merge_errors,
        "invented_claims": invented,
        "critical_misses": critical_misses,
        "anchor_bind_success": anchor_bind,
        "anchor_bound_multiple": anchor_multiple,
        "anchor_failed": anchor_failed,
        "anchor_ambiguous": anchor_ambiguous,
    }


def apply_provider_eval_v2_gate(
    *,
    case_results: list[dict[str, Any]],
    contract_failures: int,
    artifact_digest_valid: bool,
) -> dict[str, Any]:
    gate_reasons: list[str] = []
    if contract_failures:
        gate_reasons.append("provider_or_contract_failures")
    if not artifact_digest_valid:
        gate_reasons.append("artifact_digest_corruption")
    if any(r.get("silently_dropped") for r in case_results):
        gate_reasons.append("silently_dropped_emitted_claims")
    if any(r.get("critical_misses") for r in case_results):
        gate_reasons.append("critical_gold_miss")
    if any(r.get("critical_false_nonmaterial_final") for r in case_results):
        gate_reasons.append("critical_false_nonmaterial")

    if contract_failures or not artifact_digest_valid:
        overall = "INVALID"
    elif gate_reasons:
        overall = "FAIL"
    else:
        overall = "PASS"

    return {
        "overall_disposition": overall,
        "gate_reasons": sorted(set(gate_reasons)),
        "pass_conditions": {
            "A_no_provider_contract_failures": contract_failures == 0,
            "B_no_silently_dropped_claims": not any(r.get("silently_dropped") for r in case_results),
            "C_no_critical_gold_misses": not any(r.get("critical_misses") for r in case_results),
            "D_no_critical_false_nonmaterial": not any(
                r.get("critical_false_nonmaterial_final") for r in case_results
            ),
            "E_explicit_anchor_disposition": all(
                all(c.get("anchor_validity") not in (None, "") for c in r["inventory"]["claims"])
                for r in case_results
            ),
            "F_no_artifact_corruption": artifact_digest_valid,
        },
    }


def rescore_live_run_artifact(
    *,
    run_artifact_path: Path | str,
    expected_artifact_digest: str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Re-score immutable live provider run under provider-eval v2. Zero provider calls."""
    from scripts.call_a_provider_qualification_runner import verify_run_artifact_integrity

    run_path = Path(run_artifact_path)
    artifact = json.loads(run_path.read_text(encoding="utf-8"))
    integrity = verify_run_artifact_integrity(artifact)
    if artifact.get("artifact_digest") != expected_artifact_digest:
        raise ProviderEvalV2Error("source artifact digest mismatch")
    if not integrity["valid"]:
        raise ProviderEvalV2Error("source artifact integrity invalid")

    oracle = load_oracle()
    harness = load_harness_pack()
    oracle_sha = sha256_bytes(ORACLE_PATH.read_bytes())

    case_results_v2: list[dict[str, Any]] = []
    contract_failures = 0
    for live in artifact["case_results"]:
        case_id = live["case_id"]
        if live.get("disposition") == "INVALID":
            contract_failures += 1
            case_results_v2.append({
                "case_id": case_id,
                "disposition": "INVALID",
                "oracle_version": 2,
                "source_disposition": live["disposition"],
                "skipped_scoring": True,
            })
            continue

        raw = live.get("raw_provider_response") or (
            (live.get("provider_response_validation") or {}).get("raw_content") or ""
        )
        parsed_rows = live.get("parsed_rows") or parse_claim_inventory_rows(raw)
        inventory = live.get("inventory") or build_claim_inventory(
            run_id=f"rescore-v2-{case_id}",
            iteration=1,
            draft_markdown=_case_harness(harness, case_id)["draft_markdown"],
            raw_claims=parsed_rows,
            brief_requirements=_case_harness(harness, case_id).get("brief_requirements") or [],
        )
        scored = evaluate_provider_eval_v2_case(
            case_eval=_case_eval(oracle, case_id),
            harness_case=_case_harness(harness, case_id),
            inventory=inventory,
            parsed_rows=parsed_rows,
            raw_response=raw,
        )
        scored["source_disposition_v1"] = live.get("disposition")
        case_results_v2.append(scored)

    metrics = compute_provider_eval_v2_metrics(
        [r for r in case_results_v2 if not r.get("skipped_scoring")]
    )
    overall_gate = apply_provider_eval_v2_gate(
        case_results=[r for r in case_results_v2 if not r.get("skipped_scoring")],
        contract_failures=contract_failures,
        artifact_digest_valid=integrity["valid"],
    )

    rescore = {
        "rescore_id": f"{artifact['run_id']}_provider_eval_v2",
        "oracle_version": 2,
        "oracle_schema": "call_a_provider_eval_gold_v2",
        "oracle_sha256": oracle_sha,
        "source_run_id": artifact["run_id"],
        "source_artifact_path": _relative_to_repo(run_path),
        "source_artifact_digest": expected_artifact_digest,
        "source_overall_disposition_v1": artifact.get("overall_disposition"),
        "provider_calls": 0,
        "case_results_v2": case_results_v2,
        "aggregate_metrics_v2": metrics,
        "overall_gate_v2": overall_gate,
        "overall_disposition_v2": overall_gate["overall_disposition"],
        "bounded_qualification_v2": overall_gate["overall_disposition"] == "PASS",
    }
    rescore["rescore_digest"] = sha256_text(canonical_json_dumps({
        k: v for k, v in rescore.items() if k != "rescore_digest"
    }))

    out = Path(output_path) if output_path else run_path.parent / "provider_eval_v2_rescore.json"
    if out.exists():
        raise ProviderEvalV2Error(f"derived rescore artifact already exists: {out}")
    out.write_text(json.dumps(rescore, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    rescore["rescore_artifact_path"] = _relative_to_repo(out)
    return rescore
