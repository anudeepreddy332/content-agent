"""Deterministic coverage for claim-semantics v2 and P0 semantic metrics. No providers."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_claim_semantics_v2 import (
    ClaimSemanticsV2Error,
    DEFAULT_FIXTURES,
    DEFAULT_REGISTRY,
    DEFAULT_SCHEMA,
    MATCH_INELIGIBLE_ROLES,
    assert_schema_runtime_parity,
    evaluate_fixture,
    evaluate_pack,
    load_frozen_schema,
    load_metric_registry,
    load_pack,
    maximum_cardinality_matching,
    normalize_json,
    sha256_text,
    validate_against_frozen_schema,
    validate_fixture,
)


def _pack() -> dict:
    return load_pack(DEFAULT_FIXTURES)


def _fixture(pack: dict | None = None) -> dict:
    pack = pack or _pack()
    return pack["fixtures"][0]


def _metrics(pack: dict | None = None) -> dict:
    return evaluate_fixture(_fixture(pack))["metrics"]


def _ratio(metric: dict, numerator: int, denominator: int) -> None:
    assert metric["numerator"] == numerator
    assert metric["denominator"] == denominator
    assert metric["undefined"] is False
    assert metric["undefined_reason"] is None
    assert metric["value"] == numerator / denominator


def _undefined(metric: dict, reason_substring: str, *, numerator=0, denominator=0) -> None:
    assert metric["undefined"] is True
    assert metric["value"] is None
    assert reason_substring in metric["undefined_reason"]
    assert metric["numerator"] == numerator
    assert metric["denominator"] == denominator


def test_v1_evaluator_artifacts_remain_unmodified():
    v1_eval = Path("scripts/evaluate_claim_semantics.py").read_text(encoding="utf-8")
    assert 'MATCH_INELIGIBLE_ROLES = frozenset({"compound", "fragment"})' in v1_eval
    assert "qualifier_loss" not in v1_eval.split("MATCH_INELIGIBLE_ROLES")[1].split("\n")[0]


def test_registry_has_exactly_five_identities():
    registry = load_metric_registry()
    names = {metric["canonical_name"] for metric in registry["metrics"]}
    assert names == {
        "material_claim_recall.v2",
        "material_claim_unresolved_rate.v1",
        "material_false_verification_rate.v1",
        "automatic_semantic_false_pass_rate.v1",
        "unverified_verifier_row_rate.UVR_v1",
    }
    for metric in registry["metrics"]:
        required = {
            "canonical_name",
            "version",
            "formula",
            "numerator_definition",
            "denominator_definition",
            "eligibility_population",
            "canonical_evaluation_unit",
            "aggregation",
            "missing_invalid_behavior",
            "zero_denominator_behavior",
            "range_unit",
            "direction",
            "exact_meaning",
            "explicit_non_meaning",
            "authority",
            "provenance_requirements",
            "deterministic_fixture_reference",
            "hand_calculated_expected_result",
            "reference_implementation_requirements",
            "migration_versioning_rules",
        }
        assert required <= set(metric)


def test_core_fixture_hand_results():
    metrics = _metrics()
    _ratio(metrics["material_claim_recall.v2"], 3, 4)
    _ratio(metrics["material_claim_unresolved_rate.v1"], 3, 4)
    _ratio(metrics["material_false_verification_rate.v1"], 1, 2)
    _ratio(metrics["unverified_verifier_row_rate.UVR_v1"], 0, 5)
    report = evaluate_pack(_pack())
    _ratio(report["metrics"]["automatic_semantic_false_pass_rate.v1"], 1, 1)
    _ratio(report["results"][0]["metrics"]["automatic_semantic_false_pass_rate.v1"], 1, 1)


def test_uvr_sidecar_coexists_with_semantic_failure():
    result = evaluate_fixture(_fixture())
    assert result["oracle"]["semantic_pass"] is False
    assert result["oracle"]["unresolved_material_atoms"] == 3
    _ratio(result["metrics"]["unverified_verifier_row_rate.UVR_v1"], 0, 5)


def test_all_weak_material_asset_cannot_oracle_pass():
    fixture = copy.deepcopy(_fixture())
    for gold in fixture["gold_atoms"]:
        if gold["material"]:
            gold["gold_semantic_status"] = "weak"
    for candidate in fixture["candidates"]:
        if candidate["id"] != "cx":
            candidate["predicted_semantic_status"] = "weak"
    fixture["automatic_route"] = {"decision": "FAIL"}
    result = evaluate_fixture(fixture)
    assert result["oracle"]["semantic_pass"] is False


@pytest.mark.parametrize("role", sorted(MATCH_INELIGIBLE_ROLES))
def test_forbidden_role_candidate_cannot_earn_recall_credit(role: str):
    fixture = copy.deepcopy(_fixture())
    candidate_id = f"ADV01.test.{role}"
    fixture["candidates"].append({
        "id": candidate_id,
        "canonical_id": f"ADV01.C.test_{role}",
        "text": "ResNet-50 contains 25.6 million parameters.",
        "span": None,
        "roles": [role],
        "predicted_semantic_status": "verified",
    })
    fixture["allowed_matches"] = [{"candidate_id": candidate_id, "gold_id": "g1"}]
    fixture["candidates"] = [candidate for candidate in fixture["candidates"] if candidate["id"] != "c1"]
    fixture["evidence_bindings"] = [
        binding for binding in fixture["evidence_bindings"] if binding["gold_id"] != "g1"
    ]
    with pytest.raises(ClaimSemanticsV2Error, match="forbidden-role allowed edge"):
        validate_fixture(fixture)


def test_forbidden_role_allowed_edge_fails_closed():
    fixture = copy.deepcopy(_fixture())
    fixture["allowed_matches"].append({"candidate_id": "cx", "gold_id": "g1"})
    with pytest.raises(ClaimSemanticsV2Error, match="forbidden-role allowed edge"):
        validate_fixture(fixture)


def test_unknown_candidate_id_fails_closed():
    fixture = copy.deepcopy(_fixture())
    fixture["allowed_matches"][0]["candidate_id"] = "ghost"
    with pytest.raises(ClaimSemanticsV2Error, match="unknown candidate_id"):
        validate_fixture(fixture)


def test_unknown_gold_id_fails_closed():
    fixture = copy.deepcopy(_fixture())
    fixture["allowed_matches"][0]["gold_id"] = "ghost"
    with pytest.raises(ClaimSemanticsV2Error, match="unknown gold_id"):
        validate_fixture(fixture)


def test_duplicate_match_edge_fails_closed():
    fixture = copy.deepcopy(_fixture())
    fixture["allowed_matches"].append(dict(fixture["allowed_matches"][0]))
    with pytest.raises(ClaimSemanticsV2Error, match="duplicate match edge"):
        validate_fixture(fixture)


def test_duplicate_ids_fail_closed():
    fixture = copy.deepcopy(_fixture())
    fixture["gold_atoms"].append(copy.deepcopy(fixture["gold_atoms"][0]))
    with pytest.raises(ClaimSemanticsV2Error, match="duplicate gold id"):
        validate_fixture(fixture)

    fixture = copy.deepcopy(_fixture())
    fixture["candidates"].append(copy.deepcopy(fixture["candidates"][0]))
    with pytest.raises(ClaimSemanticsV2Error, match="duplicate candidate"):
        validate_fixture(fixture)


def test_invalid_and_stale_draft_sha_fail_closed():
    fixture = copy.deepcopy(_fixture())
    fixture["draft_sha256"] = "0" * 64
    with pytest.raises(ClaimSemanticsV2Error, match="stale or invalid"):
        validate_fixture(fixture)

    mutated = copy.deepcopy(_fixture())
    mutated["draft_text"] += " "
    with pytest.raises(ClaimSemanticsV2Error, match="stale or invalid"):
        validate_fixture(mutated)


def test_invalid_span_and_text_mismatch_fail_closed():
    fixture = copy.deepcopy(_fixture())
    fixture["gold_atoms"][0]["span"] = [10, 4]
    with pytest.raises(ClaimSemanticsV2Error, match="empty or inverted"):
        validate_fixture(fixture)

    mismatch = copy.deepcopy(_fixture())
    mismatch["gold_atoms"][0]["span"] = [0, 4]
    with pytest.raises(ClaimSemanticsV2Error, match="does not match text"):
        validate_fixture(mismatch)


def test_missing_required_evidence_identity_fails_closed():
    fixture = copy.deepcopy(_fixture())
    fixture["evidence_bindings"][0]["evidence_id"] = ""
    with pytest.raises(ClaimSemanticsV2Error, match="evidence_id required"):
        validate_fixture(fixture)

    missing = copy.deepcopy(_fixture())
    missing["evidence_bindings"] = [
        binding for binding in missing["evidence_bindings"] if binding["gold_id"] != "g1"
    ]
    metrics = evaluate_fixture(missing)["metrics"]
    _ratio(metrics["material_claim_unresolved_rate.v1"], 4, 4)


def test_invalid_evidence_binding_marks_unresolved():
    fixture = copy.deepcopy(_fixture())
    fixture["evidence_bindings"][0]["valid"] = False
    result = evaluate_fixture(fixture)
    assert result["oracle"]["semantic_pass"] is False
    _ratio(result["metrics"]["material_claim_unresolved_rate.v1"], 4, 4)


def test_zero_material_denominator_produces_na_not_zero():
    fixture = copy.deepcopy(_fixture())
    fixture["gold_atoms"] = [gold for gold in fixture["gold_atoms"] if not gold["material"]]
    fixture["allowed_matches"] = [
        edge for edge in fixture["allowed_matches"] if edge["gold_id"] == "g5"
    ]
    fixture["evidence_bindings"] = [
        binding for binding in fixture["evidence_bindings"] if binding["gold_id"] == "g5"
    ]
    metrics = evaluate_fixture(fixture)["metrics"]
    _undefined(metrics["material_claim_recall.v2"], "zero material gold atoms")
    _undefined(metrics["material_claim_unresolved_rate.v1"], "zero material gold atoms")


def test_zero_classifier_negative_denominator_produces_na():
    fixture = copy.deepcopy(_fixture())
    for gold in fixture["gold_atoms"]:
        if gold["material"]:
            gold["gold_semantic_status"] = "verified"
    metrics = evaluate_fixture(fixture)["metrics"]
    _undefined(
        metrics["material_false_verification_rate.v1"],
        "zero eligible material weak-or-unverified gold cases",
    )


def test_zero_oracle_fail_denominator_produces_na():
    fixture = copy.deepcopy(_fixture())
    fixture["automatic_route"] = {"decision": "FAIL"}
    fixture["candidates"].append({
        "id": "c2",
        "canonical_id": "ADV01.C.batchnorm_infer",
        "text": "Batch normalization uses running mean and variance during inference.",
        "span": [44, 112],
        "roles": [],
        "predicted_semantic_status": "verified",
    })
    fixture["allowed_matches"].append({"candidate_id": "c2", "gold_id": "g2"})
    fixture["evidence_bindings"].append({
        "id": "ADV01.BIND.g2",
        "gold_id": "g2",
        "candidate_id": "c2",
        "evidence_id": "ADV01.EV.bn_spec",
        "valid": True,
        "fully_entailed": True,
    })
    for gold in fixture["gold_atoms"]:
        if gold["material"]:
            gold["gold_semantic_status"] = "verified"
    for candidate in fixture["candidates"]:
        if candidate["id"] != "cx":
            candidate["predicted_semantic_status"] = "verified"
    for binding in fixture["evidence_bindings"]:
        binding["valid"] = True
        binding["fully_entailed"] = True
    pack = {
        "pack_id": "claim_semantics_v2",
        "schema_version": 2,
        "evaluator_id": "claim_semantics_v2",
        "description": "resolved variant",
        "fixtures": [fixture],
    }
    report = evaluate_pack(pack, require_frozen_catalog=False)
    _undefined(
        report["metrics"]["automatic_semantic_false_pass_rate.v1"],
        "zero oracle-failing assets",
    )


def test_ordering_does_not_change_results():
    baseline = evaluate_fixture(_fixture())
    shuffled = copy.deepcopy(_fixture())
    shuffled["candidates"] = list(reversed(shuffled["candidates"]))
    shuffled["allowed_matches"] = list(reversed(shuffled["allowed_matches"]))
    shuffled["gold_atoms"] = list(reversed(shuffled["gold_atoms"]))
    shuffled["evidence_bindings"] = list(reversed(shuffled["evidence_bindings"]))
    shuffled["verifier_rows"] = list(reversed(shuffled["verifier_rows"]))
    assert evaluate_fixture(shuffled)["metrics"] == baseline["metrics"]


def test_repeated_evaluation_is_byte_deterministic():
    first = normalize_json(evaluate_pack(_pack()))
    second = normalize_json(evaluate_pack(load_pack(DEFAULT_FIXTURES)))
    assert first == second
    assert first.endswith("\n")
    json.loads(first)


def test_tampering_without_identity_update_fails():
    fixture = copy.deepcopy(_fixture())
    fixture["gold_atoms"][0]["text"] = fixture["gold_atoms"][0]["text"] + " "
    with pytest.raises(ClaimSemanticsV2Error, match="does not match text"):
        validate_fixture(fixture)


def test_invention_candidate_cannot_earn_recall_credit():
    result = evaluate_fixture(_fixture())
    matched_candidates = {candidate_id for candidate_id, _gold_id in result["matching"]["pairs"]}
    assert "cx" not in matched_candidates
    _ratio(_metrics()["material_claim_recall.v2"], 3, 4)


def test_qualifier_loss_edge_fails_closed():
    fixture = copy.deepcopy(_fixture())
    fixture["candidates"].append({
        "id": "ql",
        "canonical_id": "ADV01.C.ql",
        "text": "ResNet-50 contains parameters.",
        "span": None,
        "roles": ["qualifier_loss"],
        "predicted_semantic_status": "verified",
    })
    fixture["allowed_matches"].append({"candidate_id": "ql", "gold_id": "g1"})
    with pytest.raises(ClaimSemanticsV2Error, match="forbidden-role allowed edge"):
        validate_fixture(fixture)


def test_compound_and_fragment_edges_fail_closed():
    for role in ("compound", "fragment"):
        fixture = copy.deepcopy(_fixture())
        fixture["candidates"].append({
            "id": role,
            "canonical_id": f"ADV01.C.{role}",
            "text": "compound fragment text",
            "span": None,
            "roles": [role],
            "predicted_semantic_status": "verified",
        })
        fixture["allowed_matches"].append({"candidate_id": role, "gold_id": "g1"})
        with pytest.raises(ClaimSemanticsV2Error, match="forbidden-role allowed edge"):
            validate_fixture(fixture)


def test_matching_is_maximum_cardinality_with_id_tie_break():
    pairs = maximum_cardinality_matching([
        ("C1", "G1"),
        ("C1", "G2"),
        ("C2", "G1"),
    ])
    assert pairs == [("C1", "G2"), ("C2", "G1")]


def test_draft_hashes_match_utf8_sha256():
    for fixture in _pack()["fixtures"]:
        assert fixture["draft_sha256"] == sha256_text(fixture["draft_text"])


def test_evaluator_does_not_import_runtime_or_providers():
    source = Path("scripts/evaluate_claim_semantics_v2.py").read_text(encoding="utf-8")
    forbidden = (
        "agent.nodes",
        "openai",
        "tavily",
        "qdrant",
        "sentence_transformers",
        "urllib.request",
        "requests",
        "httpx",
        "socket",
        "jsonschema",
    )
    for name in forbidden:
        assert name not in source


def test_official_pack_satisfies_frozen_schema_and_parity():
    pack = json.loads(DEFAULT_FIXTURES.read_text(encoding="utf-8"))
    schema = load_frozen_schema(DEFAULT_SCHEMA)
    assert_schema_runtime_parity(schema)
    validate_against_frozen_schema(pack, schema)
    load_pack(DEFAULT_FIXTURES)
    assert DEFAULT_REGISTRY.is_file()


def test_schema_runtime_contract_drift_is_detected():
    drifted = copy.deepcopy(load_frozen_schema())
    drifted["properties"]["pack_id"]["const"] = "drifted"
    with pytest.raises(ClaimSemanticsV2Error, match="schema/runtime contract drift"):
        assert_schema_runtime_parity(drifted)


def test_metric_registry_validates_against_contract_schema():
    registry = json.loads(Path("evals/metric_registry_v1.json").read_text(encoding="utf-8"))
    schema = json.loads(Path("evals/metric_contract_v1.schema.json").read_text(encoding="utf-8"))
    validate_against_frozen_schema(registry, schema, label="registry")


def test_evaluator_metric_ids_match_registry():
    registry = load_metric_registry()
    registry_names = {metric["canonical_name"] for metric in registry["metrics"]}
    report = evaluate_pack(_pack())
    evaluator_names = set(report["results"][0]["metrics"])
    assert evaluator_names == registry_names


def test_registry_hand_results_match_evaluator():
    registry = load_metric_registry()
    metrics = _metrics()
    report = evaluate_pack(_pack())
    mapping = {
        "material_claim_recall.v2": metrics["material_claim_recall.v2"],
        "material_claim_unresolved_rate.v1": metrics["material_claim_unresolved_rate.v1"],
        "material_false_verification_rate.v1": metrics["material_false_verification_rate.v1"],
        "automatic_semantic_false_pass_rate.v1": report["metrics"]["automatic_semantic_false_pass_rate.v1"],
        "unverified_verifier_row_rate.UVR_v1": metrics["unverified_verifier_row_rate.UVR_v1"],
    }
    for entry in registry["metrics"]:
        expected = entry["hand_calculated_expected_result"]
        actual = mapping[entry["canonical_name"]]
        assert actual["numerator"] == expected["numerator"]
        assert actual["denominator"] == expected["denominator"]
        assert actual["value"] == expected["value"]
        assert actual["undefined"] == expected["undefined"]
