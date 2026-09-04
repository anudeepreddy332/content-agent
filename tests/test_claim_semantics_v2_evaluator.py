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
    DEFAULT_F02_FIXTURES,
    DEFAULT_F02_MANIFEST,
    DEFAULT_REGISTRY,
    DEFAULT_SCHEMA,
    MATCH_INELIGIBLE_ROLES,
    REGISTERED_METRIC_NAMES,
    assert_fixture_matches_approved_identity,
    assert_schema_runtime_parity,
    evaluate_fixture,
    evaluate_pack,
    fixture_identity_sha256,
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


def test_registry_has_exactly_eight_identities():
    registry = load_metric_registry()
    names = {metric["canonical_name"] for metric in registry["metrics"]}
    assert names == REGISTERED_METRIC_NAMES
    assert len(registry["metrics"]) == 8
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
    drifted["properties"]["pack_id"]["enum"] = ["drifted"]
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


def _adv01_g3_material_mutated(material: object) -> dict:
    pack = copy.deepcopy(_pack())
    for gold in pack["fixtures"][0]["gold_atoms"]:
        if gold["id"] == "g3":
            gold["material"] = material
    return pack


def _adv01_g3_material_missing() -> dict:
    pack = copy.deepcopy(_pack())
    for gold in pack["fixtures"][0]["gold_atoms"]:
        if gold["id"] == "g3":
            del gold["material"]
    return pack


def _adv01_invalid_nested_field() -> dict:
    pack = copy.deepcopy(_pack())
    pack["fixtures"][0]["automatic_route"]["decision"] = "MAYBE"
    return pack


def test_f01_null_material_rejects_direct_evaluate_pack_without_metrics():
    pack = _adv01_g3_material_mutated(None)
    with pytest.raises(ClaimSemanticsV2Error):
        evaluate_pack(pack)


def test_f01_null_material_rejects_loader_and_direct_call_equivalently():
    pack = _adv01_g3_material_mutated(None)
    with pytest.raises(ClaimSemanticsV2Error, match="invalid type"):
        validate_against_frozen_schema(pack)
    with pytest.raises(ClaimSemanticsV2Error):
        evaluate_pack(pack)


def test_direct_call_and_loader_parity_reject_malformed_structures():
    cases = [
        ("material_null", _adv01_g3_material_mutated(None)),
        ("material_string", _adv01_g3_material_mutated("true")),
        ("material_missing", _adv01_g3_material_missing()),
        ("invalid_nested", _adv01_invalid_nested_field()),
    ]
    for label, pack in cases:
        with pytest.raises(ClaimSemanticsV2Error, match="."):
            validate_against_frozen_schema(pack)
        with pytest.raises(ClaimSemanticsV2Error, match="."):
            evaluate_pack(pack)


@pytest.mark.parametrize(
    ("material", "valid"),
    [
        (None, False),
        (0, False),
        (1, False),
        ("true", False),
        ("false", False),
        ("", False),
        ([], False),
        ({}, False),
        (True, True),
        (False, True),
    ],
)
def test_strict_boolean_material_contract(material: object, valid: bool):
    pack = _adv01_g3_material_mutated(material)
    if valid:
        evaluate_pack(pack)
        return
    with pytest.raises(ClaimSemanticsV2Error):
        evaluate_pack(pack)


def test_registry_hand_results_match_evaluator():
    registry = load_metric_registry()
    metrics = _metrics()
    report = evaluate_pack(_pack())
    f02 = load_pack(DEFAULT_F02_FIXTURES)
    fc = evaluate_pack(f02, fixture_id="F-C")
    mapping = {
        "material_claim_recall.v2": metrics["material_claim_recall.v2"],
        "material_claim_unresolved_rate.v1": metrics["material_claim_unresolved_rate.v1"],
        "material_false_verification_rate.v1": metrics["material_false_verification_rate.v1"],
        "automatic_semantic_false_pass_rate.v1": report["metrics"]["automatic_semantic_false_pass_rate.v1"],
        "unverified_verifier_row_rate.UVR_v1": metrics["unverified_verifier_row_rate.UVR_v1"],
        "material_false_verification_rate.v2": fc["results"][0]["metrics"]["material_false_verification_rate.v2"],
        "final_material_claim_unresolved_rate.v1": fc["results"][0]["metrics"]["final_material_claim_unresolved_rate.v1"],
        "automatic_semantic_false_pass_rate.v2": fc["metrics"]["automatic_semantic_false_pass_rate.v2"],
    }
    for entry in registry["metrics"]:
        expected = entry["hand_calculated_expected_result"]
        actual = mapping[entry["canonical_name"]]
        assert actual["numerator"] == expected["numerator"]
        assert actual["denominator"] == expected["denominator"]
        assert actual["value"] == expected["value"]
        assert actual["undefined"] == expected["undefined"]


def _f02_pack() -> dict:
    return load_pack(DEFAULT_F02_FIXTURES)


def _f02_fixture(fixture_id: str, pack: dict | None = None) -> dict:
    pack = pack or _f02_pack()
    return copy.deepcopy(next(item for item in pack["fixtures"] if item["id"] == fixture_id))


def _approved_fc_digest() -> str:
    manifest = json.loads(DEFAULT_F02_MANIFEST.read_text(encoding="utf-8"))
    return manifest["fixture_identities"]["F-C"]


def test_f02_fa_through_ff_oracle_table():
    expected = {
        "F-A": True,
        "F-B": False,
        "F-C": False,
        "F-D": True,
        "F-E": False,
        "F-F": True,
    }
    pack = _f02_pack()
    report = evaluate_pack(pack)
    by_id = {item["fixture_id"]: item for item in report["results"]}
    for fixture_id, semantic_pass in expected.items():
        assert by_id[fixture_id]["oracle"]["semantic_pass"] is semantic_pass


def test_f02_fc_fails_while_required_recall_remains_complete():
    result = evaluate_fixture(_f02_fixture("F-C"))
    _ratio(result["metrics"]["material_claim_recall.v2"], 1, 1)
    _ratio(result["metrics"]["material_claim_unresolved_rate.v1"], 0, 1)
    _ratio(result["metrics"]["final_material_claim_unresolved_rate.v1"], 1, 2)
    _ratio(result["metrics"]["material_false_verification_rate.v2"], 1, 1)
    assert result["oracle"]["semantic_pass"] is False
    assert result["oracle"]["required_semantic_pass"] is True
    pack = {
        "pack_id": "claim_semantics_v2_f02",
        "schema_version": 2,
        "schema_development_revision": "f02-r1",
        "evaluator_id": "claim_semantics_v2",
        "description": "F-C unit",
        "fixtures": [_f02_fixture("F-C")],
    }
    report = evaluate_pack(pack, require_frozen_catalog=False)
    _ratio(report["metrics"]["automatic_semantic_false_pass_rate.v2"], 1, 1)


def test_f02_fd_passes_with_supported_unmatched_claim():
    result = evaluate_fixture(_f02_fixture("F-D"))
    assert result["oracle"]["semantic_pass"] is True
    unmatched = [atom for atom in result["final_claims"] if atom["reference_relationship"] == "unmatched"]
    assert len(unmatched) == 1
    assert unmatched[0]["resolved"] is True
    _ratio(result["metrics"]["final_material_claim_unresolved_rate.v1"], 0, 2)


def test_f02_fe_fails_without_false_verification():
    result = evaluate_fixture(_f02_fixture("F-E"))
    assert result["oracle"]["semantic_pass"] is False
    _ratio(result["metrics"]["material_claim_recall.v2"], 1, 1)
    _ratio(result["metrics"]["material_false_verification_rate.v2"], 0, 1)
    _ratio(result["metrics"]["final_material_claim_unresolved_rate.v1"], 1, 2)


def test_f02_ff_passes_and_keeps_nonmaterial_weak_visible():
    result = evaluate_fixture(_f02_fixture("F-F"))
    assert result["oracle"]["semantic_pass"] is True
    visible = [atom for atom in result["final_claims"] if atom["id"] == "f2"]
    assert visible == [
        {
            "id": "f2",
            "text": "The catalog looks tidy.",
            "material": False,
            "reference_relationship": "unmatched",
            "independent_semantic_label": "weak",
            "predicted_semantic_status": "weak",
            "resolved": None,
        }
    ]


def test_f02_removing_fc_candidate_still_fails():
    fixture = _f02_fixture("F-C")
    fixture["candidates"] = [candidate for candidate in fixture["candidates"] if candidate["id"] != "c100"]
    result = evaluate_fixture(fixture)
    assert result["oracle"]["semantic_pass"] is False
    _ratio(result["metrics"]["final_material_claim_unresolved_rate.v1"], 1, 2)


def test_f02_changing_or_removing_fc_role_still_fails():
    fixture = _f02_fixture("F-C")
    for candidate in fixture["candidates"]:
        if candidate["id"] == "c100":
            candidate["roles"] = ["invention"]
    result = evaluate_fixture(fixture)
    assert result["oracle"]["semantic_pass"] is False

    removed = _f02_fixture("F-C")
    for candidate in removed["candidates"]:
        if candidate["id"] == "c100":
            candidate["roles"] = []
    result = evaluate_fixture(removed)
    assert result["oracle"]["semantic_pass"] is False


def test_f02_verified_final_claim_invalid_binding_fails():
    fixture = _f02_fixture("F-A")
    fixture["final_atoms"][0]["binding"]["valid"] = False
    result = evaluate_fixture(fixture)
    assert result["oracle"]["semantic_pass"] is False
    _ratio(result["metrics"]["final_material_claim_unresolved_rate.v1"], 1, 1)


def test_f02_fb_omitted_required_claim_is_absent_from_f():
    fixture = _f02_fixture("F-B")
    assert fixture["gold_atoms"][0]["text"] == "There are 10 items."
    assert "There are 10 items." not in fixture["draft_text"]
    assert fixture["final_atoms"] == []
    result = evaluate_fixture(fixture)
    _ratio(result["metrics"]["material_claim_recall.v2"], 0, 1)
    _ratio(result["metrics"]["material_claim_unresolved_rate.v1"], 1, 1)
    _undefined(
        result["metrics"]["final_material_claim_unresolved_rate.v1"],
        "zero independently adjudicated final material claims",
    )
    assert result["oracle"]["unresolved_material_atoms"] == 1
    assert result["oracle"]["unresolved_final_material_atoms"] == 0
    assert result["oracle"]["final_material_atoms"] == 0
    assert result["oracle"]["semantic_pass"] is False
    assert result["final_claims"] == []
    required_cases = [case for case in result["classification_cases"] if case["source"] == "required"]
    assert required_cases
    assert required_cases[0]["required_gold_id"] == "g1"
    assert required_cases[0]["final_atom_id"] is None


def test_f02_semantic_pass_does_not_require_positive_denominators():
    fixture = _f02_fixture("F-A")
    fixture["gold_atoms"] = []
    fixture["candidates"] = []
    fixture["allowed_matches"] = []
    fixture["evidence_bindings"] = []
    fixture["final_atoms"] = []
    fixture["fixed_classification_cases"] = []
    fixture["verifier_rows"] = []
    result = evaluate_fixture(fixture)
    assert result["oracle"]["semantic_pass"] is True
    assert result["oracle"]["unresolved_material_atoms"] == 0
    assert result["oracle"]["unresolved_final_material_atoms"] == 0
    _undefined(result["metrics"]["material_claim_recall.v2"], "zero material gold atoms")
    _undefined(result["metrics"]["material_claim_unresolved_rate.v1"], "zero material gold atoms")
    _undefined(
        result["metrics"]["final_material_claim_unresolved_rate.v1"],
        "zero independently adjudicated final material claims",
    )
    _undefined(
        result["metrics"]["material_false_verification_rate.v2"],
        "zero independently frozen material weak-or-unverified classification cases",
    )


def test_f02_fv_v2_uses_fixed_catalog_not_final_atoms():
    fixture = _f02_fixture("F-A")
    fixture["final_atoms"][0]["independent_semantic_label"] = "unverified"
    result = evaluate_fixture(fixture)
    assert result["oracle"]["semantic_pass"] is False
    _undefined(
        result["metrics"]["material_false_verification_rate.v2"],
        "zero independently frozen material weak-or-unverified classification cases",
    )

    catalog = _f02_fixture("F-A")
    catalog["fixed_classification_cases"][0]["independent_semantic_label"] = "unverified"
    catalog["fixed_classification_cases"][0]["predicted_semantic_status"] = "verified"
    result = evaluate_fixture(catalog)
    _ratio(result["metrics"]["material_false_verification_rate.v2"], 1, 1)


def test_f02_omitted_required_classifier_case_does_not_depend_on_f():
    fixture = _f02_fixture("F-B")
    assert fixture["final_atoms"] == []
    case = fixture["fixed_classification_cases"][0]
    assert case["source"] == "required"
    assert case["required_gold_id"] == "g1"
    assert case["final_atom_id"] is None
    case["independent_semantic_label"] = "unverified"
    case["predicted_semantic_status"] = "verified"
    result = evaluate_fixture(fixture)
    assert result["oracle"]["semantic_pass"] is False
    assert result["oracle"]["unresolved_final_material_atoms"] == 0
    _ratio(result["metrics"]["material_false_verification_rate.v2"], 1, 1)
    assert any(
        item["id"] == "FB.CLASS.g1" and item["final_atom_id"] is None
        for item in result["classification_cases"]
    )


def test_f02_unsafe_final_claim_with_no_required_gold_fails():
    fixture = _f02_fixture("F-C")
    fixture["gold_atoms"] = []
    fixture["allowed_matches"] = []
    fixture["evidence_bindings"] = []
    fixture["final_atoms"] = [
        atom for atom in fixture["final_atoms"] if atom["id"] == "f2"
    ]
    fixture["fixed_classification_cases"] = [
        case for case in fixture["fixed_classification_cases"] if case["source"] == "unmatched-final"
    ]
    result = evaluate_fixture(fixture)
    assert result["oracle"]["semantic_pass"] is False
    _undefined(result["metrics"]["material_claim_recall.v2"], "zero material gold atoms")
    _ratio(result["metrics"]["final_material_claim_unresolved_rate.v1"], 1, 1)


def test_f02_deleting_or_relabeling_unsafe_atom_breaks_approved_identity():
    approved = _approved_fc_digest()
    original = _f02_fixture("F-C")
    assert_fixture_matches_approved_identity(original, approved)

    deleted = _f02_fixture("F-C")
    deleted["final_atoms"] = [atom for atom in deleted["final_atoms"] if atom["id"] != "f2"]
    with pytest.raises(ClaimSemanticsV2Error, match="approved qualification digest"):
        assert_fixture_matches_approved_identity(deleted, approved)

    relabeled = _f02_fixture("F-C")
    for atom in relabeled["final_atoms"]:
        if atom["id"] == "f2":
            atom["material"] = False
    with pytest.raises(ClaimSemanticsV2Error, match="approved qualification digest"):
        assert_fixture_matches_approved_identity(relabeled, approved)
    assert fixture_identity_sha256(deleted) != approved


def _tampered_f02_pack() -> dict:
    return copy.deepcopy(_f02_pack())


def _fc_index(pack: dict) -> int:
    return next(index for index, fixture in enumerate(pack["fixtures"]) if fixture["id"] == "F-C")


def test_f02_direct_evaluate_pack_rejects_deleted_unsafe_fc_atom():
    pack = _tampered_f02_pack()
    fixture = pack["fixtures"][_fc_index(pack)]
    fixture["final_atoms"] = [atom for atom in fixture["final_atoms"] if atom["id"] != "f2"]
    with pytest.raises(ClaimSemanticsV2Error, match="F-02 fixture F-C identity"):
        evaluate_pack(pack)


def test_f02_direct_evaluate_pack_rejects_fc_materiality_flip():
    pack = _tampered_f02_pack()
    fixture = pack["fixtures"][_fc_index(pack)]
    for atom in fixture["final_atoms"]:
        if atom["id"] == "f2":
            atom["material"] = False
    with pytest.raises(ClaimSemanticsV2Error, match="F-02 fixture F-C identity"):
        evaluate_pack(pack)


def test_f02_direct_evaluate_pack_rejects_fc_semantic_label_change():
    pack = _tampered_f02_pack()
    fixture = pack["fixtures"][_fc_index(pack)]
    for atom in fixture["final_atoms"]:
        if atom["id"] == "f2":
            atom["independent_semantic_label"] = "verified"
    with pytest.raises(ClaimSemanticsV2Error, match="F-02 fixture F-C identity"):
        evaluate_pack(pack)


def test_f02_direct_evaluate_pack_rejects_fc_binding_relationship_change():
    pack = _tampered_f02_pack()
    fixture = pack["fixtures"][_fc_index(pack)]
    for atom in fixture["final_atoms"]:
        if atom["id"] == "f2":
            atom["reference_relationship"] = "required-equivalent"
            atom["required_gold_id"] = "g1"
    with pytest.raises(ClaimSemanticsV2Error, match="F-02 fixture F-C identity"):
        evaluate_pack(pack)


def test_f02_manifest_boundary_loader_and_evaluate_pack_parity(tmp_path):
    pack = _tampered_f02_pack()
    fixture = pack["fixtures"][_fc_index(pack)]
    fixture["final_atoms"] = [atom for atom in fixture["final_atoms"] if atom["id"] != "f2"]

    with pytest.raises(ClaimSemanticsV2Error, match="F-02 fixture F-C identity"):
        evaluate_pack(pack)

    tampered_path = tmp_path / "claim_semantics_v2_f02_tampered.json"
    tampered_path.write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(ClaimSemanticsV2Error):
        load_pack(tampered_path, require_frozen_catalog=True)


def test_f02_require_frozen_catalog_false_does_not_bypass_official_manifest():
    pack = _tampered_f02_pack()
    fixture = pack["fixtures"][_fc_index(pack)]
    fixture["final_atoms"] = [atom for atom in fixture["final_atoms"] if atom["id"] != "f2"]
    with pytest.raises(ClaimSemanticsV2Error, match="F-02 fixture F-C identity"):
        evaluate_pack(pack, require_frozen_catalog=False)


def test_f02_evaluate_fixture_does_not_claim_official_qualification_pack():
    """Single-fixture evaluate_fixture uses a one-fixture envelope, not the approved six-fixture catalog."""
    fixture = _f02_fixture("F-C")
    fixture["final_atoms"] = [atom for atom in fixture["final_atoms"] if atom["id"] != "f2"]
    result = evaluate_fixture(fixture)
    assert result["oracle"]["semantic_pass"] is True


def test_f02_final_atom_malformed_materiality_fails_closed():
    for material in (None, 0, 1, "true", [], {}):
        fixture = _f02_fixture("F-C")
        fixture["final_atoms"][1]["material"] = material
        with pytest.raises(ClaimSemanticsV2Error):
            evaluate_fixture(fixture)


def test_f02_candidate_and_final_atom_ordering_does_not_change_results():
    baseline = evaluate_fixture(_f02_fixture("F-C"))
    shuffled = _f02_fixture("F-C")
    shuffled["candidates"] = list(reversed(shuffled["candidates"]))
    shuffled["final_atoms"] = list(reversed(shuffled["final_atoms"]))
    shuffled["fixed_classification_cases"] = list(reversed(shuffled["fixed_classification_cases"]))
    shuffled["allowed_matches"] = list(reversed(shuffled["allowed_matches"]))
    shuffled["gold_atoms"] = list(reversed(shuffled["gold_atoms"]))
    shuffled["evidence_bindings"] = list(reversed(shuffled["evidence_bindings"]))
    shuffled["verifier_rows"] = list(reversed(shuffled["verifier_rows"]))
    shuffled_result = evaluate_fixture(shuffled)
    assert shuffled_result["metrics"] == baseline["metrics"]
    assert shuffled_result["oracle"]["semantic_pass"] is baseline["oracle"]["semantic_pass"]


def test_historical_v1_and_uvr_remain_unchanged():
    metrics = _metrics()
    _ratio(metrics["material_claim_recall.v2"], 3, 4)
    _ratio(metrics["material_claim_unresolved_rate.v1"], 3, 4)
    _ratio(metrics["material_false_verification_rate.v1"], 1, 2)
    _ratio(metrics["unverified_verifier_row_rate.UVR_v1"], 0, 5)
    report = evaluate_pack(_pack())
    _ratio(report["metrics"]["automatic_semantic_false_pass_rate.v1"], 1, 1)
    _undefined(
        metrics["final_material_claim_unresolved_rate.v1"],
        "historical revisionless fixture has no final-claim inventory",
    )
    _undefined(
        metrics["material_false_verification_rate.v2"],
        "historical revisionless fixture has no final-claim inventory",
    )
    _undefined(
        report["metrics"]["automatic_semantic_false_pass_rate.v2"],
        "historical revisionless fixture has no final-claim inventory",
    )


def test_historical_fixture_bytes_are_not_reinterpreted():
    historical = Path("evals/fixtures/claim_semantics_v2.json").read_bytes()
    assert b"final_atoms" not in historical
    assert b"schema_development_revision" not in historical
    pack = json.loads(historical)
    assert pack["pack_id"] == "claim_semantics_v2"
    assert "schema_development_revision" not in pack
