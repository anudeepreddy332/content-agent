"""Deterministic coverage for the Slice 2A claim-semantics oracle. No providers."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_claim_semantics import (
    ClaimSemanticsError,
    DEFAULT_FIXTURES,
    FROZEN_FIXTURE_IDS,
    evaluate_candidate_set,
    evaluate_pack,
    load_pack,
    maximum_cardinality_matching,
    normalize_json,
    sha256_text,
    validate_fixture,
    validate_pack,
)


def _pack() -> dict:
    return load_pack(DEFAULT_FIXTURES)


def _fixture(pack: dict, fixture_id: str) -> dict:
    return next(item for item in pack["fixtures"] if item["id"] == fixture_id)


def _set(fixture: dict, set_id: str) -> dict:
    return next(item for item in fixture["candidate_sets"] if item["id"] == set_id)


def _eval(pack: dict, fixture_id: str, set_id: str) -> dict:
    return evaluate_candidate_set(_fixture(pack, fixture_id), _set(_fixture(pack, fixture_id), set_id))


def _metrics(pack: dict, fixture_id: str, set_id: str) -> dict:
    return _eval(pack, fixture_id, set_id)["metrics"]


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
    if "numerator" in metric and metric["numerator"] is not None:
        assert metric["numerator"] == numerator
        assert metric["denominator"] == denominator


def test_official_pack_has_fourteen_fixtures_and_eighteen_canonical_gold_atoms():
    pack = _pack()
    report = evaluate_pack(pack)
    assert tuple(item["id"] for item in pack["fixtures"]) == FROZEN_FIXTURE_IDS
    assert report["fixture_count"] == 14
    assert report["gold_atom_count"] == 18
    assert report["material_gold_atom_count"] == 18
    assert report["factual_gold_atom_count"] == 18
    assert sum(len({gold["canonical_id"] for gold in fixture["gold_atoms"]}) for fixture in pack["fixtures"]) == 18
    assert sum(len(fixture["gold_atoms"]) for fixture in pack["fixtures"]) == 20


def test_perfect_extraction_is_complete_and_precise():
    pack = _pack()
    expected = {
        "F01": (2, 2),
        "F03": (2, 2),
        "F04": (2, 2),
        "F05": (1, 1),
        "F06": (3, 3),
        "F08": (1, 1),
        "F10": (2, 2),
        "F11": (1, 1),
        "F13": (1, 1),
        "F14": (1, 1),
    }
    for fixture_id, (num, den) in expected.items():
        metrics = _metrics(pack, fixture_id, "perfect")
        _ratio(metrics["material_claim_recall"], num, den)
        _ratio(metrics["full_factual_recall"], num, den)
        _ratio(metrics["extraction_precision"], num, den)
        _ratio(metrics["extraction_f1"], 2 * num * num, num * den + num * den)
        assert metrics["duplicate_count"] == 0
        assert metrics["atomicity_violations"] == 0
        assert metrics["fragmentation_violations"] == 0


def test_removing_one_candidate_changes_only_omission_metrics():
    pack = _pack()
    perfect = _eval(pack, "F01", "perfect")
    omission = _eval(pack, "F01", "omission")
    _ratio(perfect["metrics"]["material_claim_recall"], 2, 2)
    _ratio(omission["metrics"]["material_claim_recall"], 1, 2)
    _ratio(omission["metrics"]["full_factual_recall"], 1, 2)
    _ratio(omission["metrics"]["extraction_precision"], 1, 1)
    _ratio(omission["metrics"]["extraction_f1"], 2, 3)
    assert omission["matching"]["matched_gold_canonical_ids"] == ["F01.ATOM.first_moment"]
    assert omission["metrics"]["duplicate_count"] == perfect["metrics"]["duplicate_count"] == 0
    assert omission["metrics"]["atomicity_violations"] == 0
    assert omission["metrics"]["fragmentation_violations"] == 0


def test_duplicate_rows_cannot_improve_recall():
    pack = _pack()
    perfect = _metrics(pack, "F02", "perfect")
    duplicate = _metrics(pack, "F02", "duplicate")
    _ratio(perfect["material_claim_recall"], 1, 1)
    _ratio(duplicate["material_claim_recall"], 1, 1)
    _ratio(duplicate["full_factual_recall"], 1, 1)
    assert duplicate["duplicate_count"] == 1
    _ratio(duplicate["duplicate_rate"], 1, 2)


def test_paraphrase_duplicates_cannot_inflate_denominator():
    pack = _pack()
    fixture = _fixture(pack, "F02")
    assert len(fixture["gold_atoms"]) == 2
    assert len({gold["canonical_id"] for gold in fixture["gold_atoms"]}) == 1
    perfect = _metrics(pack, "F02", "perfect")
    duplicate = _metrics(pack, "F02", "duplicate")
    _ratio(perfect["extraction_precision"], 1, 1)
    _ratio(duplicate["extraction_precision"], 1, 1)
    twelve = _metrics(pack, "F12", "duplicate")
    _ratio(twelve["full_factual_recall"], 1, 1)
    _ratio(twelve["extraction_precision"], 1, 1)
    assert twelve["duplicate_count"] == 1


def test_one_compound_candidate_cannot_match_multiple_atoms():
    pack = _pack()
    metrics = _metrics(pack, "F03", "compound")
    _ratio(metrics["material_claim_recall"], 0, 2)
    _ratio(metrics["full_factual_recall"], 0, 2)
    _ratio(metrics["extraction_precision"], 0, 1)
    assert metrics["atomicity_violations"] == 1
    assert _eval(pack, "F03", "compound")["matching"]["pairs"] == []

    two_edge_pairs = maximum_cardinality_matching([
        ("F03.compound.C1", "F03.G.masks"),
        ("F03.compound.C1", "F03.G.calibration"),
    ])
    assert len(two_edge_pairs) == 1
    assert {gold_id for _candidate_id, gold_id in two_edge_pairs} <= {
        "F03.G.masks",
        "F03.G.calibration",
    }


def test_multiple_fragments_cannot_fake_one_complete_atom():
    pack = _pack()
    four = _eval(pack, "F04", "fragmentation")
    thirteen = _eval(pack, "F13", "fragmentation")
    _ratio(four["metrics"]["material_claim_recall"], 0, 2)
    _ratio(thirteen["metrics"]["full_factual_recall"], 0, 1)
    assert four["metrics"]["fragmentation_violations"] == 2
    assert thirteen["metrics"]["fragmentation_violations"] == 2
    assert four["matching"]["pairs"] == []
    assert thirteen["matching"]["pairs"] == []

    adversarial = copy.deepcopy(_set(_fixture(pack, "F13"), "fragmentation"))
    adversarial["allowed_matches"] = [
        {"candidate_id": "F13.fragmentation.C1", "gold_id": "F13.G.saturation"},
        {"candidate_id": "F13.fragmentation.C2", "gold_id": "F13.G.saturation"},
    ]
    result = evaluate_candidate_set(_fixture(pack, "F13"), adversarial)
    assert result["matching"]["pairs"] == []
    _ratio(result["metrics"]["full_factual_recall"], 0, 1)


@pytest.mark.parametrize(
    ("fixture_id", "candidate_id"),
    [
        ("F05", "F05.qualifier_loss.C_date"),
        ("F05", "F05.qualifier_loss.C_number"),
        ("F05", "F05.qualifier_loss.C_unit"),
        ("F11", "F11.qualifier_loss.C_condition"),
        ("F10", "F10.qualifier_loss.C_negation"),
        ("F10", "F10.qualifier_loss.C_modality"),
        ("F14", "F14.qualifier_loss.C_baseline"),
        ("F06", None),
    ],
)
def test_dropping_qualifiers_breaks_equivalence(fixture_id: str, candidate_id: str | None):
    pack = _pack()
    if fixture_id == "F06":
        dropped = copy.deepcopy(_set(_fixture(pack, "F06"), "perfect"))
        dropped["id"] = "qualifier_loss"
        dropped["candidates"] = [
            {
                **dropped["candidates"][1],
                "id": "F06.qualifier_loss.C_number",
                "canonical_id": "F06.C.drop_width",
                "text": "The encoder has a hidden width of 76.",
                "span": None,
                "roles": ["qualifier_loss"],
            }
        ]
        dropped["allowed_matches"] = []
        result = evaluate_candidate_set(_fixture(pack, "F06"), dropped)
        _ratio(result["metrics"]["full_factual_recall"], 0, 3)
        _ratio(result["metrics"]["extraction_precision"], 0, 1)
        return
    metrics = _metrics(pack, fixture_id, "qualifier_loss")
    assert metrics["full_factual_recall"]["numerator"] == 0
    assert metrics["extraction_precision"]["numerator"] == 0
    ids = [candidate["id"] for candidate in _set(_fixture(pack, fixture_id), "qualifier_loss")["candidates"]]
    assert candidate_id in ids
    assert _set(_fixture(pack, fixture_id), "qualifier_loss")["allowed_matches"] == []


def test_hypothetical_cannot_become_factual_and_real_example_stays_gold():
    pack = _pack()
    fixture = _fixture(pack, "F08")
    gold_ids = {gold["id"] for gold in fixture["gold_atoms"]}
    exclusion_reasons = {item["reason"] for item in fixture["exclusions"]}
    assert gold_ids == {"F08.G.imagenet"}
    assert fixture["gold_atoms"][0]["factual"] is True
    assert fixture["gold_atoms"][0]["material"] is True
    assert "explicit_hypothetical" in exclusion_reasons
    invented = _eval(pack, "F08", "invention")
    _ratio(invented["metrics"]["full_factual_recall"], 0, 1)
    _ratio(invented["metrics"]["extraction_precision"], 0, 1)
    assert invented["matching"]["pairs"] == []
    perfect = _metrics(pack, "F08", "perfect")
    _ratio(perfect["full_factual_recall"], 1, 1)


def test_own_code_only_prose_stays_excluded():
    pack = _pack()
    fixture = _fixture(pack, "F09")
    assert fixture["gold_atoms"] == []
    assert fixture["exclusions"][0]["reason"] == "draft_own_code"
    invented = _metrics(pack, "F09", "invention")
    _undefined(invented["full_factual_recall"], "zero factual gold atoms")
    _undefined(invented["material_claim_recall"], "zero material gold atoms")
    _ratio(invented["extraction_precision"], 0, 1)


def test_empty_candidates_with_nonempty_gold_yield_defined_zero_recall():
    pack = _pack()
    metrics = _metrics(pack, "F01", "empty")
    _ratio(metrics["material_claim_recall"], 0, 2)
    _ratio(metrics["full_factual_recall"], 0, 2)
    _undefined(metrics["extraction_precision"], "zero canonical candidate claims")
    _undefined(metrics["extraction_f1"], "extraction precision undefined")
    _undefined(metrics["duplicate_rate"], "zero candidate rows")
    assert metrics["duplicate_count"] == 0


def test_zero_gold_fixture_is_explicitly_undefined():
    pack = _pack()
    empty = _metrics(pack, "F07", "empty")
    _undefined(empty["material_claim_recall"], "zero material gold atoms")
    _undefined(empty["full_factual_recall"], "zero factual gold atoms")
    _undefined(empty["extraction_precision"], "zero canonical candidate claims")
    _undefined(empty["extraction_f1"], "extraction precision undefined")
    invented = _metrics(pack, "F07", "invention")
    _undefined(invented["full_factual_recall"], "zero factual gold atoms")
    _ratio(invented["extraction_precision"], 0, 1)
    _undefined(invented["extraction_f1"], "full factual recall undefined")
    assert _fixture(pack, "F07")["exclusions"][0]["reason"] == "instruction_advice"
    assert _fixture(pack, "F07")["exclusions"][1]["reason"] == "explicit_hypothetical"


def test_invalid_or_stale_hashes_fail():
    pack = _pack()
    stale = copy.deepcopy(_fixture(pack, "F01"))
    stale["draft_sha256"] = "0" * 64
    with pytest.raises(ClaimSemanticsError, match="stale or invalid"):
        validate_fixture(stale)

    mutated = copy.deepcopy(_fixture(pack, "F01"))
    mutated["draft_text"] = mutated["draft_text"] + " "
    with pytest.raises(ClaimSemanticsError, match="stale or invalid"):
        validate_fixture(mutated)


def test_invalid_spans_fail():
    pack = _pack()

    inverted = copy.deepcopy(_fixture(pack, "F14"))
    inverted["gold_atoms"][0]["span"] = [10, 4]
    with pytest.raises(ClaimSemanticsError, match="empty or inverted"):
        validate_fixture(inverted)

    mismatch = copy.deepcopy(_fixture(pack, "F14"))
    mismatch["gold_atoms"][0]["span"] = [0, 4]
    with pytest.raises(ClaimSemanticsError, match="does not match text"):
        validate_fixture(mismatch)

    oob = copy.deepcopy(_fixture(pack, "F14"))
    oob["gold_atoms"][0]["span"] = [0, len(oob["draft_text"]) + 5]
    oob["gold_atoms"][0]["text"] = oob["draft_text"] + "xxxxx"
    with pytest.raises(ClaimSemanticsError, match="out of range"):
        validate_fixture(oob)


def test_unknown_ids_and_match_edges_fail():
    pack = _pack()

    unknown_gold = copy.deepcopy(_fixture(pack, "F01"))
    unknown_gold["candidate_sets"][0]["allowed_matches"][0]["gold_id"] = "F99.G.ghost"
    with pytest.raises(ClaimSemanticsError, match="unknown gold_id"):
        validate_fixture(unknown_gold)

    unknown_candidate = copy.deepcopy(_fixture(pack, "F01"))
    unknown_candidate["candidate_sets"][0]["allowed_matches"][0]["candidate_id"] = "ghost"
    with pytest.raises(ClaimSemanticsError, match="unknown candidate_id"):
        validate_fixture(unknown_candidate)

    unknown_fixture = copy.deepcopy(pack)
    unknown_fixture["fixtures"][0]["id"] = "F99"
    with pytest.raises(ClaimSemanticsError, match="unknown fixture id"):
        validate_pack(unknown_fixture)

    with pytest.raises(ClaimSemanticsError, match="unknown fixture id"):
        evaluate_pack(pack, fixture_id="F99")

    with pytest.raises(ClaimSemanticsError, match="unknown candidate_set id"):
        evaluate_pack(pack, candidate_set_id="not_a_set")


def test_ordering_does_not_change_metrics():
    pack = _pack()
    perfect = _eval(pack, "F01", "perfect")
    reordered = _eval(pack, "F01", "reordered")
    assert perfect["metrics"] == reordered["metrics"]
    assert perfect["matching"]["pairs"] == reordered["matching"]["pairs"]


def test_repeated_evaluation_is_byte_identical():
    pack = _pack()
    first = normalize_json(evaluate_pack(pack))
    second = normalize_json(evaluate_pack(load_pack(DEFAULT_FIXTURES)))
    assert first == second
    assert first.endswith("\n")
    json.loads(first)


def test_both_zero_precision_and_recall_leave_f1_undefined():
    pack = _pack()
    metrics = _metrics(pack, "F03", "compound")
    _ratio(metrics["extraction_precision"], 0, 1)
    _ratio(metrics["full_factual_recall"], 0, 2)
    _undefined(
        metrics["extraction_f1"],
        "harmonic mean undefined when precision and recall are both 0",
        numerator=0,
        denominator=0,
    )


def test_matching_is_maximum_cardinality_with_id_tie_break():
    pairs = maximum_cardinality_matching([
        ("C1", "G1"),
        ("C1", "G2"),
        ("C2", "G1"),
    ])
    assert pairs == [("C1", "G2"), ("C2", "G1")]
    assert maximum_cardinality_matching([("C2", "G1"), ("C1", "G1")]) == [("C1", "G1")]


def test_draft_hashes_match_utf8_sha256():
    pack = _pack()
    for fixture in pack["fixtures"]:
        assert fixture["draft_sha256"] == sha256_text(fixture["draft_text"])


def test_evaluator_does_not_import_runtime_or_providers():
    source = Path("scripts/evaluate_claim_semantics.py").read_text(encoding="utf-8")
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
    )
    for name in forbidden:
        assert name not in source
