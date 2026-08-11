import pytest
from pydantic import ValidationError
from agent.nodes import _parse_verifier_verdicts

VALID = '[{"claim":"x","source_url":null,"confidence":0.0,"status":"unverified","specificity":"generic"}]'


def _verdict(claim: str) -> dict:
    return {"claim": claim, "source_url": "test://source", "confidence": 0.9,
            "status": "verified", "specificity": "substantive"}

@pytest.mark.parametrize("raw", [VALID, "  " + VALID, "```json\n" + VALID + "\n```"])
def test_contract_accepts_valid_json(raw):
    assert _parse_verifier_verdicts(raw, expected_count=1)[0]["claim"] == "x"

@pytest.mark.parametrize("raw", [
    '[{"claim":"x","source_url":null,"confidence":0.0,"status":"bad","specificity":"generic"}]',
    '[{"claim":"x","source_url":null,"confidence":0.0,"status":"unverified"}]',
    '[{"claim":"x"', 'refusal',
])
def test_contract_rejects_invalid_responses(raw):
    with pytest.raises((ValueError, ValidationError)):
        _parse_verifier_verdicts(raw, expected_count=1)

def test_contract_rejects_multi_item_single_claim_response():
    with pytest.raises(ValueError, match="expected 1"):
        _parse_verifier_verdicts("[" + VALID[1:-1] + "," + VALID[1:-1] + "]", expected_count=1)


def test_production_contract_preserves_three_claim_verdicts_and_order():
    raw = __import__("json").dumps([_verdict("first"), _verdict("second"), _verdict("third")])
    assert [item["claim"] for item in _parse_verifier_verdicts(raw, expected_count=3)] == [
        "first", "second", "third",
    ]


@pytest.mark.parametrize("count", [2, 4])
def test_production_contract_rejects_count_mismatch(count):
    raw = __import__("json").dumps([_verdict(str(index)) for index in range(count)])
    with pytest.raises(ValueError, match="expected 3"):
        _parse_verifier_verdicts(raw, expected_count=3)


def test_production_contract_validates_twelve_verdicts_without_narrowing_batch_semantics():
    raw = __import__("json").dumps([_verdict(f"claim-{index}") for index in range(12)])
    assert len(_parse_verifier_verdicts(raw, expected_count=12)) == 12
