"""Offline claim-inventory golden-set evaluator (Phase 4 Slice 2A).

Deterministic, offline, ZERO provider calls. Each frozen case in
``evals/fixtures/claim_inventory_golden_v1.json`` carries:

- the exact ``draft_markdown`` (one exact draft version);
- a hand-authored simulated Call-A response (``call_a_response``) — written
  independently of the implementation;
- hand-labeled ``gold_claims`` — the independent reference.

The harness runs ONLY the deterministic production pipeline
(``parse_claim_inventory_rows`` → ``build_claim_inventory``) and compares the
canonical inventory against gold. The extractor never grades itself: model
outputs are frozen data, gold labels are frozen data, and all comparison logic
lives here.

There is NO calibrated production threshold for claim recall, material-claim
recall, or false-nonmaterial rate. This harness computes metrics, preserves
denominators, and reports exact failures to support later calibration. It does
NOT establish general production extraction quality.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.claim_inventory import (
    build_claim_inventory,
    canonical_claim_text,
    parse_claim_inventory_rows,
)

FIXTURE = Path(__file__).resolve().parents[1] / "evals/fixtures/claim_inventory_golden_v1.json"


def _canon(text: str) -> str:
    return canonical_claim_text(text)


def _occurrences_key(claim: dict) -> list[list[int]]:
    return [[o["start"], o["end"]] for o in claim.get("occurrences", [])]


def evaluate_case(case: dict) -> dict:
    """Run the deterministic pipeline on one frozen case and compare to gold."""
    rows = parse_claim_inventory_rows(json.dumps(case["call_a_response"]))
    inventory = build_claim_inventory(
        run_id=f"golden-{case['case_id']}",
        iteration=1,
        draft_markdown=case["draft_markdown"],
        raw_claims=rows,
        brief_requirements=case.get("brief_requirements") or [],
    )
    produced = {_canon(c["claim_text"]): c for c in inventory["claims"]}
    gold = {_canon(g["claim_text"]): g for g in case["gold_claims"]}

    matched = sorted(set(produced) & set(gold))
    missing = sorted(set(gold) - set(produced))       # recall misses
    extra = sorted(set(produced) - set(gold))         # precision errors

    comparisons = []
    for key in matched:
        p, g = produced[key], gold[key]
        comparisons.append({
            "claim_text": g["claim_text"],
            "gold_type": g["claim_type"],
            "produced_type": p["claim_type"],
            "gold_material": g["material"],
            "produced_material": p["material"],
            "gold_anchor_validity": g["anchor_validity"],
            "produced_anchor_validity": p["anchor_validity"],
            "gold_occurrences": g["occurrences"],
            "produced_occurrences": _occurrences_key(p),
            "occurrences_match": _occurrences_key(p) == g["occurrences"],
            "gold_call_b_eligible": g["call_b_eligible"],
            "produced_call_b_eligible": p["call_b_eligible"],
            "gold_requires_citation": g.get("requires_citation"),
            "produced_requires_citation": p.get("requires_citation"),
        })

    return {
        "case_id": case["case_id"],
        "tags": case.get("tags", []),
        "inventory": inventory,
        "gold_claims": case["gold_claims"],
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "comparisons": comparisons,
    }


def _material_key(value) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def compute_metrics(results: list[dict]) -> dict:
    """Denominators preserved; every counted error names its case."""
    gold_factual = produced_factual = matched_factual = 0
    gold_material = material_correct = 0
    false_nonmaterial_cases: list[str] = []
    confusion: dict[str, dict[str, int]] = {
        g: {"true": 0, "false": 0, "unknown": 0} for g in ("true", "false", "unknown")
    }
    anchor_total = anchor_correct = 0
    anchor_failures: list[str] = []
    split_errors: list[dict] = []
    merge_errors: list[dict] = []
    duplicate_cases = duplicate_ok = 0

    for result in results:
        case_id = result["case_id"]
        produced = {_canon(c["claim_text"]): c for c in result["inventory"]["claims"]}
        gold_claim_texts = {_canon(g["claim_text"]) for g in result["gold_claims"]}
        gold_factual += sum(
            1 for g in result["gold_claims"] if g["claim_type"] == "factual"
        )
        produced_factual += sum(
            1 for c in result["inventory"]["claims"] if c["claim_type"] == "factual"
        )
        matched_factual += len([
            c for c in result["comparisons"] if c["gold_type"] == "factual"
        ])

        for comp in result["comparisons"]:
            gm, pm = _material_key(comp["gold_material"]), _material_key(comp["produced_material"])
            confusion[gm][pm] += 1
            if comp["gold_material"] is True:
                gold_material += 1
                if comp["produced_material"] is True:
                    material_correct += 1
                elif comp["produced_material"] is False:
                    false_nonmaterial_cases.append(f"{case_id}:{comp['claim_text'][:60]}")
            anchor_total += 1
            if comp["occurrences_match"]:
                anchor_correct += 1
            else:
                anchor_failures.append(f"{case_id}:{comp['claim_text'][:60]}")

        # split: one gold claim whose text strictly contains >=2 produced claims
        produced_texts = set(produced)
        for key in gold_claim_texts:
            contained = [p for p in produced_texts if p != key and p in key]
            if len(contained) >= 2:
                split_errors.append({"case": case_id, "gold": key, "parts": sorted(contained)})
        # merge: one produced claim whose text strictly contains >=2 gold claims
        for key in produced_texts:
            contained = [g for g in gold_claim_texts if g != key and g in key]
            if len(contained) >= 2:
                merge_errors.append({"case": case_id, "produced": key, "parts": sorted(contained)})

        if "repeated_claim" in result["tags"]:
            duplicate_cases += 1
            claims = result["inventory"]["claims"]
            if (
                len(claims) == 1
                and len(claims[0]["occurrences"]) == 2
                and claims[0]["anchor_validity"] == "bound_multiple"
            ):
                duplicate_ok += 1

    return {
        "factual_claim_recall": (matched_factual / gold_factual) if gold_factual else None,
        "factual_claim_precision": (matched_factual / produced_factual) if produced_factual else None,
        "factual_recall_denominator": gold_factual,
        "factual_precision_denominator": produced_factual,
        "material_claim_recall": (material_correct / gold_material) if gold_material else None,
        "material_claim_recall_denominator": gold_material,
        "split_error_count": len(split_errors),
        "merge_error_count": len(merge_errors),
        "split_errors": split_errors,
        "merge_errors": merge_errors,
        "duplicate_cases": duplicate_cases,
        "duplicate_cases_correct": duplicate_ok,
        "anchor_binding_accuracy": (anchor_correct / anchor_total) if anchor_total else None,
        "anchor_binding_denominator": anchor_total,
        "anchor_failures": anchor_failures,
        "materiality_confusion": confusion,
        "false_nonmaterial_count": len(false_nonmaterial_cases),
        "false_nonmaterial_rate": (
            len(false_nonmaterial_cases) / gold_material if gold_material else None
        ),
        "false_nonmaterial_cases": false_nonmaterial_cases,
    }


def evaluate_pack(pack: dict) -> dict:
    results = [evaluate_case(case) for case in pack["cases"]]
    return {
        "fixture": pack.get("schema"),
        "case_count": len(results),
        "results": results,
        "metrics": compute_metrics(results),
    }


def main() -> int:
    pack = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report = evaluate_pack(pack)
    printable = {k: v for k, v in report.items() if k != "results"}
    printable["cases"] = [
        {
            "case_id": r["case_id"],
            "matched": len(r["matched"]),
            "missing": r["missing"],
            "extra": r["extra"],
        }
        for r in report["results"]
    ]
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
