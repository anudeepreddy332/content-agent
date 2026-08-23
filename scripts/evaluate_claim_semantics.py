"""Deterministic claim-semantics oracle. No providers, models, or network."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = REPO_ROOT / "evals" / "fixtures" / "claim_semantics_v1.json"
DEFAULT_SCHEMA = REPO_ROOT / "evals" / "claim_semantics_v1.schema.json"

EVALUATOR_ID = "claim_semantics_v1"
PACK_ID = "claim_semantics_v1"
SCHEMA_VERSION = 1
FROZEN_FIXTURE_IDS = tuple(f"F{index:02d}" for index in range(1, 15))
CANDIDATE_SET_IDS = frozenset({
    "perfect",
    "omission",
    "duplicate",
    "invention",
    "compound",
    "fragmentation",
    "qualifier_loss",
    "reordered",
    "empty",
})
EXCLUSION_REASONS = frozenset({
    "opinion",
    "transition",
    "rhetorical_question",
    "instruction_advice",
    "explicit_hypothetical",
    "draft_own_code",
})
KNOWN_ROLES = frozenset({
    "compound",
    "fragment",
    "invention",
    "qualifier_loss",
    "duplicate",
})
MATCH_INELIGIBLE_ROLES = frozenset({"compound", "fragment"})
NULL_SPAN_ROLES = frozenset({"invention", "qualifier_loss", "compound", "fragment"})
SHA256_HEX_LEN = 64


class ClaimSemanticsError(ValueError):
    """Fixture pack or candidate set is not evaluable."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def maximum_cardinality_matching(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Maximum-cardinality 1-1 matching with deterministic ID tie-breaking."""
    adjacency: dict[str, list[str]] = {}
    for candidate_id, gold_id in edges:
        adjacency.setdefault(candidate_id, []).append(gold_id)
    for candidate_id, gold_ids in adjacency.items():
        adjacency[candidate_id] = sorted(set(gold_ids))

    gold_to_candidate: dict[str, str] = {}

    def dfs(candidate_id: str, seen: set[str]) -> bool:
        for gold_id in adjacency.get(candidate_id, []):
            if gold_id in seen:
                continue
            seen.add(gold_id)
            occupant = gold_to_candidate.get(gold_id)
            if occupant is None or dfs(occupant, seen):
                gold_to_candidate[gold_id] = candidate_id
                return True
        return False

    for candidate_id in sorted(adjacency):
        dfs(candidate_id, set())

    pairs = [(candidate_id, gold_id) for gold_id, candidate_id in gold_to_candidate.items()]
    pairs.sort()
    return pairs


def ratio(numerator: int, denominator: int, zero_den_reason: str) -> dict[str, Any]:
    if denominator == 0:
        return {
            "value": None,
            "numerator": numerator,
            "denominator": denominator,
            "undefined": True,
            "undefined_reason": zero_den_reason,
        }
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
        "undefined": False,
        "undefined_reason": None,
    }


def f1_from_precision_recall(
    precision: dict[str, Any],
    recall: dict[str, Any],
) -> dict[str, Any]:
    if precision["undefined"]:
        return {
            "value": None,
            "numerator": None,
            "denominator": None,
            "undefined": True,
            "undefined_reason": "extraction precision undefined",
            "precision_numerator": precision["numerator"],
            "precision_denominator": precision["denominator"],
            "recall_numerator": recall["numerator"],
            "recall_denominator": recall["denominator"],
        }
    if recall["undefined"]:
        return {
            "value": None,
            "numerator": None,
            "denominator": None,
            "undefined": True,
            "undefined_reason": "full factual recall undefined",
            "precision_numerator": precision["numerator"],
            "precision_denominator": precision["denominator"],
            "recall_numerator": recall["numerator"],
            "recall_denominator": recall["denominator"],
        }
    p_num = precision["numerator"]
    p_den = precision["denominator"]
    r_num = recall["numerator"]
    r_den = recall["denominator"]
    harmonic_num = 2 * p_num * r_num
    harmonic_den = p_num * r_den + r_num * p_den
    if harmonic_den == 0:
        return {
            "value": None,
            "numerator": harmonic_num,
            "denominator": harmonic_den,
            "undefined": True,
            "undefined_reason": (
                "harmonic mean undefined when precision and recall are both 0"
            ),
            "precision_numerator": p_num,
            "precision_denominator": p_den,
            "recall_numerator": r_num,
            "recall_denominator": r_den,
        }
    return {
        "value": harmonic_num / harmonic_den,
        "numerator": harmonic_num,
        "denominator": harmonic_den,
        "undefined": False,
        "undefined_reason": None,
        "precision_numerator": p_num,
        "precision_denominator": p_den,
        "recall_numerator": r_num,
        "recall_denominator": r_den,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClaimSemanticsError(message)


def _require_keys(obj: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in obj]
    extra = [key for key in obj if key not in keys]
    _require(not missing, f"{label} missing keys {missing}")
    _require(not extra, f"{label} has unexpected keys {extra}")


def _validate_span(span: Any, text: str, draft: str, label: str) -> None:
    _require(
        isinstance(span, list) and len(span) == 2,
        f"{label} span must be [start, end]",
    )
    start, end = span
    _require(isinstance(start, int) and isinstance(end, int), f"{label} span bounds must be ints")
    _require(start >= 0 and end >= 0, f"{label} span bounds must be non-negative")
    _require(start < end, f"{label} span is empty or inverted")
    _require(end <= len(draft), f"{label} span is out of range")
    _require(draft[start:end] == text, f"{label} span does not match text")


def _validate_sha256(value: Any, label: str) -> None:
    _require(isinstance(value, str), f"{label} must be a hex SHA-256 string")
    _require(len(value) == SHA256_HEX_LEN, f"{label} must be 64 hex characters")
    _require(all(char in "0123456789abcdef" for char in value), f"{label} must be lowercase hex")


def validate_fixture(fixture: dict[str, Any]) -> None:
    _require(isinstance(fixture, dict), "fixture must be an object")
    _require_keys(
        fixture,
        ("id", "title", "draft_text", "draft_sha256", "gold_atoms", "exclusions", "candidate_sets"),
        "fixture",
    )
    fixture_id = fixture["id"]
    _require(fixture_id in FROZEN_FIXTURE_IDS, f"unknown fixture id {fixture_id!r}")
    _require(isinstance(fixture["title"], str) and fixture["title"], f"{fixture_id} title required")
    draft = fixture["draft_text"]
    _require(isinstance(draft, str) and draft, f"{fixture_id} draft_text required")
    _validate_sha256(fixture["draft_sha256"], f"{fixture_id} draft_sha256")
    _require(
        fixture["draft_sha256"] == sha256_text(draft),
        f"{fixture_id} draft_sha256 is stale or invalid",
    )

    gold_atoms = fixture["gold_atoms"]
    exclusions = fixture["exclusions"]
    candidate_sets = fixture["candidate_sets"]
    _require(isinstance(gold_atoms, list), f"{fixture_id} gold_atoms must be a list")
    _require(isinstance(exclusions, list), f"{fixture_id} exclusions must be a list")
    _require(isinstance(candidate_sets, list) and candidate_sets, f"{fixture_id} needs candidate_sets")

    gold_ids: set[str] = set()
    canonical_flags: dict[str, tuple[bool, bool]] = {}
    gold_spans: list[tuple[int, int, str]] = []
    for gold in gold_atoms:
        _require(isinstance(gold, dict), f"{fixture_id} gold atom must be an object")
        _require_keys(
            gold,
            ("id", "canonical_id", "text", "span", "factual", "material"),
            f"{fixture_id} gold atom",
        )
        gold_id = gold["id"]
        _require(isinstance(gold_id, str) and gold_id, f"{fixture_id} gold id required")
        _require(gold_id not in gold_ids, f"{fixture_id} duplicate gold id {gold_id}")
        gold_ids.add(gold_id)
        _require(isinstance(gold["canonical_id"], str) and gold["canonical_id"], f"{gold_id} canonical_id")
        _require(isinstance(gold["text"], str) and gold["text"], f"{gold_id} text required")
        _require(isinstance(gold["factual"], bool), f"{gold_id} factual must be bool")
        _require(isinstance(gold["material"], bool), f"{gold_id} material must be bool")
        _require(not gold["material"] or gold["factual"], f"{gold_id} material atom must be factual")
        _validate_span(gold["span"], gold["text"], draft, gold_id)
        flags = (gold["factual"], gold["material"])
        previous = canonical_flags.get(gold["canonical_id"])
        if previous is None:
            canonical_flags[gold["canonical_id"]] = flags
        else:
            _require(previous == flags, f"{gold_id} canonical flags disagree with sibling rows")
        gold_spans.append((gold["span"][0], gold["span"][1], gold_id))

    exclusion_ids: set[str] = set()
    for exclusion in exclusions:
        _require(isinstance(exclusion, dict), f"{fixture_id} exclusion must be an object")
        _require_keys(exclusion, ("id", "text", "span", "reason"), f"{fixture_id} exclusion")
        exclusion_id = exclusion["id"]
        _require(isinstance(exclusion_id, str) and exclusion_id, f"{fixture_id} exclusion id required")
        _require(exclusion_id not in exclusion_ids, f"{fixture_id} duplicate exclusion id {exclusion_id}")
        _require(exclusion_id not in gold_ids, f"{exclusion_id} cannot also be a gold id")
        exclusion_ids.add(exclusion_id)
        _require(isinstance(exclusion["text"], str) and exclusion["text"], f"{exclusion_id} text")
        _require(exclusion["reason"] in EXCLUSION_REASONS, f"{exclusion_id} unknown exclusion reason")
        _validate_span(exclusion["span"], exclusion["text"], draft, exclusion_id)
        start, end = exclusion["span"]
        for gold_start, gold_end, gold_id in gold_spans:
            _require(
                end <= gold_start or start >= gold_end,
                f"{exclusion_id} overlaps gold {gold_id}",
            )

    seen_sets: set[str] = set()
    for candidate_set in candidate_sets:
        _validate_candidate_set(fixture_id, gold_ids, draft, candidate_set)
        set_id = candidate_set["id"]
        _require(set_id not in seen_sets, f"{fixture_id} duplicate candidate_set id {set_id}")
        seen_sets.add(set_id)


def _validate_candidate_set(
    fixture_id: str,
    gold_ids: set[str],
    draft: str,
    candidate_set: dict[str, Any],
) -> None:
    _require(isinstance(candidate_set, dict), f"{fixture_id} candidate_set must be an object")
    _require_keys(
        candidate_set,
        ("id", "candidates", "allowed_matches"),
        f"{fixture_id} candidate_set",
    )
    set_id = candidate_set["id"]
    _require(set_id in CANDIDATE_SET_IDS, f"{fixture_id} unknown candidate_set id {set_id!r}")
    candidates = candidate_set["candidates"]
    matches = candidate_set["allowed_matches"]
    _require(isinstance(candidates, list), f"{fixture_id}.{set_id} candidates must be a list")
    _require(isinstance(matches, list), f"{fixture_id}.{set_id} allowed_matches must be a list")

    candidate_ids: set[str] = set()
    for candidate in candidates:
        _require(isinstance(candidate, dict), f"{fixture_id}.{set_id} candidate must be an object")
        _require_keys(
            candidate,
            ("id", "canonical_id", "text", "span", "roles"),
            f"{fixture_id}.{set_id} candidate",
        )
        candidate_id = candidate["id"]
        _require(isinstance(candidate_id, str) and candidate_id, f"{fixture_id}.{set_id} candidate id")
        _require(candidate_id not in candidate_ids, f"{fixture_id}.{set_id} duplicate candidate {candidate_id}")
        candidate_ids.add(candidate_id)
        _require(
            isinstance(candidate["canonical_id"], str) and candidate["canonical_id"],
            f"{candidate_id} canonical_id required",
        )
        _require(isinstance(candidate["text"], str) and candidate["text"], f"{candidate_id} text required")
        roles = candidate["roles"]
        _require(isinstance(roles, list), f"{candidate_id} roles must be a list")
        _require(len(roles) == len(set(roles)), f"{candidate_id} roles must be unique")
        unknown_roles = set(roles) - KNOWN_ROLES
        _require(not unknown_roles, f"{candidate_id} unknown roles {sorted(unknown_roles)}")
        span = candidate["span"]
        if span is None:
            _require(
                bool(NULL_SPAN_ROLES.intersection(roles)),
                f"{candidate_id} null span requires invention or qualifier_loss",
            )
        else:
            _validate_span(span, candidate["text"], draft, candidate_id)

    seen_edges: set[tuple[str, str]] = set()
    for edge in matches:
        _require(isinstance(edge, dict), f"{fixture_id}.{set_id} allowed match must be an object")
        _require_keys(edge, ("candidate_id", "gold_id"), f"{fixture_id}.{set_id} allowed match")
        candidate_id = edge["candidate_id"]
        gold_id = edge["gold_id"]
        _require(candidate_id in candidate_ids, f"{fixture_id}.{set_id} unknown candidate_id {candidate_id}")
        _require(gold_id in gold_ids, f"{fixture_id}.{set_id} unknown gold_id {gold_id}")
        pair = (candidate_id, gold_id)
        _require(pair not in seen_edges, f"{fixture_id}.{set_id} duplicate match edge {pair}")
        seen_edges.add(pair)


def validate_pack(pack: dict[str, Any], *, require_frozen_catalog: bool = True) -> None:
    _require(isinstance(pack, dict), "pack must be an object")
    _require_keys(
        pack,
        ("pack_id", "schema_version", "evaluator_id", "description", "fixtures"),
        "pack",
    )
    _require(pack["pack_id"] == PACK_ID, f"pack_id must be {PACK_ID}")
    _require(pack["schema_version"] == SCHEMA_VERSION, "schema_version must be 1")
    _require(pack["evaluator_id"] == EVALUATOR_ID, f"evaluator_id must be {EVALUATOR_ID}")
    _require(isinstance(pack["description"], str) and pack["description"], "description required")
    fixtures = pack["fixtures"]
    _require(isinstance(fixtures, list), "fixtures must be a list")
    seen: set[str] = set()
    for fixture in fixtures:
        validate_fixture(fixture)
        _require(fixture["id"] not in seen, f"duplicate fixture id {fixture['id']}")
        seen.add(fixture["id"])
    if require_frozen_catalog:
        _require(len(fixtures) == 14, f"official pack must contain 14 fixtures, got {len(fixtures)}")
        _require(tuple(fixture["id"] for fixture in fixtures) == FROZEN_FIXTURE_IDS, "official pack IDs must be F01-F14 in order")


def load_pack(path: Path | str = DEFAULT_FIXTURES, *, require_frozen_catalog: bool = True) -> dict[str, Any]:
    pack_path = Path(path)
    try:
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClaimSemanticsError(f"unreadable fixture pack {pack_path}: {error}") from error
    validate_pack(pack, require_frozen_catalog=require_frozen_catalog)
    return pack


def evaluate_candidate_set(fixture: dict[str, Any], candidate_set: dict[str, Any]) -> dict[str, Any]:
    validate_fixture(fixture)
    gold_ids = {gold["id"] for gold in fixture["gold_atoms"]}
    _validate_candidate_set(fixture["id"], gold_ids, fixture["draft_text"], candidate_set)

    gold_by_id = {gold["id"]: gold for gold in fixture["gold_atoms"]}
    material_canonical = {
        gold["canonical_id"] for gold in fixture["gold_atoms"] if gold["material"]
    }
    factual_canonical = {
        gold["canonical_id"] for gold in fixture["gold_atoms"] if gold["factual"]
    }

    candidates = candidate_set["candidates"]
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    canonical_candidates = {candidate["canonical_id"] for candidate in candidates}

    eligible_ids = {
        candidate["id"]
        for candidate in candidates
        if MATCH_INELIGIBLE_ROLES.isdisjoint(candidate["roles"])
    }
    edges = [
        (edge["candidate_id"], edge["gold_id"])
        for edge in candidate_set["allowed_matches"]
        if edge["candidate_id"] in eligible_ids
    ]
    pairs = maximum_cardinality_matching(edges)
    matched_gold_canonical = {
        gold_by_id[gold_id]["canonical_id"] for _candidate_id, gold_id in pairs
    }
    matched_candidate_canonical = {
        candidate_by_id[candidate_id]["canonical_id"] for candidate_id, _gold_id in pairs
    }

    material_recall = ratio(
        len(matched_gold_canonical & material_canonical),
        len(material_canonical),
        "zero material gold atoms",
    )
    factual_recall = ratio(
        len(matched_gold_canonical & factual_canonical),
        len(factual_canonical),
        "zero factual gold atoms",
    )
    precision = ratio(
        len(matched_candidate_canonical),
        len(canonical_candidates),
        "zero canonical candidate claims",
    )
    extraction_f1 = f1_from_precision_recall(precision, factual_recall)
    duplicate_count = len(candidates) - len(canonical_candidates)
    duplicate_rate = ratio(duplicate_count, len(candidates), "zero candidate rows")

    return {
        "fixture_id": fixture["id"],
        "candidate_set_id": candidate_set["id"],
        "metrics": {
            "material_claim_recall": material_recall,
            "full_factual_recall": factual_recall,
            "extraction_precision": precision,
            "extraction_f1": extraction_f1,
            "duplicate_count": duplicate_count,
            "duplicate_rate": duplicate_rate,
            "atomicity_violations": sum(
                1 for candidate in candidates if "compound" in candidate["roles"]
            ),
            "fragmentation_violations": sum(
                1 for candidate in candidates if "fragment" in candidate["roles"]
            ),
        },
        "matching": {
            "pairs": [list(pair) for pair in pairs],
            "matched_gold_ids": sorted(gold_id for _candidate_id, gold_id in pairs),
            "matched_candidate_ids": sorted(candidate_id for candidate_id, _gold_id in pairs),
            "matched_gold_canonical_ids": sorted(matched_gold_canonical),
            "matched_candidate_canonical_ids": sorted(matched_candidate_canonical),
        },
        "counts": {
            "gold_rows": len(fixture["gold_atoms"]),
            "material_gold_atoms": len(material_canonical),
            "factual_gold_atoms": len(factual_canonical),
            "candidate_rows": len(candidates),
            "canonical_candidate_claims": len(canonical_candidates),
            "exclusions": len(fixture["exclusions"]),
        },
    }


def evaluate_pack(
    pack: dict[str, Any],
    *,
    require_frozen_catalog: bool = True,
    fixture_id: str | None = None,
    candidate_set_id: str | None = None,
) -> dict[str, Any]:
    validate_pack(pack, require_frozen_catalog=require_frozen_catalog)
    results: list[dict[str, Any]] = []
    gold_canonical: set[str] = set()
    material_canonical: set[str] = set()
    factual_canonical: set[str] = set()
    for fixture in pack["fixtures"]:
        if fixture_id is not None and fixture["id"] != fixture_id:
            continue
        for gold in fixture["gold_atoms"]:
            gold_canonical.add(f"{fixture['id']}:{gold['canonical_id']}")
            if gold["material"]:
                material_canonical.add(f"{fixture['id']}:{gold['canonical_id']}")
            if gold["factual"]:
                factual_canonical.add(f"{fixture['id']}:{gold['canonical_id']}")
        for candidate_set in fixture["candidate_sets"]:
            if candidate_set_id is not None and candidate_set["id"] != candidate_set_id:
                continue
            results.append(evaluate_candidate_set(fixture, candidate_set))
    if fixture_id is not None and not any(fixture["id"] == fixture_id for fixture in pack["fixtures"]):
        raise ClaimSemanticsError(f"unknown fixture id {fixture_id}")
    if candidate_set_id is not None and not results:
        raise ClaimSemanticsError(f"unknown candidate_set id {candidate_set_id}")
    results.sort(key=lambda item: (item["fixture_id"], item["candidate_set_id"]))
    return {
        "evaluator_id": EVALUATOR_ID,
        "schema_version": SCHEMA_VERSION,
        "pack_id": PACK_ID,
        "fixture_count": len(pack["fixtures"]),
        "gold_atom_count": len(gold_canonical) if fixture_id is None else len({
            gold["canonical_id"]
            for fixture in pack["fixtures"]
            if fixture["id"] == fixture_id
            for gold in fixture["gold_atoms"]
        }),
        "material_gold_atom_count": len(material_canonical) if fixture_id is None else len({
            gold["canonical_id"]
            for fixture in pack["fixtures"]
            if fixture["id"] == fixture_id
            for gold in fixture["gold_atoms"]
            if gold["material"]
        }),
        "factual_gold_atom_count": len(factual_canonical) if fixture_id is None else len({
            gold["canonical_id"]
            for fixture in pack["fixtures"]
            if fixture["id"] == fixture_id
            for gold in fixture["gold_atoms"]
            if gold["factual"]
        }),
        "results": results,
    }


def evaluate_default_pack(
    path: Path | str = DEFAULT_FIXTURES,
    **kwargs: Any,
) -> dict[str, Any]:
    return evaluate_pack(load_pack(path), **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the claim-semantics v1 oracle.")
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    parser.add_argument("--fixture", default=None)
    parser.add_argument("--candidate-set", default=None)
    args = parser.parse_args(argv)
    try:
        report = evaluate_default_pack(
            args.fixtures,
            fixture_id=args.fixture,
            candidate_set_id=args.candidate_set,
        )
    except ClaimSemanticsError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(normalize_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
