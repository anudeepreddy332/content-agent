"""Deterministic claim-semantics v2 oracle and P0 semantic metrics. No providers."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = REPO_ROOT / "evals" / "fixtures" / "claim_semantics_v2.json"
DEFAULT_SCHEMA = REPO_ROOT / "evals" / "claim_semantics_v2.schema.json"
DEFAULT_REGISTRY = REPO_ROOT / "evals" / "metric_registry_v1.json"

EVALUATOR_ID = "claim_semantics_v2"
PACK_ID = "claim_semantics_v2"
SCHEMA_VERSION = 2
FROZEN_FIXTURE_IDS = ("ADV01",)
SEMANTIC_STATUSES = frozenset({"verified", "weak", "unverified"})
ROUTE_DECISIONS = frozenset({"PASS", "FAIL"})
KNOWN_ROLES = frozenset({
    "compound",
    "fragment",
    "invention",
    "qualifier_loss",
    "duplicate",
})
MATCH_INELIGIBLE_ROLES = frozenset({
    "compound",
    "fragment",
    "qualifier_loss",
    "invention",
})
NULL_SPAN_ROLES = frozenset({"invention", "qualifier_loss", "compound", "fragment"})
EXCLUSION_REASONS = frozenset({
    "opinion",
    "transition",
    "rhetorical_question",
    "instruction_advice",
    "explicit_hypothetical",
    "draft_own_code",
})
SHA256_HEX_LEN = 64


class ClaimSemanticsV2Error(ValueError):
    """Fixture pack or asset is not evaluable."""


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClaimSemanticsV2Error(message)


def _require_keys(obj: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in obj]
    extra = [key for key in obj if key not in keys]
    _require(not missing, f"{label} missing keys {missing}")
    _require(not extra, f"{label} has unexpected keys {extra}")


PACK_REQUIRED_KEYS = (
    "pack_id",
    "schema_version",
    "evaluator_id",
    "description",
    "fixtures",
)
FIXTURE_REQUIRED_KEYS = (
    "id",
    "title",
    "draft_text",
    "draft_sha256",
    "gold_atoms",
    "exclusions",
    "candidates",
    "allowed_matches",
    "evidence_bindings",
    "verifier_rows",
    "automatic_route",
)
GOLD_REQUIRED_KEYS = (
    "id",
    "canonical_id",
    "text",
    "span",
    "factual",
    "material",
    "gold_semantic_status",
)
EXCLUSION_REQUIRED_KEYS = ("id", "text", "span", "reason")
CANDIDATE_REQUIRED_KEYS = (
    "id",
    "canonical_id",
    "text",
    "span",
    "roles",
    "predicted_semantic_status",
)
ALLOWED_MATCH_REQUIRED_KEYS = ("candidate_id", "gold_id")
EVIDENCE_BINDING_REQUIRED_KEYS = (
    "id",
    "gold_id",
    "candidate_id",
    "evidence_id",
    "valid",
    "fully_entailed",
)
VERIFIER_ROW_REQUIRED_KEYS = ("id", "status")
AUTOMATIC_ROUTE_REQUIRED_KEYS = ("decision",)
FIXTURE_ID_PATTERN = r"^ADV0[1-9]$|^ADV[1-9][0-9]$"
DRAFT_SHA256_PATTERN = r"^[a-f0-9]{64}$"


def runtime_schema_contract() -> dict[str, Any]:
    return {
        "pack_required": tuple(sorted(PACK_REQUIRED_KEYS)),
        "pack_properties": tuple(sorted(PACK_REQUIRED_KEYS)),
        "pack_id": PACK_ID,
        "schema_version": SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "fixture_cardinality": 1,
        "fixture_required": tuple(sorted(FIXTURE_REQUIRED_KEYS)),
        "fixture_properties": tuple(sorted(FIXTURE_REQUIRED_KEYS)),
        "gold_required": tuple(sorted(GOLD_REQUIRED_KEYS)),
        "gold_properties": tuple(sorted(GOLD_REQUIRED_KEYS)),
        "exclusion_required": tuple(sorted(EXCLUSION_REQUIRED_KEYS)),
        "exclusion_properties": tuple(sorted(EXCLUSION_REQUIRED_KEYS)),
        "candidate_required": tuple(sorted(CANDIDATE_REQUIRED_KEYS)),
        "candidate_properties": tuple(sorted(CANDIDATE_REQUIRED_KEYS)),
        "allowed_match_required": tuple(sorted(ALLOWED_MATCH_REQUIRED_KEYS)),
        "allowed_match_properties": tuple(sorted(ALLOWED_MATCH_REQUIRED_KEYS)),
        "evidence_binding_required": tuple(sorted(EVIDENCE_BINDING_REQUIRED_KEYS)),
        "evidence_binding_properties": tuple(sorted(EVIDENCE_BINDING_REQUIRED_KEYS)),
        "verifier_row_required": tuple(sorted(VERIFIER_ROW_REQUIRED_KEYS)),
        "verifier_row_properties": tuple(sorted(VERIFIER_ROW_REQUIRED_KEYS)),
        "automatic_route_required": tuple(sorted(AUTOMATIC_ROUTE_REQUIRED_KEYS)),
        "automatic_route_properties": tuple(sorted(AUTOMATIC_ROUTE_REQUIRED_KEYS)),
        "candidate_roles": tuple(sorted(KNOWN_ROLES)),
        "semantic_statuses": tuple(sorted(SEMANTIC_STATUSES)),
        "route_decisions": tuple(sorted(ROUTE_DECISIONS)),
        "exclusion_reasons": tuple(sorted(EXCLUSION_REASONS)),
        "additional_properties": False,
        "span_length": 2,
        "span_value_type": "integer",
        "span_minimum": 0,
        "fixture_id_pattern": FIXTURE_ID_PATTERN,
        "draft_sha256_pattern": DRAFT_SHA256_PATTERN,
        "frozen_fixture_ids": FROZEN_FIXTURE_IDS,
    }


def load_frozen_schema(path: Path | str = DEFAULT_SCHEMA) -> dict[str, Any]:
    schema_path = Path(path)
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClaimSemanticsV2Error(f"unreadable frozen schema {schema_path}: {error}") from error
    _require(isinstance(schema, dict), "frozen schema must be an object")
    _require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        "frozen schema must declare JSON Schema Draft 2020-12",
    )
    return schema


def extract_schema_contract(schema: dict[str, Any]) -> dict[str, Any]:
    _require("$defs" in schema, "frozen schema missing $defs")
    defs = schema["$defs"]

    def _required(node: dict[str, Any], label: str) -> tuple[str, ...]:
        keys = node.get("required")
        _require(isinstance(keys, list) and keys, f"{label} required keys missing")
        return tuple(sorted(keys))

    def _properties(node: dict[str, Any], label: str) -> tuple[str, ...]:
        properties = node.get("properties")
        _require(isinstance(properties, dict) and properties, f"{label} properties missing")
        return tuple(sorted(properties))

    def _additional_false(node: dict[str, Any], label: str) -> None:
        _require(node.get("additionalProperties") is False, f"{label} additionalProperties must be false")

    _additional_false(schema, "pack")
    for label, node in (
        ("fixture", defs["fixture"]),
        ("gold_atom", defs["gold_atom"]),
        ("exclusion", defs["exclusion"]),
        ("candidate", defs["candidate"]),
        ("allowed_match", defs["allowed_match"]),
        ("evidence_binding", defs["evidence_binding"]),
        ("verifier_row", defs["verifier_row"]),
        ("automatic_route", defs["automatic_route"]),
    ):
        _additional_false(node, label)

    roles = defs["roles"]["items"]["enum"]
    statuses = defs["semantic_status"]["enum"]
    decisions = defs["automatic_route"]["properties"]["decision"]["enum"]
    reasons = defs["exclusion_reason"]["enum"]
    fixture_id_pattern = defs["fixture"]["properties"]["id"]["pattern"]

    return {
        "pack_required": _required(schema, "pack"),
        "pack_properties": _properties(schema, "pack"),
        "pack_id": schema["properties"]["pack_id"]["const"],
        "schema_version": schema["properties"]["schema_version"]["const"],
        "evaluator_id": schema["properties"]["evaluator_id"]["const"],
        "fixture_cardinality": schema["properties"]["fixtures"]["minItems"],
        "fixture_required": _required(defs["fixture"], "fixture"),
        "fixture_properties": _properties(defs["fixture"], "fixture"),
        "gold_required": _required(defs["gold_atom"], "gold_atom"),
        "gold_properties": _properties(defs["gold_atom"], "gold_atom"),
        "exclusion_required": _required(defs["exclusion"], "exclusion"),
        "exclusion_properties": _properties(defs["exclusion"], "exclusion"),
        "candidate_required": _required(defs["candidate"], "candidate"),
        "candidate_properties": _properties(defs["candidate"], "candidate"),
        "allowed_match_required": _required(defs["allowed_match"], "allowed_match"),
        "allowed_match_properties": _properties(defs["allowed_match"], "allowed_match"),
        "evidence_binding_required": _required(defs["evidence_binding"], "evidence_binding"),
        "evidence_binding_properties": _properties(defs["evidence_binding"], "evidence_binding"),
        "verifier_row_required": _required(defs["verifier_row"], "verifier_row"),
        "verifier_row_properties": _properties(defs["verifier_row"], "verifier_row"),
        "automatic_route_required": _required(defs["automatic_route"], "automatic_route"),
        "automatic_route_properties": _properties(defs["automatic_route"], "automatic_route"),
        "candidate_roles": tuple(sorted(roles)),
        "semantic_statuses": tuple(sorted(statuses)),
        "route_decisions": tuple(sorted(decisions)),
        "exclusion_reasons": tuple(sorted(reasons)),
        "additional_properties": False,
        "span_length": 2,
        "span_value_type": "integer",
        "span_minimum": 0,
        "fixture_id_pattern": fixture_id_pattern,
        "draft_sha256_pattern": defs["fixture"]["properties"]["draft_sha256"]["pattern"],
        "frozen_fixture_ids": FROZEN_FIXTURE_IDS,
    }


def assert_schema_runtime_parity(schema: dict[str, Any] | None = None) -> dict[str, Any]:
    schema = load_frozen_schema() if schema is None else schema
    extracted = extract_schema_contract(schema)
    runtime = runtime_schema_contract()
    drifted = sorted(key for key in set(extracted) | set(runtime) if extracted.get(key) != runtime.get(key))
    _require(not drifted, f"schema/runtime contract drift on fields {drifted}")
    fixtures_node = schema["properties"]["fixtures"]
    _require(fixtures_node.get("minItems") == fixtures_node.get("maxItems"), "fixture minItems must equal maxItems")
    return extracted


def _schema_type_ok(instance: Any, expected: str | list[str]) -> bool:
    if isinstance(expected, list):
        return any(_schema_type_ok(instance, item) for item in expected)
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "integer":
        return type(instance) is int
    if expected == "number":
        return type(instance) is int or type(instance) is float
    if expected == "boolean":
        return type(instance) is bool
    if expected == "null":
        return instance is None
    return False


def _evaluate_json_schema(instance: Any, node: dict[str, Any], root: dict[str, Any], label: str) -> None:
    if "$ref" in node:
        ref = node["$ref"]
        _require(ref.startswith("#/$defs/"), f"{label} unsupported $ref {ref}")
        name = ref.rsplit("/", 1)[-1]
        _require(name in root.get("$defs", {}), f"{label} unknown $ref {ref}")
        _evaluate_json_schema(instance, root["$defs"][name], root, label)
        leftover = {key: value for key, value in node.items() if key != "$ref"}
        if leftover:
            _evaluate_json_schema(instance, leftover, root, label)
        return

    if "anyOf" in node:
        for alternative in node["anyOf"]:
            try:
                _evaluate_json_schema(instance, alternative, root, label)
                return
            except ClaimSemanticsV2Error:
                continue
        raise ClaimSemanticsV2Error(f"{label} matches no anyOf alternative")

    expected_type = node.get("type")
    if expected_type is not None:
        _require(_schema_type_ok(instance, expected_type), f"{label} has invalid type")

    if expected_type == "object":
        properties = node.get("properties", {})
        if node.get("additionalProperties") is False:
            extra = [key for key in instance if key not in properties]
            _require(not extra, f"{label} has unexpected keys {extra}")
        missing = [key for key in node.get("required", []) if key not in instance]
        _require(not missing, f"{label} missing keys {missing}")
        for key, child in properties.items():
            if key in instance:
                _evaluate_json_schema(instance[key], child, root, f"{label}.{key}")

    if expected_type == "array":
        if "minItems" in node:
            _require(len(instance) >= node["minItems"], f"{label} has too few items")
        if "maxItems" in node:
            _require(len(instance) <= node["maxItems"], f"{label} has too many items")
        if node.get("uniqueItems"):
            serialized = [json.dumps(item, sort_keys=True) for item in instance]
            _require(len(serialized) == len(set(serialized)), f"{label} items must be unique")
        if "prefixItems" in node:
            prefix = node["prefixItems"]
            _require(len(instance) >= len(prefix), f"{label} missing prefixItems")
            for index, child in enumerate(prefix):
                _evaluate_json_schema(instance[index], child, root, f"{label}[{index}]")
            if node.get("items") is False:
                _require(len(instance) == len(prefix), f"{label} has unexpected extra items")
        elif "items" in node and node["items"] is not False:
            for index, item in enumerate(instance):
                _evaluate_json_schema(item, node["items"], root, f"{label}[{index}]")

    if expected_type == "string":
        if "minLength" in node:
            _require(len(instance) >= node["minLength"], f"{label} is too short")
        if "pattern" in node:
            _require(re.fullmatch(node["pattern"], instance) is not None, f"{label} does not match pattern")

    if expected_type == "integer" and "minimum" in node:
        _require(instance >= node["minimum"], f"{label} is below minimum")

    if "const" in node:
        _require(instance == node["const"], f"{label} must equal {node['const']!r}")
    if "enum" in node:
        _require(instance in node["enum"], f"{label} has invalid enum value {instance!r}")


def validate_against_frozen_schema(
    instance: Any,
    schema: dict[str, Any] | None = None,
    *,
    label: str = "pack",
) -> None:
    schema = load_frozen_schema() if schema is None else schema
    _evaluate_json_schema(instance, schema, schema, label)


def _validate_span(span: Any, text: str, draft: str, label: str) -> None:
    _require(isinstance(span, list) and len(span) == 2, f"{label} span must be [start, end]")
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


def _candidate_is_match_eligible(candidate: dict[str, Any]) -> bool:
    return MATCH_INELIGIBLE_ROLES.isdisjoint(candidate["roles"])


def _validate_candidates_and_matches(
    fixture_id: str,
    gold_ids: set[str],
    draft: str,
    candidates: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidate_ids: set[str] = set()
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        _require(isinstance(candidate, dict), f"{fixture_id} candidate must be an object")
        _require_keys(candidate, CANDIDATE_REQUIRED_KEYS, f"{fixture_id} candidate")
        candidate_id = candidate["id"]
        _require(isinstance(candidate_id, str) and candidate_id, f"{fixture_id} candidate id required")
        _require(candidate_id not in candidate_ids, f"{fixture_id} duplicate candidate {candidate_id}")
        candidate_ids.add(candidate_id)
        _require(
            isinstance(candidate["canonical_id"], str) and candidate["canonical_id"],
            f"{candidate_id} canonical_id required",
        )
        _require(isinstance(candidate["text"], str) and candidate["text"], f"{candidate_id} text required")
        _require(candidate["predicted_semantic_status"] in SEMANTIC_STATUSES, f"{candidate_id} invalid predicted status")
        roles = candidate["roles"]
        _require(isinstance(roles, list), f"{candidate_id} roles must be a list")
        _require(len(roles) == len(set(roles)), f"{candidate_id} roles must be unique")
        unknown_roles = set(roles) - KNOWN_ROLES
        _require(not unknown_roles, f"{candidate_id} unknown roles {sorted(unknown_roles)}")
        span = candidate["span"]
        if span is None:
            _require(
                bool(NULL_SPAN_ROLES.intersection(roles)),
                f"{candidate_id} null span requires invention, qualifier_loss, compound, or fragment",
            )
        else:
            _validate_span(span, candidate["text"], draft, candidate_id)
        candidate_by_id[candidate_id] = candidate

    seen_edges: set[tuple[str, str]] = set()
    for edge in matches:
        _require(isinstance(edge, dict), f"{fixture_id} allowed match must be an object")
        _require_keys(edge, ALLOWED_MATCH_REQUIRED_KEYS, f"{fixture_id} allowed match")
        candidate_id = edge["candidate_id"]
        gold_id = edge["gold_id"]
        _require(candidate_id in candidate_ids, f"{fixture_id} unknown candidate_id {candidate_id}")
        _require(gold_id in gold_ids, f"{fixture_id} unknown gold_id {gold_id}")
        pair = (candidate_id, gold_id)
        _require(pair not in seen_edges, f"{fixture_id} duplicate match edge {pair}")
        seen_edges.add(pair)
        candidate = candidate_by_id[candidate_id]
        forbidden = MATCH_INELIGIBLE_ROLES.intersection(candidate["roles"])
        _require(
            not forbidden,
            f"{fixture_id} forbidden-role allowed edge ({candidate_id}, {gold_id}) roles {sorted(forbidden)}",
        )

    return candidate_by_id


def _validate_evidence_bindings(
    fixture_id: str,
    gold_ids: set[str],
    candidate_ids: set[str],
    bindings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    binding_ids: set[str] = set()
    binding_by_gold: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        _require(isinstance(binding, dict), f"{fixture_id} evidence binding must be an object")
        _require_keys(binding, EVIDENCE_BINDING_REQUIRED_KEYS, f"{fixture_id} evidence binding")
        binding_id = binding["id"]
        _require(binding_id not in binding_ids, f"{fixture_id} duplicate evidence binding id {binding_id}")
        binding_ids.add(binding_id)
        _require(isinstance(binding["evidence_id"], str) and binding["evidence_id"], f"{binding_id} evidence_id required")
        gold_id = binding["gold_id"]
        candidate_id = binding["candidate_id"]
        _require(gold_id in gold_ids, f"{fixture_id} evidence binding unknown gold_id {gold_id}")
        _require(candidate_id in candidate_ids, f"{fixture_id} evidence binding unknown candidate_id {candidate_id}")
        _require(isinstance(binding["valid"], bool), f"{binding_id} valid must be bool")
        _require(isinstance(binding["fully_entailed"], bool), f"{binding_id} fully_entailed must be bool")
        _require(gold_id not in binding_by_gold, f"{fixture_id} duplicate evidence binding for gold {gold_id}")
        binding_by_gold[gold_id] = binding
    return binding_by_gold


def _validate_verifier_rows(fixture_id: str, rows: list[dict[str, Any]]) -> None:
    row_ids: set[str] = set()
    for row in rows:
        _require(isinstance(row, dict), f"{fixture_id} verifier row must be an object")
        _require_keys(row, VERIFIER_ROW_REQUIRED_KEYS, f"{fixture_id} verifier row")
        row_id = row["id"]
        _require(row_id not in row_ids, f"{fixture_id} duplicate verifier row id {row_id}")
        row_ids.add(row_id)
        _require(row["status"] in SEMANTIC_STATUSES, f"{row_id} invalid verifier status")


def validate_fixture(fixture: dict[str, Any]) -> None:
    _require(isinstance(fixture, dict), "fixture must be an object")
    _require_keys(fixture, FIXTURE_REQUIRED_KEYS, "fixture")
    fixture_id = fixture["id"]
    _require(fixture_id in FROZEN_FIXTURE_IDS, f"unknown fixture id {fixture_id!r}")
    draft = fixture["draft_text"]
    _require(isinstance(draft, str) and draft, f"{fixture_id} draft_text required")
    _validate_sha256(fixture["draft_sha256"], f"{fixture_id} draft_sha256")
    _require(fixture["draft_sha256"] == sha256_text(draft), f"{fixture_id} draft_sha256 is stale or invalid")

    gold_atoms = fixture["gold_atoms"]
    exclusions = fixture["exclusions"]
    _require(isinstance(gold_atoms, list), f"{fixture_id} gold_atoms must be a list")
    _require(isinstance(exclusions, list), f"{fixture_id} exclusions must be a list")

    gold_ids: set[str] = set()
    canonical_flags: dict[str, tuple[bool, bool, str]] = {}
    gold_spans: list[tuple[int, int, str]] = []
    for gold in gold_atoms:
        _require(isinstance(gold, dict), f"{fixture_id} gold atom must be an object")
        _require_keys(gold, GOLD_REQUIRED_KEYS, f"{fixture_id} gold atom")
        gold_id = gold["id"]
        _require(gold_id not in gold_ids, f"{fixture_id} duplicate gold id {gold_id}")
        gold_ids.add(gold_id)
        _require(gold["gold_semantic_status"] in SEMANTIC_STATUSES, f"{gold_id} invalid gold semantic status")
        _require(not gold["material"] or gold["factual"], f"{gold_id} material atom must be factual")
        _validate_span(gold["span"], gold["text"], draft, gold_id)
        flags = (gold["factual"], gold["material"], gold["gold_semantic_status"])
        previous = canonical_flags.get(gold["canonical_id"])
        if previous is None:
            canonical_flags[gold["canonical_id"]] = flags
        else:
            _require(previous == flags, f"{gold_id} canonical flags disagree with sibling rows")
        gold_spans.append((gold["span"][0], gold["span"][1], gold_id))

    exclusion_ids: set[str] = set()
    for exclusion in exclusions:
        _require(isinstance(exclusion, dict), f"{fixture_id} exclusion must be an object")
        _require_keys(exclusion, EXCLUSION_REQUIRED_KEYS, f"{fixture_id} exclusion")
        exclusion_id = exclusion["id"]
        _require(exclusion_id not in exclusion_ids, f"{fixture_id} duplicate exclusion id {exclusion_id}")
        _require(exclusion_id not in gold_ids, f"{exclusion_id} cannot also be a gold id")
        exclusion_ids.add(exclusion_id)
        _require(exclusion["reason"] in EXCLUSION_REASONS, f"{exclusion_id} unknown exclusion reason")
        _validate_span(exclusion["span"], exclusion["text"], draft, exclusion_id)
        start, end = exclusion["span"]
        for gold_start, gold_end, overlap_gold_id in gold_spans:
            _require(
                end <= gold_start or start >= gold_end,
                f"{exclusion_id} overlaps gold {overlap_gold_id}",
            )

    candidates = fixture["candidates"]
    matches = fixture["allowed_matches"]
    _require(isinstance(candidates, list), f"{fixture_id} candidates must be a list")
    _require(isinstance(matches, list), f"{fixture_id} allowed_matches must be a list")
    candidate_by_id = _validate_candidates_and_matches(fixture_id, gold_ids, draft, candidates, matches)

    bindings = fixture["evidence_bindings"]
    _require(isinstance(bindings, list), f"{fixture_id} evidence_bindings must be a list")
    _validate_evidence_bindings(fixture_id, gold_ids, set(candidate_by_id), bindings)

    verifier_rows = fixture["verifier_rows"]
    _require(isinstance(verifier_rows, list), f"{fixture_id} verifier_rows must be a list")
    _validate_verifier_rows(fixture_id, verifier_rows)

    automatic_route = fixture["automatic_route"]
    _require(isinstance(automatic_route, dict), f"{fixture_id} automatic_route must be an object")
    _require_keys(automatic_route, AUTOMATIC_ROUTE_REQUIRED_KEYS, f"{fixture_id} automatic_route")
    _require(automatic_route["decision"] in ROUTE_DECISIONS, f"{fixture_id} invalid automatic route decision")


def validate_pack(pack: dict[str, Any], *, require_frozen_catalog: bool = True) -> None:
    schema = load_frozen_schema(DEFAULT_SCHEMA)
    assert_schema_runtime_parity(schema)
    validate_against_frozen_schema(pack, schema)
    _validate_pack_semantics(pack, require_frozen_catalog=require_frozen_catalog)


def _validate_pack_semantics(pack: dict[str, Any], *, require_frozen_catalog: bool = True) -> None:
    _require(isinstance(pack, dict), "pack must be an object")
    _require_keys(pack, PACK_REQUIRED_KEYS, "pack")
    _require(pack["pack_id"] == PACK_ID, f"pack_id must be {PACK_ID}")
    _require(pack["schema_version"] == SCHEMA_VERSION, "schema_version must be 2")
    _require(pack["evaluator_id"] == EVALUATOR_ID, f"evaluator_id must be {EVALUATOR_ID}")
    fixtures = pack["fixtures"]
    _require(isinstance(fixtures, list), "fixtures must be a list")
    seen: set[str] = set()
    for fixture in fixtures:
        validate_fixture(fixture)
        _require(fixture["id"] not in seen, f"duplicate fixture id {fixture['id']}")
        seen.add(fixture["id"])
    if require_frozen_catalog:
        _require(len(fixtures) == 1, f"official pack must contain 1 fixture, got {len(fixtures)}")
        _require(tuple(fixture["id"] for fixture in fixtures) == FROZEN_FIXTURE_IDS, "official pack IDs must match frozen catalog")


def load_pack(path: Path | str = DEFAULT_FIXTURES, *, require_frozen_catalog: bool = True) -> dict[str, Any]:
    pack_path = Path(path)
    try:
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClaimSemanticsV2Error(f"unreadable fixture pack {pack_path}: {error}") from error
    validate_pack(pack, require_frozen_catalog=require_frozen_catalog)
    return pack


def _ephemeral_pack(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "pack_id": PACK_ID,
        "schema_version": SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "description": "ephemeral fixture evaluation envelope",
        "fixtures": [fixture],
    }


def _binding_for_gold(
    gold_id: str,
    binding_by_gold: dict[str, dict[str, Any]],
    matched_candidate_id: str | None,
) -> dict[str, Any] | None:
    binding = binding_by_gold.get(gold_id)
    if binding is None:
        return None
    if matched_candidate_id is not None and binding["candidate_id"] != matched_candidate_id:
        return None
    return binding


def _is_oracle_proven_false_verification(gold: dict[str, Any], candidate: dict[str, Any] | None) -> bool:
    if not gold["material"]:
        return False
    if gold["gold_semantic_status"] not in {"weak", "unverified"}:
        return False
    return candidate is not None and candidate["predicted_semantic_status"] == "verified"


def _material_gold_is_unresolved(
    gold: dict[str, Any],
    matched_candidate: dict[str, Any] | None,
    binding: dict[str, Any] | None,
) -> bool:
    if not gold["material"]:
        return False
    if matched_candidate is None:
        return True
    if not _candidate_is_match_eligible(matched_candidate):
        return True
    if _is_oracle_proven_false_verification(gold, matched_candidate):
        return True
    if binding is None:
        return True
    if not binding["valid"]:
        return True
    if not binding["fully_entailed"]:
        return True
    if gold["gold_semantic_status"] == "unverified":
        return True
    if gold["gold_semantic_status"] == "weak":
        return True
    if matched_candidate["predicted_semantic_status"] != "verified":
        return True
    return False


def _material_gold_is_resolved(
    gold: dict[str, Any],
    matched_candidate: dict[str, Any] | None,
    binding: dict[str, Any] | None,
) -> bool:
    return gold["material"] and not _material_gold_is_unresolved(gold, matched_candidate, binding)


def compute_uvr_v1(verifier_rows: list[dict[str, Any]]) -> dict[str, Any]:
    unverified = sum(1 for row in verifier_rows if row["status"] == "unverified")
    verified = sum(1 for row in verifier_rows if row["status"] == "verified")
    weak = sum(1 for row in verifier_rows if row["status"] == "weak")
    denominator = verified + weak + unverified
    return ratio(unverified, denominator, "zero post-dedup emitted verifier rows")


def _compute_fixture_metrics(fixture: dict[str, Any]) -> dict[str, Any]:
    material_golds = [gold for gold in fixture["gold_atoms"] if gold["material"]]
    material_gold_ids = {gold["id"] for gold in material_golds}

    candidate_by_id = {candidate["id"]: candidate for candidate in fixture["candidates"]}
    binding_by_gold = {binding["gold_id"]: binding for binding in fixture["evidence_bindings"]}

    edges = [(edge["candidate_id"], edge["gold_id"]) for edge in fixture["allowed_matches"]]
    pairs = maximum_cardinality_matching(edges)
    gold_to_candidate = {gold_id: candidate_id for candidate_id, gold_id in pairs}

    matched_material_gold_ids: set[str] = set()
    for candidate_id, gold_id in pairs:
        if gold_id not in material_gold_ids:
            continue
        candidate = candidate_by_id[candidate_id]
        if not _candidate_is_match_eligible(candidate):
            continue
        matched_material_gold_ids.add(gold_id)

    material_recall = ratio(
        len(matched_material_gold_ids),
        len(material_golds),
        "zero material gold atoms",
    )

    unresolved_material = 0
    for gold in material_golds:
        candidate_id = gold_to_candidate.get(gold["id"])
        candidate = candidate_by_id.get(candidate_id) if candidate_id is not None else None
        binding = _binding_for_gold(gold["id"], binding_by_gold, candidate_id)
        if _material_gold_is_unresolved(gold, candidate, binding):
            unresolved_material += 1

    material_unresolved_rate = ratio(
        unresolved_material,
        len(material_golds),
        "zero material gold atoms",
    )

    classifier_negative_golds = [
        gold for gold in material_golds if gold["gold_semantic_status"] in {"weak", "unverified"}
    ]
    false_verifications = 0
    for gold in classifier_negative_golds:
        candidate_id = gold_to_candidate.get(gold["id"])
        candidate = candidate_by_id.get(candidate_id) if candidate_id is not None else None
        if candidate is not None and candidate["predicted_semantic_status"] == "verified":
            false_verifications += 1

    false_verification_rate = ratio(
        false_verifications,
        len(classifier_negative_golds),
        "zero eligible material weak-or-unverified gold cases",
    )

    oracle_semantic_pass = unresolved_material == 0
    automatic_route = fixture["automatic_route"]["decision"]

    return {
        "fixture_id": fixture["id"],
        "metrics": {
            "material_claim_recall.v2": material_recall,
            "material_claim_unresolved_rate.v1": material_unresolved_rate,
            "material_false_verification_rate.v1": false_verification_rate,
            "unverified_verifier_row_rate.UVR_v1": compute_uvr_v1(fixture["verifier_rows"]),
        },
        "oracle": {
            "semantic_pass": oracle_semantic_pass,
            "unresolved_material_atoms": unresolved_material,
            "material_gold_atoms": len(material_golds),
        },
        "automatic_route": {
            "decision": automatic_route,
        },
        "matching": {
            "pairs": [list(pair) for pair in pairs],
            "matched_material_gold_ids": sorted(matched_material_gold_ids),
        },
    }


def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    validate_pack(_ephemeral_pack(fixture), require_frozen_catalog=False)
    return _compute_fixture_metrics(fixture)


def evaluate_pack(
    pack: dict[str, Any],
    *,
    require_frozen_catalog: bool = True,
    fixture_id: str | None = None,
) -> dict[str, Any]:
    validate_pack(pack, require_frozen_catalog=require_frozen_catalog)
    results: list[dict[str, Any]] = []
    oracle_failing_assets = 0
    false_pass_assets = 0

    for fixture in pack["fixtures"]:
        if fixture_id is not None and fixture["id"] != fixture_id:
            continue
        result = _compute_fixture_metrics(fixture)
        results.append(result)
        if not result["oracle"]["semantic_pass"]:
            oracle_failing_assets += 1
            if result["automatic_route"]["decision"] == "PASS":
                false_pass_assets += 1

    if fixture_id is not None and not any(fixture["id"] == fixture_id for fixture in pack["fixtures"]):
        raise ClaimSemanticsV2Error(f"unknown fixture id {fixture_id}")

    results.sort(key=lambda item: item["fixture_id"])
    false_pass_rate = ratio(
        false_pass_assets,
        oracle_failing_assets,
        "zero oracle-failing assets",
    )

    for result in results:
        result["metrics"]["automatic_semantic_false_pass_rate.v1"] = false_pass_rate

    return {
        "evaluator_id": EVALUATOR_ID,
        "schema_version": SCHEMA_VERSION,
        "pack_id": PACK_ID,
        "fixture_count": len(pack["fixtures"]),
        "metrics": {
            "automatic_semantic_false_pass_rate.v1": false_pass_rate,
        },
        "results": results,
    }


def evaluate_default_pack(path: Path | str = DEFAULT_FIXTURES, **kwargs: Any) -> dict[str, Any]:
    return evaluate_pack(load_pack(path), **kwargs)


def load_metric_registry(path: Path | str = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_path = Path(path)
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClaimSemanticsV2Error(f"unreadable metric registry {registry_path}: {error}") from error
    _require(isinstance(registry, dict), "metric registry must be an object")
    _require(registry.get("registry_id") == "metric_registry_v1", "registry_id must be metric_registry_v1")
    metrics = registry.get("metrics")
    _require(isinstance(metrics, list) and len(metrics) == 5, "registry must contain exactly five metrics")
    names = {metric["canonical_name"] for metric in metrics}
    expected = {
        "material_claim_recall.v2",
        "material_claim_unresolved_rate.v1",
        "material_false_verification_rate.v1",
        "automatic_semantic_false_pass_rate.v1",
        "unverified_verifier_row_rate.UVR_v1",
    }
    _require(names == expected, f"registry metric identities must be {sorted(expected)}, got {sorted(names)}")
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the claim-semantics v2 oracle.")
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    parser.add_argument("--fixture", default=None)
    args = parser.parse_args(argv)
    try:
        report = evaluate_default_pack(args.fixtures, fixture_id=args.fixture)
        load_metric_registry()
    except ClaimSemanticsV2Error as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(normalize_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
