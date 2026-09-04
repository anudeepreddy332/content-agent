"""Stage 2D evidence-exposure provider preflight harness. No provider calls by default."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import tiktoken

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_FIXTURES = REPO_ROOT / "evals" / "fixtures" / "evidence_exposure_2d.json"
DEFAULT_PRICE_SCHEDULE = REPO_ROOT / "evals" / "evidence_exposure_2d_price_schedule.json"
DEFAULT_2C_FIXTURES = REPO_ROOT / "evals" / "fixtures" / "evidence_exposure_2c.json"
VERIFY_SYSTEM_PATH = REPO_ROOT / "prompts" / "verify_system.md"

PACK_ID = "evidence_exposure_2d"
EVALUATOR_ID = "evidence_exposure_2d_preflight"
SCHEMA_VERSION = 1
STAGE = "2d"

HARD_SPEND_CEILING_USD = 0.08
MAX_OUTPUT_TOKENS = 4000
VERIFY_TEMPERATURE = 0.1
MAX_ATTEMPTS_PER_CELL = 1
PROVIDER_RETRIES_DISABLED = True

ASSET_IDS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
CELL_IDS = (
    "P1-PREFIX",
    "P1-COMPLETE",
    "P2-COMPLETE",
    "P3-COMPLETE",
    "P4-COMPLETE",
    "P5-COMPLETE",
    "P6-PREFIX",
    "P6-COMPLETE",
    "P7-PREFIX",
    "P7-COMPLETE",
)
PAIRED_ASSETS = ("P1", "P6", "P7")
EXPOSURE_ARMS = frozenset({"prefix", "complete"})


class EvidenceExposure2DError(ValueError):
    """Stage 2D pack or preflight asset is not evaluable."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceExposure2DError(message)


def provider_execution_authorized() -> bool:
    return os.getenv("EVIDENCE_EXPOSURE_2D_EXECUTE", "").strip() == "1"


def load_2c_case(case_id: str) -> dict[str, Any]:
    pack = json.loads(DEFAULT_2C_FIXTURES.read_text(encoding="utf-8"))
    for case in pack["cases"]:
        if case["id"] == case_id:
            return case
    raise EvidenceExposure2DError(f"2C case {case_id} not found")


def expose_source_context(*, source_kind: str, source_text: str, source_url: str, exposure_arm: str) -> str:
    from scripts.evaluate_evidence_exposure_2c import expose_arm_a, expose_arm_b

    if exposure_arm == "prefix":
        exposed, _, _ = expose_arm_a(source_kind, source_text)
    elif exposure_arm == "complete":
        exposed, _, _ = expose_arm_b(source_text)
    else:
        raise EvidenceExposure2DError(f"unknown exposure arm {exposure_arm!r}")
    label = "WEB" if source_kind == "web" else "KB"
    return f"[{label}] {source_url}\n{exposed}"


def load_verify_system() -> str:
    return VERIFY_SYSTEM_PATH.read_text(encoding="utf-8")


def build_verifier_user_message(draft_text: str, source_context: str) -> str:
    return f"""
Draft to verify:
{draft_text}

Available sources:
{source_context}

Return a JSON array. Each element:
{{"claim": "...", "source_url": "..." or null, "confidence": 0.0-1.0,
  "status": "verified" | "weak" | "unverified",
  "specificity": "substantive" | "generic"}}

Return ONLY the JSON array. No preamble.
""".strip()


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def shadow_runtime_acceptance(
    grounding_report: list[dict[str, Any]],
    *,
    verification_status: str = "completed",
    uvr_threshold: float = 0.15,
) -> dict[str, Any]:
    """Shadow current-runtime UVR gate without mutating production routing."""
    if verification_status != "completed":
        return {"accepted": False, "reason": "verification_status_not_completed", "uvr": None}
    if not grounding_report:
        return {"accepted": False, "reason": "empty_grounding_report", "uvr": None}
    unverified = sum(1 for row in grounding_report if row.get("status") == "unverified")
    uvr = unverified / len(grounding_report)
    accepted = uvr <= uvr_threshold
    return {
        "accepted": accepted,
        "reason": "uvr_within_threshold" if accepted else "uvr_above_threshold",
        "uvr": uvr,
        "uvr_threshold": uvr_threshold,
    }


def evaluate_semantic_oracle(semantic_fixture: dict[str, Any]) -> dict[str, Any]:
    from scripts.evaluate_claim_semantics_v2 import _compute_fixture_metrics

    return _compute_fixture_metrics(canonicalize_semantic_fixture(semantic_fixture))


def canonicalize_semantic_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Ensure v2 schema-required fields for ephemeral oracle evaluation."""
    fid = fixture["id"]
    out = json.loads(json.dumps(fixture))
    for item in out.get("gold_atoms", []):
        item.setdefault("canonical_id", f"{fid}.ATOM.{item['id']}")
    for item in out.get("candidates", []):
        item.setdefault("canonical_id", f"{fid}.C.{item['id']}")
    return out


def _source_record(
    *,
    source_id: str,
    source_kind: str,
    source_url: str,
    source_text: str,
    source_rank: int = 1,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_kind": source_kind,
        "source_url": source_url,
        "source_rank": source_rank,
        "source_text": source_text,
        "source_sha256": sha256_text(source_text),
        "source_character_length": len(source_text),
    }


def build_frozen_pack() -> dict[str, Any]:
    w02 = load_2c_case("E2C-W02")
    w03 = load_2c_case("E2C-W03")
    k03 = load_2c_case("E2C-K03")

    p2_source = (
        "FluxCap array service note FC-881. "
        "FluxCap array stores at most 4096 records per shard under nominal load."
    )
    p3_source = (
        "Project NEXUS-7 internal field manual. "
        "NEXUS-7 bounded replay retains exactly 512 events per shard."
    )
    p4_source = (
        "Inventory ledger IL-44. There are 10 items in partition A. "
        "There are 15 items in partition B."
    )
    p5_source = "Warehouse count sheet WC-12. There are 10 items in the active bin."

    assets: list[dict[str, Any]] = []

    p1_draft = "Quorum latch engages only after seven replica acknowledgements."
    p1_source = _source_record(
        source_id="SRC-P1-W02",
        source_kind=w02["source_kind"],
        source_url=w02["source_url"],
        source_text=w02["source_text"],
    )
    assets.append(
        {
            "asset_id": "P1",
            "title": "late supported material claim",
            "source": p1_source,
            "draft_text": p1_draft,
            "draft_sha256": sha256_text(p1_draft),
            "cells": [
                {"cell_id": "P1-PREFIX", "exposure_arm": "prefix"},
                {"cell_id": "P1-COMPLETE", "exposure_arm": "complete"},
            ],
            "expected_corrected_semantic_pass": {"P1-PREFIX": False, "P1-COMPLETE": True},
            "expected_primary_label_complete": "verified",
            "semantic_fixture": {
                "id": "ADV02",
                "title": "P1 late supported claim",
                "draft_text": p1_draft,
                "draft_sha256": sha256_text(p1_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p1_draft,
                        "span": [0, len(p1_draft)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "verified",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p1_draft,
                        "span": [0, len(p1_draft)],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    }
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P1.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P1.EV.quorum",
                        "valid": True,
                        "fully_entailed": True,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p1_draft,
                        "span": [0, len(p1_draft)],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "verified",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "verified",
                        "binding": {
                            "id": "P1.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P1.EV.quorum",
                            "valid": True,
                            "fully_entailed": True,
                        },
                    }
                ],
                "verifier_rows": [{"id": "P1.VR.c1", "status": "verified"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P1.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "verified",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    }
                ],
            },
        }
    )

    p2_draft = "FluxCap array stores 8192 records per shard under nominal load."
    p2_src = _source_record(
        source_id="SRC-P2",
        source_kind="kb",
        source_url="kb://fluxcap/array-fc881",
        source_text=p2_source,
    )
    assets.append(
        {
            "asset_id": "P2",
            "title": "unsupported material claim",
            "source": p2_src,
            "draft_text": p2_draft,
            "draft_sha256": sha256_text(p2_draft),
            "cells": [{"cell_id": "P2-COMPLETE", "exposure_arm": "complete"}],
            "expected_corrected_semantic_pass": {"P2-COMPLETE": False},
            "expected_primary_label_complete": "unverified",
            "semantic_fixture": {
                "id": "ADV03",
                "title": "P2 unsupported claim",
                "draft_text": p2_draft,
                "draft_sha256": sha256_text(p2_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p2_draft,
                        "span": [0, len(p2_draft)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "unverified",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p2_draft,
                        "span": [0, len(p2_draft)],
                        "roles": [],
                        "predicted_semantic_status": "unverified",
                    }
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P2.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P2.EV.fluxcap",
                        "valid": False,
                        "fully_entailed": False,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p2_draft,
                        "span": [0, len(p2_draft)],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "unverified",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "unverified",
                        "binding": {
                            "id": "P2.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P2.EV.fluxcap",
                            "valid": False,
                            "fully_entailed": False,
                        },
                    }
                ],
                "verifier_rows": [{"id": "P2.VR.c1", "status": "unverified"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P2.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "unverified",
                        "predicted_semantic_status": "unverified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    }
                ],
            },
        }
    )

    p3_draft = "NEXUS-7 bounded replay retains thousands of events per shard."
    p3_src = _source_record(
        source_id="SRC-P3",
        source_kind="web",
        source_url="https://example.test/nexus-7/manual",
        source_text=p3_source,
    )
    assets.append(
        {
            "asset_id": "P3",
            "title": "partial weak material claim",
            "source": p3_src,
            "draft_text": p3_draft,
            "draft_sha256": sha256_text(p3_draft),
            "cells": [{"cell_id": "P3-COMPLETE", "exposure_arm": "complete"}],
            "expected_corrected_semantic_pass": {"P3-COMPLETE": False},
            "expected_primary_label_complete": "weak",
            "semantic_fixture": {
                "id": "ADV04",
                "title": "P3 partial weak claim",
                "draft_text": p3_draft,
                "draft_sha256": sha256_text(p3_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p3_draft,
                        "span": [0, len(p3_draft)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "weak",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p3_draft,
                        "span": [0, len(p3_draft)],
                        "roles": [],
                        "predicted_semantic_status": "weak",
                    }
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P3.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P3.EV.nexus",
                        "valid": True,
                        "fully_entailed": False,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p3_draft,
                        "span": [0, len(p3_draft)],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "weak",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "weak",
                        "binding": {
                            "id": "P3.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P3.EV.nexus",
                            "valid": True,
                            "fully_entailed": False,
                        },
                    }
                ],
                "verifier_rows": [{"id": "P3.VR.c1", "status": "weak"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P3.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "weak",
                        "predicted_semantic_status": "weak",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    }
                ],
            },
        }
    )

    p4_draft = "There are 10 items in partition A. There are 15 items in partition B."
    p4_src = _source_record(
        source_id="SRC-P4",
        source_kind="kb",
        source_url="kb://inventory/ledger-il44",
        source_text=p4_source,
    )
    assets.append(
        {
            "asset_id": "P4",
            "title": "supported-new material claim",
            "source": p4_src,
            "draft_text": p4_draft,
            "draft_sha256": sha256_text(p4_draft),
            "cells": [{"cell_id": "P4-COMPLETE", "exposure_arm": "complete"}],
            "expected_corrected_semantic_pass": {"P4-COMPLETE": True},
            "expected_primary_label_complete": "verified",
            "semantic_fixture": {
                "id": "ADV05",
                "title": "P4 supported-new claim",
                "draft_text": p4_draft,
                "draft_sha256": sha256_text(p4_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": "There are 10 items in partition A.",
                        "span": [0, 34],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "verified",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": "There are 10 items in partition A.",
                        "span": [0, 34],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    },
                    {
                        "id": "c2",
                        "text": "There are 15 items in partition B.",
                        "span": [35, 69],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    },
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P4.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P4.EV.ledger",
                        "valid": True,
                        "fully_entailed": True,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": "There are 10 items in partition A.",
                        "span": [0, 34],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "verified",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "verified",
                        "binding": {
                            "id": "P4.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P4.EV.ledger",
                            "valid": True,
                            "fully_entailed": True,
                        },
                    },
                    {
                        "id": "f2",
                        "text": "There are 15 items in partition B.",
                        "span": [35, 69],
                        "material": True,
                        "reference_relationship": "unmatched",
                        "required_gold_id": None,
                        "independent_semantic_label": "verified",
                        "prediction_id": "c2",
                        "predicted_semantic_status": "verified",
                        "binding": {
                            "id": "P4.FBIND.f2",
                            "target_final_claim_id": "f2",
                            "evidence_id": "P4.EV.ledger",
                            "valid": True,
                            "fully_entailed": True,
                        },
                    },
                ],
                "verifier_rows": [
                    {"id": "P4.VR.c1", "status": "verified"},
                    {"id": "P4.VR.c2", "status": "verified"},
                ],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P4.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "verified",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    },
                    {
                        "id": "P4.CLASS.f2",
                        "source": "unmatched-final",
                        "material": True,
                        "independent_semantic_label": "verified",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c2",
                        "required_gold_id": None,
                        "final_atom_id": "f2",
                    },
                ],
            },
        }
    )

    p5_draft = "The active bin is labeled WC-12."
    p5_gold = "There are 10 items in the active bin."
    p5_src = _source_record(
        source_id="SRC-P5",
        source_kind="web",
        source_url="https://example.test/warehouse/wc12",
        source_text=p5_source,
    )
    assets.append(
        {
            "asset_id": "P5",
            "title": "required material omission",
            "source": p5_src,
            "draft_text": p5_draft,
            "draft_sha256": sha256_text(p5_draft),
            "cells": [{"cell_id": "P5-COMPLETE", "exposure_arm": "complete"}],
            "expected_corrected_semantic_pass": {"P5-COMPLETE": False},
            "expected_primary_label_complete": "verified",
            "semantic_fixture": {
                "id": "ADV06",
                "title": "P5 omission",
                "draft_text": p5_draft,
                "draft_sha256": sha256_text(p5_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p5_gold,
                        "span": [0, len(p5_gold)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "verified",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p5_draft,
                        "span": [0, len(p5_draft)],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    }
                ],
                "allowed_matches": [],
                "evidence_bindings": [],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p5_draft,
                        "span": [0, len(p5_draft)],
                        "material": False,
                        "reference_relationship": "unmatched",
                        "required_gold_id": None,
                        "independent_semantic_label": "verified",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "verified",
                        "binding": None,
                    }
                ],
                "verifier_rows": [{"id": "P5.VR.c1", "status": "verified"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P5.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "verified",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": None,
                    }
                ],
            },
        }
    )

    p6_draft = "Helix gate permits export when mode flag HG-ENABLE is set."
    p6_source = _source_record(
        source_id="SRC-P6-W03",
        source_kind=w03["source_kind"],
        source_url=w03["source_url"],
        source_text=w03["source_text"],
    )
    assets.append(
        {
            "asset_id": "P6",
            "title": "late contradiction",
            "source": p6_source,
            "draft_text": p6_draft,
            "draft_sha256": sha256_text(p6_draft),
            "cells": [
                {"cell_id": "P6-PREFIX", "exposure_arm": "prefix"},
                {"cell_id": "P6-COMPLETE", "exposure_arm": "complete"},
            ],
            "expected_corrected_semantic_pass": {"P6-PREFIX": False, "P6-COMPLETE": False},
            "expected_primary_label_complete": "weak",
            "semantic_fixture": {
                "id": "ADV07",
                "title": "P6 contradiction",
                "draft_text": p6_draft,
                "draft_sha256": sha256_text(p6_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p6_draft,
                        "span": [0, len(p6_draft)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "weak",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p6_draft,
                        "span": [0, len(p6_draft)],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    }
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P6.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P6.EV.helix",
                        "valid": True,
                        "fully_entailed": False,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p6_draft,
                        "span": [0, len(p6_draft)],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "weak",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "verified",
                        "binding": {
                            "id": "P6.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P6.EV.helix",
                            "valid": True,
                            "fully_entailed": False,
                        },
                    }
                ],
                "verifier_rows": [{"id": "P6.VR.c1", "status": "verified"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P6.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "weak",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    }
                ],
            },
        }
    )

    p7_draft = "Cache tier T3 admits writes when isolation level IL-2 holds."
    p7_source = _source_record(
        source_id="SRC-P7-K03",
        source_kind=k03["source_kind"],
        source_url=k03["source_url"],
        source_text=k03["source_text"],
    )
    assets.append(
        {
            "asset_id": "P7",
            "title": "late qualifier",
            "source": p7_source,
            "draft_text": p7_draft,
            "draft_sha256": sha256_text(p7_draft),
            "cells": [
                {"cell_id": "P7-PREFIX", "exposure_arm": "prefix"},
                {"cell_id": "P7-COMPLETE", "exposure_arm": "complete"},
            ],
            "expected_corrected_semantic_pass": {"P7-PREFIX": False, "P7-COMPLETE": False},
            "expected_primary_label_complete": "weak",
            "semantic_fixture": {
                "id": "ADV08",
                "title": "P7 qualifier loss",
                "draft_text": p7_draft,
                "draft_sha256": sha256_text(p7_draft),
                "gold_atoms": [
                    {
                        "id": "g1",
                        "text": p7_draft,
                        "span": [0, len(p7_draft)],
                        "factual": True,
                        "material": True,
                        "gold_semantic_status": "weak",
                    }
                ],
                "exclusions": [],
                "candidates": [
                    {
                        "id": "c1",
                        "text": p7_draft,
                        "span": [0, len(p7_draft)],
                        "roles": [],
                        "predicted_semantic_status": "verified",
                    }
                ],
                "allowed_matches": [{"candidate_id": "c1", "gold_id": "g1"}],
                "evidence_bindings": [
                    {
                        "id": "P7.BIND.g1",
                        "gold_id": "g1",
                        "candidate_id": "c1",
                        "evidence_id": "P7.EV.cache",
                        "valid": True,
                        "fully_entailed": False,
                    }
                ],
                "final_atoms": [
                    {
                        "id": "f1",
                        "text": p7_draft,
                        "span": [0, len(p7_draft)],
                        "material": True,
                        "reference_relationship": "required-equivalent",
                        "required_gold_id": "g1",
                        "independent_semantic_label": "weak",
                        "prediction_id": "c1",
                        "predicted_semantic_status": "verified",
                        "binding": {
                            "id": "P7.FBIND.f1",
                            "target_final_claim_id": "f1",
                            "evidence_id": "P7.EV.cache",
                            "valid": True,
                            "fully_entailed": False,
                        },
                    }
                ],
                "verifier_rows": [{"id": "P7.VR.c1", "status": "verified"}],
                "automatic_route": {"decision": "PASS"},
                "fixed_classification_cases": [
                    {
                        "id": "P7.CLASS.g1",
                        "source": "required",
                        "material": True,
                        "independent_semantic_label": "weak",
                        "predicted_semantic_status": "verified",
                        "prediction_id": "c1",
                        "required_gold_id": "g1",
                        "final_atom_id": "f1",
                    }
                ],
            },
        }
    )

    verify_system = load_verify_system()
    pack = {
        "pack_id": PACK_ID,
        "schema_version": SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "stage": STAGE,
        "description": "Frozen Stage 2D provider preflight pack",
        "provider_execution_default": "disabled",
        "hard_spend_ceiling_usd": HARD_SPEND_CEILING_USD,
        "max_attempts_per_cell": MAX_ATTEMPTS_PER_CELL,
        "provider_retries_disabled": PROVIDER_RETRIES_DISABLED,
        "model_config": {
            "model_alias": "deepseek-chat",
            "temperature": VERIFY_TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
        },
        "prompt_identity": {
            "verify_system_path": "prompts/verify_system.md",
            "verify_system_sha256": sha256_text(verify_system),
        },
        "zero_tolerance_gates": {
            "material_false_verification_rate.v2_numerator": 0,
            "automatic_semantic_false_pass_rate.v2_numerator": 0,
        },
        "assets": assets,
    }
    validate_pack(pack)
    for asset in pack["assets"]:
        asset["semantic_fixture"] = canonicalize_semantic_fixture(asset["semantic_fixture"])
    return pack


def validate_pack(pack: dict[str, Any]) -> None:
    _require(pack["pack_id"] == PACK_ID, "pack_id mismatch")
    _require(pack["evaluator_id"] == EVALUATOR_ID, "evaluator_id mismatch")
    _require(pack["schema_version"] == SCHEMA_VERSION, "schema_version mismatch")
    _require(pack.get("stage") == STAGE, "stage must be 2d")
    assets = pack["assets"]
    _require(len(assets) == len(ASSET_IDS), "asset count mismatch")
    asset_ids: set[str] = set()
    cell_ids: set[str] = set()
    requirement_ids: set[str] = set()
    for asset in assets:
        _require(asset["asset_id"] in ASSET_IDS, "unknown asset id")
        _require(asset["asset_id"] not in asset_ids, f"duplicate asset id {asset['asset_id']}")
        asset_ids.add(asset["asset_id"])
        _require(asset["draft_sha256"] == sha256_text(asset["draft_text"]), "draft hash mismatch")
        source = asset["source"]
        _require(source["source_sha256"] == sha256_text(source["source_text"]), "source hash mismatch")
        _require(source["source_rank"] == 1, "source rank must be 1")
        for cell in asset["cells"]:
            cid = cell["cell_id"]
            _require(cid not in cell_ids, f"duplicate cell id {cid}")
            cell_ids.add(cid)
            _require(cell["exposure_arm"] in EXPOSURE_ARMS, "invalid exposure arm")
        sem = asset["semantic_fixture"]
        for req_key in ("gold_atoms", "final_atoms", "fixed_classification_cases"):
            for item in sem.get(req_key, []):
                rid = item.get("id")
                if rid:
                    full = f"{asset['asset_id']}:{rid}"
                    _require(full not in requirement_ids, f"duplicate semantic identity {full}")
                    requirement_ids.add(full)
    _require(asset_ids == set(ASSET_IDS), "asset catalog mismatch")
    _require(cell_ids == set(CELL_IDS), "cell catalog mismatch")


def load_pack(path: Path | str = DEFAULT_FIXTURES) -> dict[str, Any]:
    pack = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_pack(pack)
    return pack


def find_asset(pack: dict[str, Any], asset_id: str) -> dict[str, Any]:
    for asset in pack["assets"]:
        if asset["asset_id"] == asset_id:
            return asset
    raise EvidenceExposure2DError(f"asset {asset_id} not found")


def find_cell(asset: dict[str, Any], cell_id: str) -> dict[str, Any]:
    for cell in asset["cells"]:
        if cell["cell_id"] == cell_id:
            return cell
    raise EvidenceExposure2DError(f"cell {cell_id} not found in asset {asset['asset_id']}")


def build_cell_request(pack: dict[str, Any], cell_id: str) -> dict[str, Any]:
    for asset in pack["assets"]:
        for cell in asset["cells"]:
            if cell["cell_id"] == cell_id:
                source = asset["source"]
                verify_system = load_verify_system()
                exposed_context = expose_source_context(
                    source_kind=source["source_kind"],
                    source_text=source["source_text"],
                    source_url=source["source_url"],
                    exposure_arm=cell["exposure_arm"],
                )
                user_message = build_verifier_user_message(asset["draft_text"], exposed_context)
                messages = [
                    {"role": "system", "content": verify_system},
                    {"role": "user", "content": user_message},
                ]
                return {
                    "cell_id": cell_id,
                    "asset_id": asset["asset_id"],
                    "call_order": CELL_IDS.index(cell_id) + 1,
                    "source_id": source["source_id"],
                    "source_sha256": source["source_sha256"],
                    "complete_source_text": source["source_text"],
                    "draft_text": asset["draft_text"],
                    "draft_sha256": asset["draft_sha256"],
                    "exposure_arm": cell["exposure_arm"],
                    "exposed_verifier_context": exposed_context,
                    "exposed_context_sha256": sha256_text(exposed_context),
                    "verify_system_sha256": sha256_text(verify_system),
                    "messages": messages,
                    "model_config": pack["model_config"],
                    "max_attempts": MAX_ATTEMPTS_PER_CELL,
                    "provider_retries_disabled": PROVIDER_RETRIES_DISABLED,
                    "expected_corrected_semantic_pass": asset["expected_corrected_semantic_pass"][cell_id],
                    "telemetry_contract": [
                        "cell_id",
                        "execution_git_sha",
                        "clean_state_attestation",
                        "source_sha256",
                        "complete_source_text",
                        "exposure_arm",
                        "exposed_verifier_context",
                        "exposed_context_sha256",
                        "verify_system_message",
                        "verify_user_message",
                        "prompt_hashes",
                        "draft_sha256",
                        "raw_provider_response",
                        "provider_response_id",
                        "requested_model_alias",
                        "returned_model_identity",
                        "finish_reason",
                        "parsed_claims",
                        "parsed_statuses",
                        "parse_errors",
                        "evidence_bindings",
                        "semantic_oracle_inputs_outputs",
                        "shadow_runtime_decision",
                        "input_tokens",
                        "output_tokens",
                        "cache_tokens",
                        "price_schedule_id",
                        "calculated_cost_usd",
                        "actual_cost_usd",
                        "latency_ms",
                        "timestamps",
                        "artifact_sha256",
                    ],
                }
    raise EvidenceExposure2DError(f"cell {cell_id} not found")


def build_all_requests(pack: dict[str, Any]) -> list[dict[str, Any]]:
    return [build_cell_request(pack, cell_id) for cell_id in CELL_IDS]


def load_price_schedule(path: Path | str = DEFAULT_PRICE_SCHEDULE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def estimate_cell_cost_usd(request: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    input_tokens = sum(count_tokens(message["content"]) for message in request["messages"])
    output_tokens = schedule["max_output_tokens_per_cell"]
    input_cost = input_tokens * schedule["input_cost_per_million_tokens_usd"] / 1_000_000
    output_cost = output_tokens * schedule["output_cost_per_million_tokens_usd"] / 1_000_000
    return {
        "cell_id": request["cell_id"],
        "input_tokens": input_tokens,
        "output_tokens_bound": output_tokens,
        "conservative_max_cost_usd": round(input_cost + output_cost, 6),
    }


def preflight_budget(
    pack: dict[str, Any],
    *,
    schedule_path: Path | str = DEFAULT_PRICE_SCHEDULE,
    ceiling_usd: float = HARD_SPEND_CEILING_USD,
) -> dict[str, Any]:
    schedule = load_price_schedule(schedule_path)
    requests = build_all_requests(pack)
    estimates = [estimate_cell_cost_usd(request, schedule) for request in requests]
    total = round(sum(item["conservative_max_cost_usd"] for item in estimates), 6)
    authorized = total <= ceiling_usd
    return {
        "schedule_id": schedule["schedule_id"],
        "cell_estimates": estimates,
        "conservative_max_spend_usd": total,
        "hard_ceiling_usd": ceiling_usd,
        "budget_authorized": authorized,
        "provider_execution_authorized_flag": provider_execution_authorized(),
    }


def verify_paired_isolation(pack: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for asset_id in PAIRED_ASSETS:
        asset = find_asset(pack, asset_id)
        prefix_cell = next(cell for cell in asset["cells"] if cell["exposure_arm"] == "prefix")
        complete_cell = next(cell for cell in asset["cells"] if cell["exposure_arm"] == "complete")
        prefix_req = build_cell_request(pack, prefix_cell["cell_id"])
        complete_req = build_cell_request(pack, complete_cell["cell_id"])
        same = (
            prefix_req["draft_sha256"] == complete_req["draft_sha256"]
            and prefix_req["source_sha256"] == complete_req["source_sha256"]
            and prefix_req["model_config"] == complete_req["model_config"]
            and prefix_req["exposure_arm"] != complete_req["exposure_arm"]
            and prefix_req["exposed_context_sha256"] != complete_req["exposed_context_sha256"]
        )
        results[asset_id] = {
            "isolated": same,
            "prefix_context_sha256": prefix_req["exposed_context_sha256"],
            "complete_context_sha256": complete_req["exposed_context_sha256"],
        }
    return results


def execute_provider_if_authorized(pack: dict[str, Any], cell_id: str) -> dict[str, Any]:
    if not provider_execution_authorized():
        raise EvidenceExposure2DError(
            "provider execution disabled; set EVIDENCE_EXPOSURE_2D_EXECUTE=1 after independent preflight review"
        )
    raise EvidenceExposure2DError("provider execution path not enabled in pre-provider checkpoint")


def write_frozen_fixtures(path: Path | str = DEFAULT_FIXTURES) -> None:
    pack = build_frozen_pack()
    Path(path).write_text(json.dumps(pack, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run_preflight(pack: dict[str, Any]) -> dict[str, Any]:
    requests = build_all_requests(pack)
    budget = preflight_budget(pack)
    paired = verify_paired_isolation(pack)
    semantic_checks = {}
    for asset in pack["assets"]:
        result = evaluate_semantic_oracle(asset["semantic_fixture"])
        semantic_checks[asset["asset_id"]] = {
            "semantic_pass": result["oracle"]["semantic_pass"],
            "expected_by_cell": asset["expected_corrected_semantic_pass"],
        }
    return {
        "pack_id": pack["pack_id"],
        "stage": STAGE,
        "cell_count": len(requests),
        "asset_count": len(pack["assets"]),
        "requests_built": [request["cell_id"] for request in requests],
        "budget": budget,
        "paired_isolation": paired,
        "semantic_template_checks": semantic_checks,
        "provider_execution_default_disabled": not provider_execution_authorized(),
        "preflight_ready": budget["budget_authorized"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2D evidence exposure preflight")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--write-fixtures", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.write_fixtures:
        write_frozen_fixtures(args.fixtures)
        print(f"wrote {args.fixtures}")
        return 0
    pack = load_pack(args.fixtures)
    report = run_preflight(pack)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(f"preflight_ready={report['preflight_ready']} cells={report['cell_count']}")
    return 0 if report["preflight_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
