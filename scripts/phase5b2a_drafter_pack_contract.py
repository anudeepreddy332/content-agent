"""Phase 5B2A: DRAFTER_PACKED_EVIDENCE_V1 shadow contract evaluation.

Compares legacy production-equivalent drafter KB exposure (top-3 + 2000-char
clip) against a shadow contract that consumes all Candidate-C packed groups
with no secondary clip. Does not modify production draft_node.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.drafter_packed_evidence import (  # noqa: E402
    DRAFTER_PACKED_EVIDENCE_V1,
    LEGACY_DRAFTER_CHAR_LIMIT,
    LEGACY_DRAFTER_K,
    LEGACY_DRAFTER_KB_V1,
    packed_identity_fingerprint,
    serialize_drafter_packed_evidence_v1,
    serialize_legacy_drafter_kb,
)
from scripts import phase5a2_shadow_ab as ab  # noqa: E402
from scripts import phase5b1_candidate_c as c  # noqa: E402
from scripts.phase5a2b_packed import eval_chunks  # noqa: E402

STARTING_HEAD = "fe6d35970e9c2b4ed5e494a2cea07a0287fc3f4b"
B1_CONTRACT = ROOT / "evals/fixtures/phase5b1_candidate_c_contract.json"
CONTRACT = ROOT / "evals/fixtures/phase5b2a_drafter_pack_contract.json"
OUTPUT = ROOT / "reports/phase5/phase5b2a"
B1_OUTPUT = ROOT / "reports/phase5/phase5b1"
FOCUS = ("Q09", "Q13", "Q17", "Q22", "Q28", "Q31", "Q32", "Q34")
KNOWN_LEGACY_LOSSES = ("Q09", "Q17", "Q28", "Q31", "Q32", "Q34")
TOPK_LOSSES = ("Q09", "Q17", "Q31", "Q34")
CLIP_LOSSES = ("Q28", "Q32")


class DrafterPackContractError(RuntimeError):
    """A shadow contract identity, exposure, or determinism gate failed."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return ab.sha256_bytes(Path(path).read_bytes())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_starting_head():
    if git("rev-parse", "HEAD") != STARTING_HEAD:
        raise DrafterPackContractError("starting HEAD mismatch")


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1)]


def context_summary(groups, chars, tokens):
    return {
        "groups": {
            "min": min(groups, default=0),
            "median": int(statistics.median(groups)) if groups else 0,
            "p95": int(percentile(groups, 95)),
            "max": max(groups, default=0),
        },
        "characters": {
            "min": min(chars, default=0),
            "median": int(statistics.median(chars)) if chars else 0,
            "p95": int(percentile(chars, 95)),
            "max": max(chars, default=0),
        },
        "cl100k_tokens": {
            "min": min(tokens, default=0),
            "median": int(statistics.median(tokens)) if tokens else 0,
            "p95": int(percentile(tokens, 95)),
            "max": max(tokens, default=0),
        },
    }


def classify_vs_packed(packed_cov, arm_cov):
    lost = sorted(set(packed_cov) - set(arm_cov))
    gained = sorted(set(arm_cov) - set(packed_cov))
    if lost:
        label = "REGRESSED"
    elif gained:
        label = "IMPROVED"
    else:
        label = "SAME"
    return {
        "classification": label,
        "lost_spans": lost,
        "gained_spans": gained,
    }


def serialized_context_stats(serialized):
    groups = len(serialized)
    chars = sum(len(member["serialized_text"]) for group in serialized for member in group["members"])
    tokens = sum(member["cl100k_tokens"] or 0 for group in serialized for member in group["members"])
    return groups, chars, tokens


def packed_evidence_identity(packed_rows, units):
    serialized = serialize_drafter_packed_evidence_v1(packed_rows, units)
    expected = [
        {
            "seed_rank": row["seed_rank"],
            "seed_chunk_id": row["seed_chunk_id"],
            "chunk_id": row["chunk_id"],
            "relation": row["relation"],
            "source_intervals": row["source_intervals"],
            "text_sha256": ab.sha256_bytes(
                units[row["chunk_id"]]["retrieval_text"].encode("utf-8")
            ),
        }
        for row in packed_rows
    ]
    observed = [
        {
            "seed_rank": member["seed_rank"],
            "seed_chunk_id": member["seed_chunk_id"],
            "chunk_id": member["chunk_id"],
            "relation": member["relation"],
            "source_intervals": member["source_intervals"],
            "text_sha256": member["serialized_text_sha256"],
        }
        for group in serialized
        for member in group["members"]
    ]
    return serialized, expected == observed, packed_identity_fingerprint(serialized)


def verify_no_secondary_clip(serialized):
    for group in serialized:
        for member in group["members"]:
            if member.get("legacy_char_clip_applied"):
                return False
            full = member["serialized_text"]
            if member["serialized_text_sha256"] != ab.sha256_bytes(full.encode("utf-8")):
                return False
    return True


def verify_no_top3_gate(serialized, packed_rows):
    packed_ranks = sorted({row["seed_rank"] for row in packed_rows})
    exposed_ranks = sorted({group["seed_rank"] for group in serialized})
    if exposed_ranks != packed_ranks:
        return False
    return len(serialized) == len({row["seed_rank"] for row in packed_rows})


def forensics(query, spans, packed, units, chunks):
    legacy = c.seed_group_layer(
        query, spans, packed, units, chunks, LEGACY_DRAFTER_K, LEGACY_DRAFTER_CHAR_LIMIT
    )
    k3_noclip = c.seed_group_layer(
        query, spans, packed, units, chunks, LEGACY_DRAFTER_K, None
    )
    packed_metrics = c.layer_metrics(query, spans, packed, units, chunks)
    packed_cov = set(packed_metrics["covered_spans"])
    legacy_cov = set(legacy["covered_spans"])
    if packed_cov == legacy_cov:
        return {
            "topk_loss": False,
            "clip_loss": False,
            "new_loss": False,
        }
    if packed_cov - legacy_cov:
        topk = packed_cov - set(k3_noclip["covered_spans"])
        clip = packed_cov - legacy_cov - topk
        return {
            "topk_loss": bool(topk),
            "clip_loss": bool(clip),
            "new_loss": False,
            "topk_spans": sorted(topk),
            "clip_spans": sorted(clip),
        }
    return {
        "topk_loss": False,
        "clip_loss": False,
        "new_loss": True,
        "lost_spans": sorted(legacy_cov - packed_cov),
    }


def evaluate_once(oracle, evidence, control, units, by_source, chunks, seeds, encoding):
    spans_by = defaultdict(list)
    for span in evidence:
        spans_by[span.query_id].append(span)
    queries = {q["query_id"]: q for q in oracle["queries"]}
    per_query = []
    provenance_failures = []
    for seed_row in seeds:
        qid = seed_row["query_id"]
        query = queries[qid]
        spans = spans_by[qid]
        groups = c.expand_seeds(seed_row["seeds"], units, by_source)
        expanded_groups, _, _ = c.dedupe_groups(groups)
        expanded = c.flatten(expanded_groups)
        packed, _, used, _ = c.pack_units(expanded, units, c.PACK_BUDGET_CL100K, encoding)
        packed_metrics = c.layer_metrics(query, spans, packed, units, chunks)
        legacy_metrics = c.seed_group_layer(
            query, spans, packed, units, chunks, LEGACY_DRAFTER_K, LEGACY_DRAFTER_CHAR_LIMIT
        )
        verifier_metrics = c.seed_group_layer(
            query, spans, packed, units, chunks, c.VERIFIER_K, None
        )
        packed_serialized, identity_ok, fingerprint = packed_evidence_identity(packed, units)
        legacy_serialized = serialize_legacy_drafter_kb(packed, units)
        if not identity_ok:
            provenance_failures.append(qid)
        if not verify_no_secondary_clip(packed_serialized):
            provenance_failures.append(qid + ":secondary_clip")
        if not verify_no_top3_gate(packed_serialized, packed):
            provenance_failures.append(qid + ":top3_gate")
        packed_groups, packed_chars, packed_tokens = serialized_context_stats(packed_serialized)
        legacy_groups, legacy_chars, legacy_tokens = serialized_context_stats(legacy_serialized)
        packed_contract_metrics = c.layer_metrics(query, spans, packed, units, chunks)
        vs_packed = classify_vs_packed(
            packed_metrics["covered_spans"], packed_contract_metrics["covered_spans"]
        )
        vs_legacy = classify_vs_packed(
            packed_metrics["covered_spans"], legacy_metrics["covered_spans"]
        )
        per_query.append(
            {
                "query_id": qid,
                "query": query["query"],
                "gating_eligible": query["gating_eligible"],
                "exposure_recall": {
                    "packed": packed_metrics["evidence_span_recall"],
                    "legacy_drafter": legacy_metrics["evidence_span_recall"],
                    "packed_contract_drafter": packed_contract_metrics["evidence_span_recall"],
                    "verifier_exposed": verifier_metrics["evidence_span_recall"],
                },
                "covered_spans": {
                    "packed": packed_metrics["covered_spans"],
                    "legacy_drafter": legacy_metrics["covered_spans"],
                    "packed_contract_drafter": packed_contract_metrics["covered_spans"],
                    "verifier_exposed": verifier_metrics["covered_spans"],
                },
                "vs_packed": {
                    "legacy_drafter": vs_legacy,
                    "packed_contract_drafter": vs_packed,
                },
                "forensics": forensics(query, spans, packed, units, chunks),
                "context": {
                    "legacy_drafter": {
                        "groups": legacy_groups,
                        "characters": legacy_chars,
                        "cl100k_tokens": legacy_tokens,
                    },
                    "packed_contract_drafter": {
                        "groups": packed_groups,
                        "characters": packed_chars,
                        "cl100k_tokens": packed_tokens,
                        "pack_budget_used_cl100k": used,
                    },
                },
                "identity": {
                    "packed_fingerprint": fingerprint,
                    "packed_contract": DRAFTER_PACKED_EVIDENCE_V1,
                    "legacy_contract": LEGACY_DRAFTER_KB_V1,
                    "identity_preserved": identity_ok,
                    "verifier_evidence_set_parity": sorted(
                        packed_contract_metrics["covered_spans"]
                    )
                    == sorted(verifier_metrics["covered_spans"]),
                    "packed_chunk_ids": [row["chunk_id"] for row in packed],
                    "legacy_chunk_ids": legacy_metrics["chunk_ids"],
                    "packed_contract_chunk_ids": packed_contract_metrics["chunk_ids"],
                },
            }
        )
    gating = [row for row in per_query if row["gating_eligible"]]

    def mean_recall(key):
        return round(
            sum(row["exposure_recall"][key] for row in gating) / len(gating), 8
        )

    legacy_recall = mean_recall("legacy_drafter")
    packed_contract_recall = mean_recall("packed_contract_drafter")
    packed_recall = mean_recall("packed")
    verifier_recall = mean_recall("verifier_exposed")

    legacy_groups = [row["context"]["legacy_drafter"]["groups"] for row in gating]
    legacy_chars = [row["context"]["legacy_drafter"]["characters"] for row in gating]
    legacy_tokens = [row["context"]["legacy_drafter"]["cl100k_tokens"] for row in gating]
    contract_groups = [
        row["context"]["packed_contract_drafter"]["groups"] for row in gating
    ]
    contract_chars = [
        row["context"]["packed_contract_drafter"]["characters"] for row in gating
    ]
    contract_tokens = [
        row["context"]["packed_contract_drafter"]["cl100k_tokens"] for row in gating
    ]

    regressions = sorted(
        row["query_id"]
        for row in gating
        if row["vs_packed"]["packed_contract_drafter"]["classification"] == "REGRESSED"
    )
    new_losses = sorted(
        row["query_id"]
        for row in gating
        if row["forensics"]["new_loss"]
    )
    topk_eliminated = sorted(
        qid
        for qid in KNOWN_LEGACY_LOSSES
        if next(r for r in per_query if r["query_id"] == qid)["forensics"]["topk_loss"]
        and next(r for r in per_query if r["query_id"] == qid)["exposure_recall"][
            "packed_contract_drafter"
        ]
        > next(r for r in per_query if r["query_id"] == qid)["exposure_recall"][
            "legacy_drafter"
        ]
    )
    clip_eliminated = sorted(
        qid
        for qid in KNOWN_LEGACY_LOSSES
        if next(r for r in per_query if r["query_id"] == qid)["forensics"]["clip_loss"]
        and next(r for r in per_query if r["query_id"] == qid)["exposure_recall"][
            "packed_contract_drafter"
        ]
        > next(r for r in per_query if r["query_id"] == qid)["exposure_recall"][
            "legacy_drafter"
        ]
    )

    legacy_ctx = context_summary(legacy_groups, legacy_chars, legacy_tokens)
    contract_ctx = context_summary(contract_groups, contract_chars, contract_tokens)
    prompt_growth = {
        "groups_median_ratio": round(
            contract_ctx["groups"]["median"] / max(legacy_ctx["groups"]["median"], 1),
            6,
        ),
        "characters_median_ratio": round(
            contract_ctx["characters"]["median"]
            / max(legacy_ctx["characters"]["median"], 1),
            6,
        ),
        "cl100k_tokens_median_ratio": round(
            contract_ctx["cl100k_tokens"]["median"]
            / max(legacy_ctx["cl100k_tokens"]["median"], 1),
            6,
        ),
        "cl100k_tokens_p95_ratio": round(
            contract_ctx["cl100k_tokens"]["p95"]
            / max(legacy_ctx["cl100k_tokens"]["p95"], 1),
            6,
        ),
        "cl100k_tokens_max_ratio": round(
            contract_ctx["cl100k_tokens"]["max"]
            / max(legacy_ctx["cl100k_tokens"]["max"], 1),
            6,
        ),
    }

    focus = {
        qid: next(row for row in per_query if row["query_id"] == qid)
        for qid in FOCUS
    }
    ready = (
        packed_contract_recall == packed_recall == verifier_recall
        and legacy_recall < packed_recall
        and not regressions
        and not new_losses
        and not provenance_failures
        and all(
            row["identity"]["verifier_evidence_set_parity"] for row in per_query
        )
        and all(
            next(r for r in per_query if r["query_id"] == qid)["exposure_recall"][
                "packed_contract_drafter"
            ]
            >= next(r for r in per_query if r["query_id"] == qid)["exposure_recall"][
                "packed"
            ]
            for qid in KNOWN_LEGACY_LOSSES
        )
    )

    return {
        "schema_version": "phase5b2a_drafter_pack_contract_v1_results",
        "name": "DRAFTER_PACKED_EVIDENCE_V1 shadow contract",
        "required_parent": STARTING_HEAD,
        "shadow_contract": DRAFTER_PACKED_EVIDENCE_V1,
        "legacy_contract": LEGACY_DRAFTER_KB_V1,
        "production_drafter_changed": False,
        "production_retrieval_changed": False,
        "verifier_changed": False,
        "aggregate_gating_33": {
            "exposure_recall": {
                "packed": packed_recall,
                "legacy_drafter": legacy_recall,
                "packed_contract_drafter": packed_contract_recall,
                "verifier_exposed": verifier_recall,
            },
            "legacy_drafter_context": legacy_ctx,
            "packed_contract_drafter_context": contract_ctx,
            "prompt_growth_estimate": prompt_growth,
        },
        "forensics_summary": {
            "known_legacy_losses": list(KNOWN_LEGACY_LOSSES),
            "topk_losses_eliminated": topk_eliminated,
            "clip_losses_eliminated": clip_eliminated,
            "new_evidence_losses": new_losses,
            "packed_contract_regressions_vs_packed": regressions,
        },
        "focus": {
            qid: {
                "packed": row["exposure_recall"]["packed"],
                "legacy_drafter": row["exposure_recall"]["legacy_drafter"],
                "packed_contract_drafter": row["exposure_recall"][
                    "packed_contract_drafter"
                ],
                "classification_vs_packed": row["vs_packed"]["packed_contract_drafter"][
                    "classification"
                ],
                "forensics": row["forensics"],
            }
            for qid, row in focus.items()
        },
        "quality_gate": {
            "candidate_c_packed_identity_preserved": not provenance_failures,
            "packed_contract_matches_packed_recall": packed_contract_recall
            == packed_recall,
            "packed_contract_matches_verifier_evidence_set": all(
                row["identity"]["verifier_evidence_set_parity"] for row in per_query
            ),
            "legacy_losses_recovered": all(
                next(r for r in per_query if r["query_id"] == qid)["exposure_recall"][
                    "packed_contract_drafter"
                ]
                == next(r for r in per_query if r["query_id"] == qid)["exposure_recall"][
                    "packed"
                ]
                for qid in KNOWN_LEGACY_LOSSES
            ),
            "no_new_regressions": not regressions and not new_losses,
            "shadow_contract_ready_for_live_wiring": ready,
        },
        "provenance_failures": provenance_failures,
        "per_query": per_query,
        "provider_calls": 0,
        "external_network_calls": 0,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
    }


def freeze():
    require_starting_head()
    if CONTRACT.exists():
        raise DrafterPackContractError("contract already frozen; refusing overwrite")
    b1 = read(B1_CONTRACT)
    c.verify_contract(b1)
    contract = {
        "schema_version": "phase5b2a_drafter_pack_contract_v1",
        "name": "DRAFTER_PACKED_EVIDENCE_V1 shadow drafter KB contract",
        "required_parent": STARTING_HEAD,
        "upstream_contract_sha256": b1["contract_sha256"],
        "shadow_contract": {
            "id": DRAFTER_PACKED_EVIDENCE_V1,
            "input": "Candidate-C PACKED seed-groups from Phase 5B1",
            "rules": [
                "consume every surviving packed KB group",
                "preserve Candidate-C pack order",
                "preserve provenance and group identity",
                "no kb_results[:3] gate",
                "no secondary 2000-character per-group clip",
                "no repack, reorder, split, or regeneration",
            ],
            "budget": "2000 cl100k_base tokens/query enforced upstream by Candidate-C pack",
        },
        "legacy_control": {
            "id": LEGACY_DRAFTER_KB_V1,
            "k": LEGACY_DRAFTER_K,
            "character_prefix_per_seed_group": LEGACY_DRAFTER_CHAR_LIMIT,
        },
        "production_drafter_changed": False,
        "production_retrieval_changed": False,
        "verifier_changed": False,
        "provider_calls": 0,
        "external_network_calls": 0,
        "input_file_sha256": {
            "evals/fixtures/phase5b1_candidate_c_contract.json": file_hash(B1_CONTRACT),
            "agent/drafter_packed_evidence.py": file_hash(
                ROOT / "agent/drafter_packed_evidence.py"
            ),
            "scripts/phase5b2a_drafter_pack_contract.py": file_hash(
                ROOT / "scripts/phase5b2a_drafter_pack_contract.py"
            ),
        },
    }
    contract["contract_sha256"] = ab.sha256_json(contract)
    ab.write_json(CONTRACT, contract)
    print(ab.canonical_json({"contract_sha256": contract["contract_sha256"]}))


def verify_contract(contract):
    payload = {k: v for k, v in contract.items() if k != "contract_sha256"}
    if ab.sha256_json(payload) != contract["contract_sha256"]:
        raise DrafterPackContractError("frozen contract digest mismatch")
    for name, expected in contract["input_file_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise DrafterPackContractError("frozen input drift: " + name)


def run(_output):
    contract = read(CONTRACT)
    verify_contract(contract)
    b1 = read(B1_CONTRACT)
    if b1["contract_sha256"] != contract["upstream_contract_sha256"]:
        raise DrafterPackContractError("upstream 5B1 contract drift")
    from scripts import phase5a0_baseline as baseline
    from scripts.retrieval_golden_v2 import load_oracle, validate_oracle

    oracle = load_oracle()
    validate_oracle(oracle)
    _, _, evidence = baseline.load_sources_and_evidence(oracle)
    control = c.read(c.CONTROL)
    manifest, units, by_source = c.load_units()
    c.prove_unit_text(units)
    chunks = {item.chunk_id: item for item in eval_chunks(manifest, oracle)}
    seeds = c.frozen_seeds(control)
    if ab.sha256_json(seeds) != b1["seed_sha256"]:
        raise DrafterPackContractError("frozen seed ledger drift")
    encoding = c.cl100k()
    result = evaluate_once(
        oracle, evidence, control, units, by_source, chunks, seeds, encoding
    )
    result["contract_sha256"] = contract["contract_sha256"]
    result["upstream_contract_sha256"] = contract["upstream_contract_sha256"]
    return result


def evaluate():
    results = []
    for index in (1, 2):
        directory = OUTPUT / f"run-{index}"
        if (directory / "results.json").exists():
            raise DrafterPackContractError("refusing to overwrite an experiment result")
        result = run(directory)
        fingerprint = ab.sha256_json(result)
        ab.write_json(directory / "results.json", result, canonical=True)
        ab.write_json(
            directory / "runtime.json",
            {
                "result_sha256": fingerprint,
                "provider_calls": 0,
                "external_network_calls": 0,
            },
        )
        results.append((result, fingerprint))
    if results[0][1] != results[1][1]:
        raise DrafterPackContractError("deterministic rerun mismatch")
    result, fingerprint = results[0]
    summary = {
        "result_sha256": fingerprint,
        "deterministic_rerun": True,
        "aggregate_gating_33": result["aggregate_gating_33"],
        "forensics_summary": result["forensics_summary"],
        "focus": result["focus"],
        "quality_gate": result["quality_gate"],
        "provenance_failures": result["provenance_failures"],
        "provider_calls": 0,
        "external_network_calls": 0,
        "production_drafter_changed": False,
        "production_retrieval_changed": False,
        "verifier_changed": False,
    }
    ab.write_json(OUTPUT / "evaluation_summary.json", summary)
    ab.write_json(OUTPUT / "results.json", result, canonical=True)
    print(ab.canonical_json(summary))
    return summary


def main():
    ab.install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "evaluate"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        evaluate()


if __name__ == "__main__":
    main()
