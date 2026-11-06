"""Phase 5B2C: Seed-first packing integration + combined exposure requalification.

Contiguous Structural Window Packing (CSWP) representation and Candidate C
bounded ±1 neighbor expansion stay frozen. The only change is packing priority:
all unique retrieved SEED units first, then deduplicated neighbors within the
2000 cl100k_base whole-unit budget.
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
    serialize_drafter_packed_evidence_v1,
    serialize_legacy_drafter_kb,
)
from scripts import phase5a2_shadow_ab as ab  # noqa: E402
from scripts import phase5b1_candidate_c as c  # noqa: E402
from scripts import phase5b2a_drafter_pack_contract as b2a  # noqa: E402
from scripts.phase5a2b_packed import eval_chunks  # noqa: E402

STARTING_HEAD = "41261a1a12ffba8a8051debc84b27055985c1b4f"
B1_CONTRACT = ROOT / "evals/fixtures/phase5b1_candidate_c_contract.json"
B2A_CONTRACT = ROOT / "evals/fixtures/phase5b2a_drafter_pack_contract.json"
CONTRACT = ROOT / "evals/fixtures/phase5b2c_seed_first_pack_contract.json"
OUTPUT = ROOT / "reports/phase5/phase5b2c"
FOCUS = ("Q09", "Q13", "Q17", "Q19", "Q21", "Q22", "Q28", "Q30", "Q31", "Q32", "Q34")
OUT_OF_SCOPE = ("Q19", "Q21", "Q30")


class SeedFirstPackError(RuntimeError):
    """A seed-first pack identity, exposure, or determinism gate failed."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return ab.sha256_bytes(Path(path).read_bytes())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_starting_head():
    if git("rev-parse", "HEAD") != STARTING_HEAD:
        raise SeedFirstPackError("starting HEAD mismatch")


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1)]


def token_summary(values):
    ordered = [int(v) for v in values]
    return {
        "min": min(ordered, default=0),
        "median": int(statistics.median(ordered)) if ordered else 0,
        "p95": int(percentile(ordered, 95)),
        "max": max(ordered, default=0),
        "mean": round(statistics.fmean(ordered), 6) if ordered else 0.0,
    }


def packed_contract_metrics(query, spans, packed, units, chunks):
    return c.layer_metrics(query, spans, packed, units, chunks)


def evaluate_once(oracle, evidence, control, units, by_source, chunks, seeds, encoding):
    spans_by = defaultdict(list)
    for span in evidence:
        spans_by[span.query_id].append(span)
    queries = {q["query_id"]: q for q in oracle["queries"]}
    per_query = []
    provenance_failures = []
    seeds_exceeding_2000 = []

    for seed_row in seeds:
        qid = seed_row["query_id"]
        query = queries[qid]
        spans = spans_by[qid]
        retrieved = [
            {
                "chunk_id": seed["chunk_id"],
                "relation": "SEED",
                "seed_rank": seed["rank"],
                "seed_chunk_id": seed["chunk_id"],
            }
            for seed in seed_row["seeds"]
        ]
        groups = c.expand_seeds(seed_row["seeds"], units, by_source)
        expanded_groups, _, _ = c.dedupe_groups(groups)
        expanded = c.flatten(expanded_groups)

        try:
            legacy_packed, _, legacy_used, legacy_exhausted = c.pack_units(
                expanded, units, c.PACK_BUDGET_CL100K, encoding
            )
        except c.CandidateCError:
            legacy_packed, legacy_used = [], 0

        try:
            packed, skipped, used, exhausted, pack_stats = c.pack_units_seed_first(
                expanded, units, c.PACK_BUDGET_CL100K, encoding
            )
            c.assert_seed_preservation_invariant(
                expanded, units, c.PACK_BUDGET_CL100K, encoding
            )
        except c.CandidateCError as exc:
            if "seed-only cl100k tokens" in str(exc):
                seeds_exceeding_2000.append(qid)
            raise

        retrieved_metrics = c.layer_metrics(query, spans, retrieved, units, chunks)
        expanded_metrics = c.layer_metrics(query, spans, expanded, units, chunks)
        seed_first_packed = c.layer_metrics(query, spans, packed, units, chunks)
        legacy_packed_metrics = c.layer_metrics(
            query, spans, legacy_packed, units, chunks
        )
        packed_contract = packed_contract_metrics(query, spans, packed, units, chunks)
        legacy_drafter = c.seed_group_layer(
            query, spans, packed, units, chunks, LEGACY_DRAFTER_K, LEGACY_DRAFTER_CHAR_LIMIT
        )
        verifier = c.seed_group_layer(
            query, spans, packed, units, chunks, c.VERIFIER_K, None
        )

        serialized, identity_ok, fingerprint = b2a.packed_evidence_identity(packed, units)
        if not identity_ok:
            provenance_failures.append(qid)
        if not b2a.verify_no_secondary_clip(serialized):
            provenance_failures.append(qid + ":secondary_clip")
        if not b2a.verify_no_top3_gate(serialized, packed):
            provenance_failures.append(qid + ":top3_gate")

        legacy_serialized = serialize_legacy_drafter_kb(packed, units)
        legacy_groups, legacy_chars, legacy_tokens = b2a.serialized_context_stats(
            legacy_serialized
        )
        contract_groups, contract_chars, contract_tokens = b2a.serialized_context_stats(
            serialized
        )

        vs_expanded = b2a.classify_vs_packed(
            expanded_metrics["covered_spans"], seed_first_packed["covered_spans"]
        )

        per_query.append(
            {
                "query_id": qid,
                "query": query["query"],
                "gating_eligible": query["gating_eligible"],
                "exposure_recall": {
                    "retrieved": retrieved_metrics["evidence_span_recall"],
                    "expanded": expanded_metrics["evidence_span_recall"],
                    "seed_first_packed": seed_first_packed["evidence_span_recall"],
                    "legacy_frozen_order_packed": legacy_packed_metrics[
                        "evidence_span_recall"
                    ],
                    "packed_contract_drafter": packed_contract["evidence_span_recall"],
                    "legacy_drafter": legacy_drafter["evidence_span_recall"],
                    "verifier_exposed": verifier["evidence_span_recall"],
                },
                "vs_expanded": vs_expanded,
                "pack": {
                    "policy": pack_stats["policy"],
                    "used_cl100k": used,
                    "legacy_frozen_order_used_cl100k": legacy_used,
                    "seeds_retained": pack_stats["seeds_retained"],
                    "neighbors_retained": pack_stats["neighbors_retained"],
                    "seed_only_cl100k": pack_stats["seed_only_cl100k"],
                    "exhausted": exhausted,
                    "skipped_units": skipped,
                },
                "context": {
                    "seed_first_packed": {
                        "groups": len(serialize_drafter_packed_evidence_v1(packed, units)),
                        "cl100k_tokens": seed_first_packed["cl100k_tokens"],
                        "characters": sum(
                            len(units[row["chunk_id"]]["retrieval_text"]) for row in packed
                        ),
                    },
                    "legacy_frozen_order_packed": {
                        "cl100k_tokens": legacy_packed_metrics["cl100k_tokens"],
                    },
                    "legacy_drafter": {
                        "groups": legacy_groups,
                        "characters": legacy_chars,
                        "cl100k_tokens": legacy_tokens,
                    },
                    "packed_contract_drafter": {
                        "groups": contract_groups,
                        "characters": contract_chars,
                        "cl100k_tokens": contract_tokens,
                    },
                },
                "identity": {
                    "packed_fingerprint": fingerprint,
                    "identity_preserved": identity_ok,
                    "verifier_evidence_set_parity": sorted(
                        packed_contract["covered_spans"]
                    )
                    == sorted(verifier["covered_spans"]),
                },
            }
        )

    gating = [row for row in per_query if row["gating_eligible"]]

    def mean_recall(key):
        return round(
            sum(row["exposure_recall"][key] for row in gating) / len(gating), 8
        )

    recalls = {
        "retrieved": mean_recall("retrieved"),
        "expanded": mean_recall("expanded"),
        "seed_first_packed": mean_recall("seed_first_packed"),
        "packed_contract_drafter": mean_recall("packed_contract_drafter"),
        "verifier_exposed": mean_recall("verifier_exposed"),
        "legacy_frozen_order_packed": mean_recall("legacy_frozen_order_packed"),
        "legacy_drafter": mean_recall("legacy_drafter"),
    }

    regressions = sorted(
        row["query_id"]
        for row in gating
        if row["vs_expanded"]["classification"] == "REGRESSED"
    )

    seed_first_tokens = [row["pack"]["used_cl100k"] for row in gating]
    legacy_order_tokens = [
        row["pack"]["legacy_frozen_order_used_cl100k"] for row in gating
    ]
    seeds_retained = [row["pack"]["seeds_retained"] for row in gating]
    neighbors_retained = [row["pack"]["neighbors_retained"] for row in gating]
    groups_per_query = [
        row["context"]["seed_first_packed"]["groups"] for row in gating
    ]

    legacy_drafter_tokens = [
        row["context"]["legacy_drafter"]["cl100k_tokens"] for row in gating
    ]
    contract_drafter_tokens = [
        row["context"]["packed_contract_drafter"]["cl100k_tokens"] for row in gating
    ]
    legacy_ctx = token_summary(legacy_drafter_tokens)
    contract_ctx = token_summary(contract_drafter_tokens)
    prompt_growth = {
        "cl100k_tokens_median_ratio": round(
            contract_ctx["median"] / max(legacy_ctx["median"], 1), 6
        ),
        "cl100k_tokens_p95_ratio": round(
            contract_ctx["p95"] / max(legacy_ctx["p95"], 1), 6
        ),
        "cl100k_tokens_max_ratio": round(
            contract_ctx["max"] / max(legacy_ctx["max"], 1), 6
        ),
    }

    focus = {
        qid: next(row for row in per_query if row["query_id"] == qid) for qid in FOCUS
    }

    ready = (
        recalls["seed_first_packed"] == recalls["expanded"]
        == recalls["packed_contract_drafter"] == recalls["verifier_exposed"]
        and not regressions
        and not provenance_failures
        and not seeds_exceeding_2000
        and focus["Q22"]["exposure_recall"]["seed_first_packed"] == 1.0
        and all(
            row["identity"]["verifier_evidence_set_parity"] for row in per_query
        )
    )

    return {
        "schema_version": "phase5b2c_seed_first_pack_v1_results",
        "name": "Seed-first packing + combined exposure requalification",
        "required_parent": STARTING_HEAD,
        "pack_policy": "SEED_FIRST",
        "shadow_contract": DRAFTER_PACKED_EVIDENCE_V1,
        "production_drafter_changed": False,
        "production_retrieval_changed": False,
        "verifier_changed": False,
        "aggregate_gating_33": {
            "exposure_recall": recalls,
            "seed_first_packed_tokens_cl100k": token_summary(seed_first_tokens),
            "legacy_frozen_order_packed_tokens_cl100k": token_summary(legacy_order_tokens),
            "mean_seeds_retained": round(statistics.fmean(seeds_retained), 6),
            "mean_neighbors_retained": round(statistics.fmean(neighbors_retained), 6),
            "groups_per_query": token_summary(groups_per_query),
            "packed_contract_drafter_vs_legacy_drafter": {
                "legacy_drafter_context": legacy_ctx,
                "packed_contract_drafter_context": contract_ctx,
                "prompt_growth_estimate": prompt_growth,
            },
        },
        "forensics_summary": {
            "seeds_exceeding_2000": seeds_exceeding_2000,
            "regressions_vs_expanded": regressions,
            "provenance_failures": provenance_failures,
        },
        "focus": {
            qid: {
                "retrieved": row["exposure_recall"]["retrieved"],
                "expanded": row["exposure_recall"]["expanded"],
                "seed_first_packed": row["exposure_recall"]["seed_first_packed"],
                "packed_contract_drafter": row["exposure_recall"][
                    "packed_contract_drafter"
                ],
                "verifier_exposed": row["exposure_recall"]["verifier_exposed"],
                "vs_expanded": row["vs_expanded"]["classification"],
            }
            for qid, row in focus.items()
        },
        "quality_gate": {
            "seed_preservation_invariant": not seeds_exceeding_2000,
            "q22_seed_first_recovery": focus["Q22"]["exposure_recall"][
                "seed_first_packed"
            ]
            == 1.0,
            "exposure_layers_aligned": recalls["seed_first_packed"]
            == recalls["packed_contract_drafter"]
            == recalls["verifier_exposed"],
            "no_regressions_vs_expanded": not regressions,
            "combined_contract_ready_for_live_wiring": ready,
        },
        "per_query": per_query,
        "provider_calls": 0,
        "external_network_calls": 0,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
    }


def freeze():
    require_starting_head()
    if CONTRACT.exists():
        raise SeedFirstPackError("contract already frozen; refusing overwrite")
    contract = {
        "schema_version": "phase5b2c_seed_first_pack_v1",
        "name": "Seed-first packing integration (Candidate C v1)",
        "required_parent": STARTING_HEAD,
        "upstream_contracts": {
            "phase5b1_candidate_c": read(B1_CONTRACT)["contract_sha256"],
            "phase5b2a_drafter_pack": read(B2A_CONTRACT)["contract_sha256"],
        },
        "pack_policy": {
            "id": "SEED_FIRST",
            "budget_cl100k": c.PACK_BUDGET_CL100K,
            "tokenizer": "cl100k_base",
            "pass_1": (
                "unique original frozen top-5 retrieved SEED units in retrieval-rank order"
            ),
            "pass_2": (
                "remaining budget for deduplicated ±1 neighbors ordered by "
                "originating seed rank, source order, chunk_id"
            ),
            "invariant": (
                "supplementary expansion context may never evict an original "
                "retrieved seed when the seed set fits the global budget"
            ),
            "fail_closed": "seed-only cl100k total exceeding 2000",
        },
        "exposure_layers": [
            "retrieved",
            "expanded",
            "seed_first_packed",
            "packed_contract_drafter",
            "verifier_exposed",
        ],
        "shadow_contract": DRAFTER_PACKED_EVIDENCE_V1,
        "production_drafter_changed": False,
        "production_retrieval_changed": False,
        "verifier_changed": False,
        "provider_calls": 0,
        "external_network_calls": 0,
        "input_file_sha256": {
            "scripts/phase5b1_candidate_c.py": file_hash(
                ROOT / "scripts/phase5b1_candidate_c.py"
            ),
            "scripts/phase5b2a_drafter_pack_contract.py": file_hash(
                ROOT / "scripts/phase5b2a_drafter_pack_contract.py"
            ),
            "scripts/phase5b2c_seed_first_pack.py": file_hash(
                ROOT / "scripts/phase5b2c_seed_first_pack.py"
            ),
            "agent/drafter_packed_evidence.py": file_hash(
                ROOT / "agent/drafter_packed_evidence.py"
            ),
        },
    }
    contract["contract_sha256"] = ab.sha256_json(contract)
    ab.write_json(CONTRACT, contract)
    print(ab.canonical_json({"contract_sha256": contract["contract_sha256"]}))


def verify_contract(contract):
    payload = {k: v for k, v in contract.items() if k != "contract_sha256"}
    if ab.sha256_json(payload) != contract["contract_sha256"]:
        raise SeedFirstPackError("frozen contract digest mismatch")
    for name, expected in contract["input_file_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise SeedFirstPackError("frozen input drift: " + name)


def run(_output):
    contract = read(CONTRACT)
    verify_contract(contract)
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
    encoding = c.cl100k()
    result = evaluate_once(
        oracle, evidence, control, units, by_source, chunks, seeds, encoding
    )
    result["contract_sha256"] = contract["contract_sha256"]
    return result


def evaluate():
    results = []
    for index in (1, 2):
        directory = OUTPUT / f"run-{index}"
        if (directory / "results.json").exists():
            raise SeedFirstPackError("refusing to overwrite an experiment result")
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
        raise SeedFirstPackError("deterministic rerun mismatch")
    result, fingerprint = results[0]
    summary = {
        "result_sha256": fingerprint,
        "deterministic_rerun": True,
        "aggregate_gating_33": result["aggregate_gating_33"],
        "forensics_summary": result["forensics_summary"],
        "focus": result["focus"],
        "quality_gate": result["quality_gate"],
        "provider_calls": 0,
        "external_network_calls": 0,
        "production_drafter_changed": False,
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
