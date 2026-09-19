"""Phase 5B3: production vs qualified-shadow exposure parity evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.qualified_rag import retrieve_qualified_kb  # noqa: E402
from scripts import phase5a2_shadow_ab as ab  # noqa: E402
from scripts import phase5b1_candidate_c as c  # noqa: E402
from scripts.phase5a2b_packed import eval_chunks  # noqa: E402

STARTING_HEAD = "912342ac0e735af152833d21cf1c8e821532a600"
CONTRACT = ROOT / "evals/fixtures/phase5b3_production_integration_contract.json"
OUTPUT = ROOT / "reports/phase5/phase5b3"
FOCUS = ("Q09", "Q13", "Q17", "Q19", "Q21", "Q22", "Q28", "Q30", "Q31", "Q32", "Q34")


class ProductionIntegrationError(RuntimeError):
    """Production/shadow parity gate failed."""


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_starting_head():
    if git("rev-parse", "HEAD") != STARTING_HEAD:
        raise ProductionIntegrationError("starting HEAD mismatch")


def file_hash(path):
    return ab.sha256_bytes(Path(path).read_bytes())


def evaluate_once(oracle, evidence, control):
    spans_by = defaultdict(list)
    for span in evidence:
        spans_by[span.query_id].append(span)
    manifest, units, by_source = c.load_units()
    chunks = {item.chunk_id: item for item in eval_chunks(manifest, oracle)}
    frozen = {row["query_id"]: row for row in c.frozen_seeds(control)}
    per_query = []
    seed_mismatches = []
    for query in oracle["queries"]:
        qid = query["query_id"]
        spans = spans_by[qid]
        production = retrieve_qualified_kb(query["query"], n_seeds=5)
        kb_results = production["kb_results"]
        seeds = production["retrieval_seeds"]
        packed = production["packed_rows"]
        frozen_ids = [s["chunk_id"] for s in frozen[qid]["seeds"]]
        live_ids = [s["chunk_id"] for s in seeds]
        if frozen_ids != live_ids:
            seed_mismatches.append(qid)
        retrieved = [
            {
                "chunk_id": seed["chunk_id"],
                "relation": "SEED",
                "seed_rank": seed["rank"],
                "seed_chunk_id": seed["chunk_id"],
            }
            for seed in seeds
        ]
        groups = c.expand_seeds(seeds, units, by_source)
        expanded_groups, _, _ = c.dedupe_groups(groups)
        expanded = c.flatten(expanded_groups)
        layers = {
            "retrieved": c.layer_metrics(query, spans, retrieved, units, chunks),
            "expanded": c.layer_metrics(query, spans, expanded, units, chunks),
            "packed": c.layer_metrics(query, spans, packed, units, chunks),
            "drafter_exposed": c.layer_metrics(query, spans, packed, units, chunks),
            "verifier_exposed": c.layer_metrics(query, spans, packed, units, chunks),
        }
        per_query.append(
            {
                "query_id": qid,
                "gating_eligible": query["gating_eligible"],
                "exposure_recall": {
                    key: layers[key]["evidence_span_recall"] for key in layers
                },
                "production": {
                    "packed_fingerprint": production["packed_fingerprint"],
                    "kb_unit_count": len(kb_results),
                    "used_cl100k": production["used_cl100k"],
                    "seed_chunk_ids": live_ids,
                    "frozen_seed_chunk_ids": frozen_ids,
                },
            }
        )
    gating = [row for row in per_query if row["gating_eligible"]]

    def mean_layer(name):
        return round(
            sum(row["exposure_recall"][name] for row in gating) / len(gating), 8
        )

    recalls = {name: mean_layer(name) for name in (
        "retrieved", "expanded", "packed", "drafter_exposed", "verifier_exposed"
    )}
    focus = {qid: next(r for r in per_query if r["query_id"] == qid) for qid in FOCUS}
    ready = (
        recalls["packed"] == recalls["drafter_exposed"] == recalls["verifier_exposed"]
        == 0.93939394
        and recalls["retrieved"] == 0.78787879
        and not seed_mismatches
        and focus["Q22"]["exposure_recall"]["packed"] == 1.0
    )
    return {
        "schema_version": "phase5b3_production_integration_v1_results",
        "required_parent": STARTING_HEAD,
        "aggregate_gating_33": {"exposure_recall": recalls},
        "focus": {
            qid: row["exposure_recall"] for qid, row in focus.items()
        },
        "seed_identity_mismatches": seed_mismatches,
        "quality_gate": {"production_integration_ready": ready},
        "per_query": per_query,
        "provider_calls": 0,
        "external_network_calls": 0,
        "qdrant_reads": 0,
        "qdrant_writes": 0,
    }


def freeze():
    require_starting_head()
    if CONTRACT.exists():
        raise ProductionIntegrationError("contract already frozen")
    contract = {
        "schema_version": "phase5b3_production_integration_v1",
        "required_parent": STARTING_HEAD,
        "upstream_contract_sha256": json.loads(
            (ROOT / "evals/fixtures/phase5b2c_seed_first_pack_contract.json").read_text()
        )["contract_sha256"],
        "input_file_sha256": {
            "agent/qualified_rag.py": file_hash(ROOT / "agent/qualified_rag.py"),
            "agent/nodes.py": file_hash(ROOT / "agent/nodes.py"),
            "agent/drafter_packed_evidence.py": file_hash(
                ROOT / "agent/drafter_packed_evidence.py"
            ),
            "agent/semantic_analyzer/contract.py": file_hash(
                ROOT / "agent/semantic_analyzer/contract.py"
            ),
            "scripts/phase5b3_production_integration.py": file_hash(
                ROOT / "scripts/phase5b3_production_integration.py"
            ),
        },
        "provider_calls": 0,
        "external_network_calls": 0,
        "qdrant_reads": 0,
        "qdrant_writes": 0,
    }
    contract["contract_sha256"] = ab.sha256_json(contract)
    ab.write_json(CONTRACT, contract)
    print(ab.canonical_json({"contract_sha256": contract["contract_sha256"]}))


def verify_contract(contract):
    payload = {k: v for k, v in contract.items() if k != "contract_sha256"}
    if ab.sha256_json(payload) != contract["contract_sha256"]:
        raise ProductionIntegrationError("contract digest mismatch")
    for name, expected in contract["input_file_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ProductionIntegrationError("input drift: " + name)


def run():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    verify_contract(contract)
    from scripts import phase5a0_baseline as baseline
    from scripts.retrieval_golden_v2 import load_oracle, validate_oracle

    oracle = load_oracle()
    validate_oracle(oracle)
    _, _, evidence = baseline.load_sources_and_evidence(oracle)
    control = c.read(c.CONTROL)
    return evaluate_once(oracle, evidence, control)


def evaluate():
    results = []
    for index in (1, 2):
        directory = OUTPUT / f"run-{index}"
        if (directory / "results.json").exists():
            raise ProductionIntegrationError("refusing to overwrite results")
        result = run()
        fingerprint = ab.sha256_json(result)
        ab.write_json(directory / "results.json", result, canonical=True)
        ab.write_json(directory / "runtime.json", {"result_sha256": fingerprint})
        results.append((result, fingerprint))
    if results[0][1] != results[1][1]:
        raise ProductionIntegrationError("deterministic rerun mismatch")
    result, fingerprint = results[0]
    summary = {
        "result_sha256": fingerprint,
        "aggregate_gating_33": result["aggregate_gating_33"],
        "focus": result["focus"],
        "quality_gate": result["quality_gate"],
        "seed_identity_mismatches": result["seed_identity_mismatches"],
    }
    ab.write_json(OUTPUT / "evaluation_summary.json", summary)
    ab.write_json(OUTPUT / "results.json", result, canonical=True)
    print(ab.canonical_json(summary))


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
