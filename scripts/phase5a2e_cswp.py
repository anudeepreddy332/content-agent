"""Phase 5A2E: frozen CSWP-v1 lossless contiguous packing. A/B/B-PACKED unchanged."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import phase5a1_shadow as shadow  # noqa: E402
from scripts import phase5a2_shadow_ab as ab  # noqa: E402
from scripts.phase5a2b_packed import (  # noqa: E402
    diversity,
    eval_chunks,
    historical_control_check,
    pack as pack_b_packed,
)

CONTRACT = ROOT / "evals/fixtures/phase5a2e_cswp_contract.json"
OUTPUT = ROOT / "reports/phase5/phase5a2e"
STARTING_HEAD = "fa7eefbc0eeb6eb8570577a8e19497bfbd2a21b8"
CANDIDATE = "CSWP"
from agent.cswp.errors import CSWPError  # noqa: E402
from agent.cswp.packer import pack  # noqa: E402

def freeze(output):
    contract = json.loads(CONTRACT.read_text())
    old = json.loads(
        (ROOT / "reports/phase5/phase5a2/experiment_manifest.json").read_text()
    )
    files = [
        CONTRACT,
        Path(__file__),
        ROOT / "scripts/phase5a0_baseline.py",
        ROOT / "scripts/phase5a1_shadow.py",
        ROOT / "scripts/phase5a2_shadow_ab.py",
        ROOT / "scripts/phase5a2b_packed.py",
        ROOT / "agent/cswp/compiler.py",
        ROOT / "agent/cswp/constants.py",
        ROOT / "agent/cswp/identity.py",
        ROOT / "agent/cswp/packer.py",
        ROOT / "agent/cswp/source.py",
        ROOT / "agent/cswp/structural.py",
        ROOT / "agent/cswp/tokenizer.py",
        ROOT / "evals/fixtures/retrieval_golden_v2.json",
        ab.BASELINE_MANIFEST,
        ab.CANDIDATE_B_MANIFEST,
    ]
    manifest = {
        "required_parent": STARTING_HEAD,
        "candidate": CANDIDATE,
        "packing_contract": contract,
        "held_constant": old["held_constant"],
        "control_arms": old["arms"],
        "file_sha256": {
            str(path.relative_to(ROOT)): ab.sha256_bytes(path.read_bytes())
            for path in files
        },
        "provider_calls": 0,
        "external_network_calls": 0,
        "semantic_chunking": False,
        "production_retrieval_changed": False,
        "qdrant_mutated": False,
    }
    path = output / "experiment_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["manifest_sha256"] != ab.sha256_json(
            {key: value for key, value in existing.items() if key != "manifest_sha256"}
        ):
            raise CSWPError("preregistration hash mismatch")
        observed = {
            key: value
            for key, value in existing.items()
            if key not in ("frozen_at_utc", "manifest_sha256")
        }
        if observed != manifest:
            raise CSWPError("preregistered implementation changed; do not retune")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    manifest["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["manifest_sha256"] = ab.sha256_json(manifest)
    ab.write_json(path, manifest)
    return manifest


def comparisons(arms):
    rows = []
    for index, packed_query in enumerate(arms[CANDIDATE]["per_query"]):
        channels = {}
        for channel in ("dense", "bm25", "hybrid"):
            metrics = {
                name: arm["per_query"][index]["channels"][channel]["metrics"]
                for name, arm in arms.items()
            }
            channels[channel] = {
                "metrics": metrics,
                **{
                    "vs_" + name: {
                        "classification": ab._classification(
                            metrics[name], metrics[CANDIDATE]
                        ),
                        "deltas": {
                            key: ab._delta(metrics[CANDIDATE][key], metrics[name][key])
                            for key in metrics[name]
                        },
                    }
                    for name in ("A", "B", "B-PACKED")
                },
            }
        rows.append(
            {
                "query_id": packed_query["query_id"],
                "gating_eligible": packed_query["gating_eligible"],
                "channels": channels,
                "vs_A": channels["hybrid"]["vs_A"]["classification"],
                "vs_B": channels["hybrid"]["vs_B"]["classification"],
                "vs_B-PACKED": channels["hybrid"]["vs_B-PACKED"]["classification"],
            }
        )
    return rows


def evaluate(output, manifest):
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    from scripts import phase5a0_baseline as baseline

    oracle, legacy, chunks_b, evidence, _, context = ab.load_representations()
    contract = manifest["packing_contract"]
    tokenizer = shadow.MiniLMTokenizer()
    packed_contract = json.loads(
        (ROOT / "evals/fixtures/phase5a2b_packing_contract.json").read_text()
    )
    builds = []
    for run in (1, 2):
        rebuilt, _ = shadow.build_corpus(tokenizer)
        if rebuilt != context["candidate_b_manifest"]:
            raise CSWPError("B drift")
        packed, report = pack(rebuilt, tokenizer, contract)
        builds.append(packed)
        ab.write_json(output / f"run-{run}/build_report.json", report)
    if builds[0] != builds[1]:
        raise CSWPError("packing fingerprint mismatch")
    b_packed, _ = pack_b_packed(context["candidate_b_manifest"], tokenizer, packed_contract)
    ab.write_json(output / "candidate_cswp_manifest.json", builds[0])
    ab.write_json(output / "build_report.json", report)
    chunks = {
        "A": ab._a_eval_chunks(legacy),
        "B": chunks_b,
        "B-PACKED": eval_chunks(b_packed, oracle),
        CANDIDATE: eval_chunks(builds[0], oracle),
    }
    encoder = SentenceTransformer(str(context["snapshot"]), local_files_only=True)
    if (
        encoder.max_seq_length != baseline.MODEL_MAX_SEQUENCE_LENGTH
        or encoder.get_embedding_dimension() != 384
    ):
        raise CSWPError("MiniLM identity drift")
    results, runtimes = [], []
    previous = json.loads(
        (ROOT / "reports/phase5/phase5a2/aggregate_comparison.json").read_text()
    )
    previous_queries = json.loads(
        (ROOT / "reports/phase5/phase5a2/per_query_comparison.json").read_text()
    )["queries"]
    previous_packed = json.loads(
        (ROOT / "reports/phase5/phase5a2b/results.json").read_text()
    )
    for run in (1, 2):
        embeddings, embedding_ms = {}, {}
        for name, units in chunks.items():
            embeddings[name], embedding_ms[name] = ab._encode(
                encoder, [unit.retrieval_text for unit in units]
            )
        query_vectors, query_ms = ab._encode(
            encoder, [query["query"] for query in oracle["queries"]]
        )
        arms, runtime, storage = {}, {}, {}
        for name, units in chunks.items():
            arms[name], runtime[name] = ab.evaluate_arm(
                name, units, embeddings[name], query_vectors, oracle, evidence
            )
            bm25 = BM25Okapi([unit.retrieval_text.lower().split() for unit in units])
            storage[name] = {
                "vectors": len(units),
                "vector_multiplier_vs_A": len(units) / len(chunks["A"]),
                "float32_embedding_bytes": embeddings[name].nbytes,
                "retrieval_text_utf8_bytes": sum(
                    len(unit.retrieval_text.encode()) for unit in units
                ),
                "bm25_pickle_protocol_4_bytes": len(pickle.dumps(bm25, protocol=4)),
            }
        for name in ("A", "B"):
            if arms[name]["aggregate_metrics"] != previous["arms"][name]:
                raise CSWPError("A/B metric reproduction failure")
        if (
            arms["B-PACKED"]["aggregate_metrics"]
            != previous_packed["arms"]["B-PACKED"]["aggregate_metrics"]
        ):
            raise CSWPError("B-PACKED metric reproduction failure")
        control_checks = []
        for index, previous_query in enumerate(previous_queries):
            for name in ("A", "B"):
                if arms[name]["per_query"][index]["query_id"] != previous_query["query_id"]:
                    raise CSWPError("A/B query order drift")
                for channel in ("dense", "bm25", "hybrid"):
                    control_checks.append(
                        {
                            "arm": name,
                            "query_id": previous_query["query_id"],
                            "channel": channel,
                            **historical_control_check(
                                arms[name]["per_query"][index]["channels"][channel],
                                previous_query["channels"][channel][name],
                            ),
                        }
                    )
        rows = comparisons(arms)
        gating = {
            name: {
                label: [
                    row["query_id"]
                    for row in rows
                    if row["gating_eligible"] and row["vs_" + name] == label
                ]
                for label in ("IMPROVED", "SAME", "REGRESSED")
            }
            for name in ("A", "B", "B-PACKED")
        }
        recall = {
            name: arm["aggregate_metrics"]["hybrid"]["gating_33_excludes_q25_q26"][
                "evidence_span_recall@5"
            ]
            for name, arm in arms.items()
        }
        advance = (
            recall[CANDIDATE] >= recall["A"]
            and not gating["A"]["REGRESSED"]
            and report["children_above_254"] == 0
            and report["source_coverage_gaps"] == 0
            and report["provenance_failures"] == 0
        )
        result = {
            "experiment_manifest_sha256": manifest["manifest_sha256"],
            "arms": arms,
            "per_query_comparison": rows,
            "gating_queries": gating,
            "diversity": {name: diversity(arm) for name, arm in arms.items()},
            "cost": storage,
            "CSWP_ready_to_advance": advance,
            "absence_metrics": None,
            "provider_calls": 0,
            "external_network_calls": 0,
            "semantic_chunking": False,
            "production_retrieval_changed": False,
            "qdrant_mutated": False,
            "historical_control_checks": control_checks,
        }
        runtime_row = {
            "repetition": run,
            "embedding_ms": embedding_ms,
            "query_embedding_ms": query_ms,
            "retrieval": runtime,
            "deterministic_result_sha256": ab.sha256_json(result),
        }
        results.append(result)
        runtimes.append(runtime_row)
        ab.write_json(output / f"run-{run}/runtime_observation.json", runtime_row)
        ab.write_json(
            output / f"run-{run}/determinism_report.json",
            {
                "result_sha256": ab.sha256_json(result),
                "cswp_fingerprint": report["deterministic_fingerprint"],
                "arm_sha256": {name: ab.sha256_json(arm) for name, arm in arms.items()},
                "embedding_sha256": {
                    name: ab.sha256_bytes(value.tobytes())
                    for name, value in embeddings.items()
                },
                "query_embedding_sha256": ab.sha256_bytes(query_vectors.tobytes()),
            },
        )
    if results[0] != results[1]:
        raise CSWPError("retrieval rerun mismatch")
    for name, value in (
        ("results", results[0]),
        (
            "runtime_summary",
            {"repetitions": runtimes, "deterministic_rerun_equal": True},
        ),
        ("per_query_comparison", rows),
    ):
        ab.write_json(output / f"{name}.json", value)
    print(
        json.dumps(
            {
                "build": report,
                "gating": gating,
                "hybrid_evidence_recall@5": recall,
                "ready_to_advance": advance,
                "deterministic_rerun_equal": True,
            },
            indent=2,
        )
    )


def main():
    ab.install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    manifest = freeze(OUTPUT)
    if args.freeze_only:
        print(manifest["manifest_sha256"])
    else:
        evaluate(OUTPUT, manifest)


if __name__ == "__main__":
    main()
