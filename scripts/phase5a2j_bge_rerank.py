"""Offline BGE second-reranker evaluation over frozen CSWP union pools."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import phase5a2_shadow_ab as ab  # noqa: E402
from scripts import phase5a2h_rerank as h  # noqa: E402

STARTING_HEAD = "d684cb3ade29ff29905cfbc00d56233bc85dd16a"
DEPENDENCY = ROOT / "evals/fixtures/phase5a2j_bge_reranker_dependency.json"
CONTRACT = ROOT / "evals/fixtures/phase5a2j_bge_reranker_contract.json"
TOKENIZER_FIXTURE = ROOT / "evals/fixtures/phase5a2j_bge_reranker_tokenizer"
OUTPUT = ROOT / "reports/phase5/phase5a2j"
H2H_CONTRACT = ROOT / "evals/fixtures/phase5a2h_reranker_contract.json"
FROZEN_POOLS = ROOT / "reports/phase5/phase5a2h/candidate_pools.json"
FROZEN_POOL_SHA256 = "df00e312a0891fc65befa83b05f59e032809e78f6447f351aea16ca34e6db52c"
MSMARCO_RESULTS = ROOT / "reports/phase5/phase5a2h/run-1/results.json"
MODEL_ID = "BAAI/bge-reranker-base"
REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)
FOCUS = ("Q09", "Q13", "Q16", "Q17", "Q20", "Q21", "Q22", "Q19", "Q30")
BATCH_SIZE = 8
MAX_INPUT = 512


class BgeRerankError(RuntimeError):
    """A frozen identity, candidate, token, score or provenance gate failed."""


def read(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return ab.sha256_bytes(path.read_bytes())


def snapshot_path() -> Path:
    return (
        Path.home()
        / ".cache/huggingface/hub/models--BAAI--bge-reranker-base/snapshots"
        / REVISION
    )


def library_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "tokenizers",
            "sentence-transformers",
            "safetensors",
            "numpy",
            "huggingface_hub",
        )
    }


def load_dependency() -> dict:
    payload = read(DEPENDENCY)
    if payload.get("canonical_model_id") != MODEL_ID or payload.get("revision") != REVISION:
        raise BgeRerankError("dependency manifest model identity mismatch")
    return payload


def verify_dependency_hashes(dependency: dict) -> None:
    snapshot = snapshot_path()
    if snapshot.name != REVISION:
        raise BgeRerankError("ACQUISITION_DEPENDENCY: pinned BGE snapshot missing")
    for name in dependency["required_artifacts"]:
        path = snapshot / name
        if not path.is_file():
            raise BgeRerankError(f"missing required artifact: {name}")
        observed = file_hash(path)
        expected = dependency["file_sha256"][name]
        if observed != expected:
            raise BgeRerankError(
                f"artifact hash drift for {name}: observed={observed} expected={expected}"
            )


def model_identity(snapshot: Path, dependency: dict) -> dict:
    from safetensors import safe_open

    verify_dependency_hashes(dependency)
    config = read(snapshot / "config.json")
    if (
        config.get("architectures") != ["XLMRobertaForSequenceClassification"]
        or config.get("model_type") != "xlm-roberta"
        or config.get("num_hidden_layers") != 12
        or config.get("max_position_embeddings") != 514
        or len(config.get("id2label", {})) != 1
    ):
        raise BgeRerankError("BGE architecture or capacity drift")
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as weights:
        layers = sorted(
            {
                int(key.split(".")[3])
                for key in weights.keys()
                if key.startswith("roberta.encoder.layer.")
            }
        )
        out_proj = list(weights.get_slice("classifier.out_proj.weight").get_shape())
        dense = list(weights.get_slice("classifier.dense.weight").get_shape())
    if layers != list(range(12)) or out_proj != [1, 768] or dense != [768, 768]:
        raise BgeRerankError("BGE weight architecture mismatch")
    identity = dependency["model_identity"]
    if (
        identity["parameter_count"] != 278_044_417
        or identity["num_hidden_layers"] != 12
        or identity["max_position_embeddings"] != 514
    ):
        raise BgeRerankError("dependency model_identity drift")
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "file_sha256": {name: file_hash(snapshot / name) for name in FILES},
        "weights_bytes": (snapshot / "model.safetensors").stat().st_size,
        "architecture": "XLMRobertaForSequenceClassification",
        "encoder_layers": layers,
        "classifier_out_proj_shape": out_proj,
        "classifier_dense_shape": dense,
        "max_input_tokens": MAX_INPUT,
        "score": "raw single float32 relevance logit; higher is better; no score fusion",
        "parameter_count": identity["parameter_count"],
    }


def load_tokenizer(path: Path, model: dict) -> object:
    from transformers import AutoTokenizer

    path = Path(path)
    for name in FILES:
        if name == "model.safetensors":
            continue
        if not (path / name).is_file() or file_hash(path / name) != model["file_sha256"][name]:
            raise BgeRerankError("tokenizer identity mismatch: " + name)
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if (
        not tokenizer.is_fast
        or tokenizer.model_max_length != MAX_INPUT
        or tokenizer.num_special_tokens_to_add(pair=True) != 4
        or tokenizer.vocab_size != 250_002
    ):
        raise BgeRerankError("unexpected BGE pair tokenizer")
    return tokenizer


def pair_encoding(tokenizer, query: str, candidate: str, max_input: int = MAX_INPUT):
    q = tokenizer(query, add_special_tokens=False, truncation=False)["input_ids"]
    c = tokenizer(candidate, add_special_tokens=False, truncation=False)["input_ids"]
    encoded = tokenizer(
        query, candidate, add_special_tokens=True, truncation=False, padding=False
    )
    ids = encoded["input_ids"]
    specials = tokenizer.num_special_tokens_to_add(pair=True)
    if len(ids) != len(q) + len(c) + specials:
        raise BgeRerankError("pair tokenizer changed or silently truncated content")
    if len(ids) > max_input:
        raise BgeRerankError(
            f"RERANKER_PAIR_OVERFLOW: {len(ids)} > {max_input}; STOP without scoring"
        )
    counts = {
        "query_tokens": len(q),
        "candidate_tokens": len(c),
        "special_tokens": specials,
        "total_pair_tokens": len(ids),
        "truncated_tokens": 0,
        "truncation": False,
        "pair_input_sha256": ab.sha256_json(dict(encoded)),
    }
    return encoded, counts


def attach_bge_pair_tokens(pools: list[dict], units: dict, tokenizer) -> list[dict]:
    rows = []
    for pool in pools:
        candidates = []
        for candidate in pool["candidates"]:
            _, counts = pair_encoding(
                tokenizer,
                pool["query"],
                units[candidate["chunk_id"]]["retrieval_text"],
            )
            candidates.append({**candidate, "pair_tokens": counts})
        rows.append({**pool, "candidates": candidates})
    return rows


def load_frozen_pools() -> list[dict]:
    pools = read(FROZEN_POOLS)
    if ab.sha256_json(pools) != FROZEN_POOL_SHA256:
        raise BgeRerankError("frozen candidate-pool fingerprint mismatch")
    h2h = read(H2H_CONTRACT)
    if FROZEN_POOL_SHA256 != h2h["candidate_pool_sha256"]:
        raise BgeRerankError("Phase-5A2H candidate_pool_sha256 drift")
    return pools


def configure_torch() -> None:
    import random

    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def load_model(snapshot: Path):
    from transformers import AutoModelForSequenceClassification
    import torch

    model, info = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise BgeRerankError("incomplete or incompatible BGE weights")
    model.to("cpu").eval()
    if model.config.num_labels != 1 or model.config.num_hidden_layers != 12:
        raise BgeRerankError("loaded BGE architecture mismatch")
    return model


def sort_scores(pool: list[dict], scores: list[float]) -> list[dict]:
    if (
        len(scores) != len(pool)
        or len({r["chunk_id"] for r in pool}) != len(pool)
        or not all(math.isfinite(s) for s in scores)
    ):
        raise BgeRerankError("score roster mismatch or nonfinite relevance score")
    rows = [
        {
            "chunk_id": row["chunk_id"],
            "source": row["source"],
            "reranker_score": float(score),
        }
        for row, score in zip(pool, scores, strict=True)
    ]
    rows.sort(key=lambda row: (-row["reranker_score"], row["chunk_id"]))
    return [{**row, "rank": rank} for rank, row in enumerate(rows, 1)]


def score_pool(model, tokenizer, query: str, pool: list[dict], units: dict) -> list[dict]:
    import torch

    scores: list[float] = []
    for start in range(0, len(pool), BATCH_SIZE):
        batch = pool[start : start + BATCH_SIZE]
        encodings = []
        for candidate in batch:
            encoded, counts = pair_encoding(
                tokenizer, query, units[candidate["chunk_id"]]["retrieval_text"]
            )
            if counts != candidate["pair_tokens"]:
                raise BgeRerankError("scored pair differs from preflight")
            encodings.append(dict(encoded))
        padded = tokenizer.pad(encodings, padding=True, return_tensors="pt")
        if padded["input_ids"].shape[1] > MAX_INPUT or padded["attention_mask"].sum(
            dim=1
        ).tolist() != [c["pair_tokens"]["total_pair_tokens"] for c in batch]:
            raise BgeRerankError("padded pair length/truncation mismatch")
        with torch.inference_mode():
            logits = model(**padded).logits
        if tuple(logits.shape) != (len(batch), 1):
            raise BgeRerankError("unexpected score shape")
        scores.extend(logits[:, 0].tolist())
    return sort_scores(pool, scores)


def evaluate_rows(
    oracle,
    evidence,
    chunks,
    units,
    control,
    msmarco,
    pools,
    ranked,
):
    from scripts import phase5a0_baseline as baseline

    cswp_queries = {q["query_id"]: q for q in control["arms"]["CSWP"]["per_query"]}
    a_queries = {q["query_id"]: q for q in control["arms"]["A"]["per_query"]}
    msmarco_by_id = {row["query_id"]: row for row in msmarco["per_query"]}
    a_manifest = read(ROOT / "reports/phase5/phase5a0/baseline_a_manifest.json")
    source_paths = {entry["source"]: entry["path"] for entry in oracle["corpus_manifest"]}
    a_chunks = {
        c["chunk_id"]: ab.EvalChunk(
            c["ordinal"],
            c["chunk_id"],
            c["source"],
            source_paths[c["source"]],
            "",
            ((c["source_char_start"], c["source_char_end"]),),
            0,
        )
        for c in a_manifest["chunking"]["ordered_chunks"]
    }
    spans_by_query = defaultdict(list)
    for span in evidence:
        spans_by_query[span.query_id].append(span)

    rows = []
    for query, pool in zip(oracle["queries"], pools, strict=True):
        qid = query["query_id"]
        ordered = ranked[qid]
        if {r["chunk_id"] for r in ordered} != {r["chunk_id"] for r in pool["candidates"]}:
            raise BgeRerankError("reranked result is not the complete frozen pool")
        spans = spans_by_query[qid]
        control_rows = cswp_queries[qid]["channels"]["hybrid"]["top10"]
        a_rows = a_queries[qid]["channels"]["hybrid"]["top10"]
        msmarco_row = msmarco_by_id[qid]
        msmarco_top10 = msmarco_row["channels"]["reranked"]["top10"]
        cm = ab._rank_metrics(query, spans, control_rows, chunks)
        am = ab._rank_metrics(query, spans, a_rows, a_chunks)
        mm = msmarco_row["channels"]["reranked"]["metrics"]
        bm = ab._rank_metrics(query, spans, ordered, chunks)
        if (
            cm != cswp_queries[qid]["channels"]["hybrid"]["metrics"]
            or am != a_queries[qid]["channels"]["hybrid"]["metrics"]
            or mm != msmarco_row["channels"]["reranked"]["metrics"]
        ):
            raise BgeRerankError("frozen control/MS-MARCO metric reproduction failed")
        coverage = {
            name: h.span_coverage(spans, result, mapping, 5)
            for name, result, mapping in [
                ("control", control_rows, chunks),
                ("msmarco", msmarco_top10, chunks),
                ("bge", ordered, chunks),
                ("A", a_rows, a_chunks),
            ]
        }
        pool_coverage = h.span_coverage(spans, pool["candidates"], chunks, pool["pool_size"])
        evidence_delta = h.classify_evidence(coverage["control"], coverage["bge"])
        ranking_delta = h.classify_ranking(cm, bm)
        critical = {h.span_key(s) for s in spans if s.grade == 2}
        rank_by_id = {r["chunk_id"]: r["rank"] for r in ordered}
        control_rank = {r["chunk_id"]: r["rank"] for r in control_rows}
        h1 = []
        for candidate in pool["candidates"]:
            if candidate["h1_only"]:
                h1.append(
                    {
                        "chunk_id": candidate["chunk_id"],
                        "source": candidate["source"],
                        "dense_rank": candidate["dense_rank"],
                        "bm25_rank": candidate["bm25_rank"],
                        "control_hybrid_rank_at_10": control_rank.get(candidate["chunk_id"]),
                        "bge_rank": rank_by_id[candidate["chunk_id"]],
                        "demoted_from_control_top5": candidate["chunk_id"]
                        in {r["chunk_id"] for r in control_rows[:5]}
                        and rank_by_id[candidate["chunk_id"]] > 5,
                    }
                )
        rows.append(
            {
                "query_id": qid,
                "query": query["query"],
                "gating_eligible": query["gating_eligible"],
                "channels": {
                    "control": {"metrics": cm, "top10": control_rows},
                    "msmarco": {"metrics": mm, "top10": msmarco_top10},
                    "bge": {"metrics": bm, "top10": ordered[:10], "top5": ordered[:5]},
                    "A": {"metrics": am, "top10": a_rows},
                },
                "evidence_comparison_vs_control": evidence_delta,
                "evidence_comparison_vs_msmarco": h.classify_evidence(
                    coverage["msmarco"], coverage["bge"]
                ),
                "ranking_comparison_vs_control": ranking_delta,
                "ranking_comparison_vs_msmarco": h.classify_ranking(mm, bm),
                "ranking_only_regression": evidence_delta["classification"] == "EVIDENCE SAME"
                and ranking_delta["classification"] == "RANKING-METRIC REGRESSED",
                "evidence_vs_A": h.classify_evidence(coverage["A"], coverage["bge"]),
                "critical_spans_lost_vs_control": sorted(
                    (coverage["control"] - coverage["bge"]) & critical
                ),
                "critical_spans_lost_vs_msmarco": sorted(
                    (coverage["msmarco"] - coverage["bge"]) & critical
                ),
                "critical_spans_lost_vs_A": sorted((coverage["A"] - coverage["bge"]) & critical),
                "span_coverage_at_5": {name: sorted(value) for name, value in coverage.items()},
                "candidate_pool": {
                    "size": pool["pool_size"],
                    "evidence_recall": len(pool_coverage) / len(spans) if spans else 0.0,
                    "source_recall": baseline.source_recall_at_k(
                        pool["candidates"],
                        {r["source"]: r["grade"] for r in query["relevant_sources"]},
                        pool["pool_size"],
                    ),
                    "covered_spans": sorted(pool_coverage),
                    "missing_spans": sorted({h.span_key(s) for s in spans} - pool_coverage),
                },
                "ranked_pool": ordered,
                "h1_observations": h1,
                "occupancy": {
                    "control": h.occupancy(control_rows),
                    "msmarco": h.occupancy(msmarco_top10),
                    "bge": h.occupancy(ordered),
                    "A": h.occupancy(a_rows),
                },
            }
        )

    aggregate = {
        channel: {
            scope: ab._aggregate(rows, channel, gating_only=gating)
            for scope, gating in [
                ("gating_33_excludes_q25_q26", True),
                ("all_35_diagnostic", False),
            ]
        }
        for channel in ("control", "msmarco", "bge", "A")
    }
    for channel, name in [("control", "CSWP"), ("A", "A")]:
        if aggregate[channel] != control["arms"][name]["aggregate_metrics"]["hybrid"]:
            raise BgeRerankError("frozen control aggregate mismatch")
    if aggregate["msmarco"] != msmarco["aggregate_metrics"]["reranked"]:
        raise BgeRerankError("frozen MS-MARCO aggregate mismatch")

    gating = [r for r in rows if r["gating_eligible"]]
    lists = {
        label: [r["query_id"] for r in gating if r[field]["classification"] == label]
        for field, prefix in [
            ("evidence_comparison_vs_control", "EVIDENCE"),
            ("ranking_comparison_vs_control", "RANKING-METRIC"),
        ]
        for label in [prefix + " IMPROVED", prefix + " SAME", prefix + " REGRESSED"]
    }
    lists["ranking_only_regressions"] = [
        r["query_id"] for r in gating if r["ranking_only_regression"]
    ]
    pool_summary = {
        scope: {
            "query_count": len(selected),
            "mean_candidates": statistics.fmean(r["candidate_pool"]["size"] for r in selected),
            "min_candidates": min(r["candidate_pool"]["size"] for r in selected),
            "max_candidates": max(r["candidate_pool"]["size"] for r in selected),
            "evidence_recall": statistics.fmean(
                r["candidate_pool"]["evidence_recall"] for r in selected
            ),
            "source_recall": statistics.fmean(
                r["candidate_pool"]["source_recall"] for r in selected
            ),
        }
        for scope, selected in [("gating_33", gating), ("all_35", rows)]
    }
    diversity = {
        channel: {
            str(k): {
                metric: statistics.fmean(r["occupancy"][channel][str(k)][metric] for r in gating)
                for metric in gating[0]["occupancy"][channel][str(k)]
            }
            for k in ab.K_VALUES
        }
        for channel in ("control", "msmarco", "bge", "A")
    }
    recalls = {
        name: aggregate[name]["gating_33_excludes_q25_q26"]["evidence_span_recall@5"]
        for name in aggregate
    }
    critical_losses = [
        r["query_id"]
        for r in gating
        if r["critical_spans_lost_vs_control"] or r["critical_spans_lost_vs_A"]
    ]
    new_vs_control = [
        r["query_id"]
        for r in gating
        if r["critical_spans_lost_vs_control"]
        and not msmarco_by_id[r["query_id"]]["critical_spans_lost_vs_control"]
    ]
    return {
        "aggregate_metrics": aggregate,
        "candidate_pool": pool_summary,
        "gating_classification": lists,
        "per_query": rows,
        "focus_query_ids": list(FOCUS),
        "occupancy_gating": diversity,
        "comparison_arms": {
            "control": "CSWP hybrid/RRF",
            "msmarco": "cross-encoder/ms-marco-MiniLM-L-6-v2 reranker (Phase 5A2H)",
            "bge": "BAAI/bge-reranker-base reranker (Phase 5A2J)",
            "A": "historical legacy baseline A",
        },
        "quality_gate": {
            "material_evidence_improvement": recalls["bge"] > recalls["control"]
            and recalls["bge"] >= recalls["A"],
            "beats_msmarco_evidence_recall_at_5": recalls["bge"] > recalls["msmarco"],
            "critical_evidence_regressions": critical_losses,
            "new_critical_regressions_vs_control": new_vs_control,
            "quality_pass": recalls["bge"] > recalls["control"]
            and recalls["bge"] >= recalls["A"]
            and not critical_losses,
            "reranker_ready_to_advance": recalls["bge"] > recalls["control"]
            and recalls["bge"] >= recalls["A"]
            and not critical_losses,
        },
        "absence_metrics": None,
        "absence_reason": "golden-v2 has zero ABSENT queries; all 35 are development/diagnostic",
        "provider_calls": 0,
        "external_network_calls": 0,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
        "embeddings_generated": 0,
    }


def rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def verify_contract(contract: dict, check_model: bool = True) -> None:
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if ab.sha256_json(payload) != contract["contract_sha256"]:
        raise BgeRerankError("frozen contract digest mismatch")
    for name, expected in contract["input_file_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise BgeRerankError("frozen input drift: " + name)
    if check_model:
        dependency = load_dependency()
        if (
            model_identity(snapshot_path(), dependency) != contract["model"]
            or library_versions() != contract["library_versions"]
        ):
            raise BgeRerankError("BGE/runtime identity drift")


def freeze() -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    if (
        git("rev-parse", "HEAD") != STARTING_HEAD
        or git("rev-parse", "refs/remotes/origin/feature/phase5-rag-hardening")
        != STARTING_HEAD
        or git("branch", "--show-current") != "feature/phase5-rag-hardening"
    ):
        raise BgeRerankError("starting checkout/ref mismatch")
    if CONTRACT.exists():
        raise BgeRerankError("contract already frozen; refusing overwrite")

    dependency = load_dependency()
    model = model_identity(snapshot_path(), dependency)
    tokenizer = load_tokenizer(snapshot_path(), model)
    oracle, _, units, control, _ = h.input_data()
    pools = attach_bge_pair_tokens(load_frozen_pools(), units, tokenizer)
    msmarco = read(MSMARCO_RESULTS)
    paths = [
        "scripts/phase5a0_baseline.py",
        "scripts/phase5a1_shadow.py",
        "scripts/phase5a2_shadow_ab.py",
        "scripts/phase5a2b_packed.py",
        "scripts/phase5a2e_cswp.py",
        "scripts/phase5a2h_rerank.py",
        "evals/fixtures/retrieval_golden_v2.json",
        "evals/fixtures/phase5a2e_cswp_contract.json",
        "evals/fixtures/phase5a2h_reranker_contract.json",
        "evals/fixtures/phase5a2j_bge_reranker_dependency.json",
        "reports/phase5/phase5a0/baseline_a_manifest.json",
        "reports/phase5/phase5a2e/experiment_manifest.json",
        "reports/phase5/phase5a2e/candidate_cswp_manifest.json",
        "reports/phase5/phase5a2e/results.json",
        "reports/phase5/phase5a2h/candidate_pools.json",
        "reports/phase5/phase5a2h/run-1/results.json",
    ] + [row["path"] for row in oracle["corpus_manifest"]]
    contract = {
        "schema_version": "phase5a2j_bge_reranker_contract_v1",
        "required_parent": STARTING_HEAD,
        "remote_check": "local origin/feature/phase5-rag-hardening ref matches required HEAD; no network fetch",
        "model": model,
        "dependency_manifest_sha256": dependency.get(
            "dependency_manifest_sha256", ab.sha256_json(dependency)
        ),
        "library_versions": library_versions(),
        "input_file_sha256": {name: file_hash(ROOT / name) for name in paths},
        "cswp_fingerprint": ab.sha256_json(read(h.CSWP)),
        "candidate_pool_sha256": FROZEN_POOL_SHA256,
        "msmarco_results_sha256": ab.sha256_json(msmarco),
        "candidate_generation": h2h_candidate_generation(),
        "determinism": {
            "device": "cpu",
            "dtype": "float32",
            "seed": 0,
            "torch_threads": 1,
            "torch_interop_threads": 1,
            "deterministic_algorithms": True,
            "mkldnn_enabled": False,
            "attention_implementation": "eager",
            "eval_mode": True,
            "inference_mode": True,
            "batch_size": BATCH_SIZE,
            "candidate_order": "chunk_id ascending",
            "final_order": "raw relevance logit descending then chunk_id ascending",
        },
        "tokenization": {
            "input": "query and exact CSWP retrieval text as separate pair arguments, without stripping",
            "max_input_tokens": MAX_INPUT,
            "special_tokens": 4,
            "truncation": False,
            "padding": "longest in each fixed batch after length preflight",
            "overflow": "STOP before any experiment scoring; never invent a truncation policy",
        },
        "metric_implementation": "unchanged phase5a2_shadow_ab._rank_metrics and _span_covered",
        "classification": read(H2H_CONTRACT)["classification"],
        "advance_gate": read(H2H_CONTRACT)["advance_gate"],
        "diagnostic_non_gating": ["Q25", "Q26"],
        "production_retrieval_changed": False,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
        "provider_calls": 0,
        "external_network_calls": 0,
        "mmr": False,
        "candidate_c": False,
    }
    contract["contract_sha256"] = ab.sha256_json(contract)
    ab.write_json(CONTRACT, contract)
    ab.write_json(OUTPUT / "candidate_pools_bge_tokens.json", pools)
    print(
        json.dumps(
            {
                "contract_sha256": contract["contract_sha256"],
                "pairs": sum(q["pool_size"] for q in pools),
                "max_pair_tokens": max(
                    c["pair_tokens"]["total_pair_tokens"]
                    for q in pools
                    for c in q["candidates"]
                ),
            },
            indent=2,
        )
    )


def h2h_candidate_generation() -> str:
    return read(H2H_CONTRACT)["candidate_generation"]


def run(output: Path) -> None:
    contract = read(CONTRACT)
    verify_contract(contract)
    dependency = load_dependency()
    oracle, evidence, units, control, chunks = h.input_data()
    msmarco = read(MSMARCO_RESULTS)
    if ab.sha256_json(msmarco) != contract["msmarco_results_sha256"]:
        raise BgeRerankError("frozen MS-MARCO results drift")
    tokenizer = load_tokenizer(snapshot_path(), contract["model"])
    pools = attach_bge_pair_tokens(load_frozen_pools(), units, tokenizer)
    configure_torch()
    before = rss_bytes()
    start = time.perf_counter_ns()
    model = load_model(snapshot_path())
    load_ms = (time.perf_counter_ns() - start) / 1e6
    parameters = sum(p.numel() for p in model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    ranked, timings = {}, []
    for pool in pools:
        start = time.perf_counter_ns()
        ranked[pool["query_id"]] = score_pool(
            model, tokenizer, pool["query"], pool["candidates"], units
        )
        timings.append(
            {
                "query_id": pool["query_id"],
                "candidates": pool["pool_size"],
                "elapsed_ms": (time.perf_counter_ns() - start) / 1e6,
            }
        )
    result = evaluate_rows(oracle, evidence, chunks, units, control, msmarco, pools, ranked)
    result["contract_sha256"] = contract["contract_sha256"]
    result["dependency_manifest_sha256"] = dependency.get(
        "dependency_manifest_sha256", ab.sha256_json(dependency)
    )
    result["token_safety"] = {
        "candidate_pairs": sum(q["pool_size"] for q in pools),
        "max_pair_tokens": max(
            c["pair_tokens"]["total_pair_tokens"] for q in pools for c in q["candidates"]
        ),
        "pairs_truncated": 0,
        "pairs_exceeding_max_input": 0,
    }
    result["deterministic_score_sha256"] = ab.sha256_json(ranked)
    result["scoring_runner_sha256"] = file_hash(Path(__file__))
    runtime = {
        "result_sha256": ab.sha256_json(result),
        "model_load_ms": load_ms,
        "latency": ab._latency_summary([t["elapsed_ms"] for t in timings]),
        "per_query": timings,
        "latency_scope": "one pass over all 35 queries; includes pair tokenization, batching, inference and sorting, excludes model loading; first query cold; no warmup or retries",
        "resources": {
            "parameters": parameters,
            "float32_parameter_bytes": parameter_bytes,
            "weights_file_bytes": contract["model"]["weights_bytes"],
            "process_peak_rss_before_model_bytes": before,
            "process_peak_rss_after_evaluation_bytes": rss_bytes(),
            "rss_scope": "process lifetime peak, includes Python/torch/transformers/evaluator and model, not isolated model allocation",
            "device": "cpu",
            "threads": 1,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "provider_calls": 0,
        "external_network_calls": 0,
    }
    ab.write_json(output / "results.json", result, canonical=True)
    ab.write_json(output / "runtime.json", runtime)
    print(
        json.dumps(
            {
                "result_sha256": runtime["result_sha256"],
                "token_safety": result["token_safety"],
                "quality_gate": result["quality_gate"],
                "latency": runtime["latency"],
            },
            indent=2,
        )
    )


def main() -> None:
    ab.install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        if (
            args.output is None
            or not args.output.resolve().is_relative_to(OUTPUT.resolve())
            or args.output.resolve() == OUTPUT.resolve()
        ):
            raise BgeRerankError("run output must be an isolated phase5a2j subdirectory")
        if (args.output / "results.json").exists():
            raise BgeRerankError("refusing to overwrite an experiment result")
        run(args.output)


if __name__ == "__main__":
    main()
