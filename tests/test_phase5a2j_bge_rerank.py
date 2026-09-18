"""Offline BGE reranker experiment gates; real inference qualified by two saved runs."""

from copy import deepcopy
import subprocess
import sys

import pytest

from scripts import phase5a2h_rerank as h
from scripts import phase5a2j_bge_rerank as bge
from scripts import phase5a2_shadow_ab as ab


@pytest.fixture
def contract():
    return bge.read(bge.CONTRACT)


@pytest.fixture
def dependency():
    return bge.load_dependency()


@pytest.fixture
def tokenizer(contract):
    return bge.load_tokenizer(bge.snapshot_path(), contract["model"])


def test_pinned_bge_identity(contract, dependency):
    model = contract["model"]
    assert model["model_id"] == "BAAI/bge-reranker-base"
    assert model["revision"] == "2cfc18c9415c912f9d8155881c133215df768a70"
    assert model["parameter_count"] == 278_044_417
    assert model["encoder_layers"] == list(range(12))
    assert model["classifier_out_proj_shape"] == [1, 768]
    assert dependency["file_sha256"]["model.safetensors"] == model["file_sha256"]["model.safetensors"]
    bge.verify_contract(contract, check_model=False)


def test_frozen_candidate_pool_matches_phase5a2h(contract):
    pools = bge.load_frozen_pools()
    assert ab.sha256_json(pools) == contract["candidate_pool_sha256"]
    assert contract["candidate_pool_sha256"] == bge.FROZEN_POOL_SHA256


def test_bge_pair_preflight_within_limit(tokenizer):
    pools = bge.read(bge.FROZEN_POOLS)
    _, _, units, _, _ = h.input_data()
    refreshed = bge.attach_bge_pair_tokens(pools, units, tokenizer)
    assert (
        max(
            c["pair_tokens"]["total_pair_tokens"]
            for row in refreshed
            for c in row["candidates"]
        )
        == 353
    )
    assert all(
        not c["pair_tokens"]["truncation"]
        for row in refreshed
        for c in row["candidates"]
    )


def test_pair_overflow_fails_closed(tokenizer):
    with pytest.raises(bge.BgeRerankError, match="PAIR_OVERFLOW"):
        bge.pair_encoding(tokenizer, "hello", "world " * 520)


def test_two_real_runs_are_identical_and_quality_gate_recorded():
    first = bge.OUTPUT / "run-1/results.json"
    second = bge.OUTPUT / "run-2/results.json"
    assert first.read_bytes() == second.read_bytes()
    result = bge.read(first)
    for run in ("run-1", "run-2"):
        runtime = bge.read(bge.OUTPUT / run / "runtime.json")
        assert runtime["result_sha256"] == ab.sha256_json(result)
    assert result["token_safety"] == {
        "candidate_pairs": 1046,
        "max_pair_tokens": 353,
        "pairs_truncated": 0,
        "pairs_exceeding_max_input": 0,
    }
    assert result["quality_gate"]["beats_msmarco_evidence_recall_at_5"]
    assert result["quality_gate"]["critical_evidence_regressions"] == ["Q13", "Q22"]
    assert result["quality_gate"]["new_critical_regressions_vs_control"] == []
    assert result["quality_gate"]["reranker_ready_to_advance"] is False
    by_id = {row["query_id"]: row for row in result["per_query"]}
    assert by_id["Q20"]["evidence_comparison_vs_msmarco"]["classification"] == "EVIDENCE IMPROVED"
    assert by_id["Q09"]["channels"]["bge"]["metrics"]["evidence_span_recall@5"] == 1.0
    assert by_id["Q13"]["channels"]["bge"]["metrics"]["evidence_span_recall@5"] == 0.0
    assert by_id["Q19"]["candidate_pool"]["missing_spans"]
    assert by_id["Q30"]["candidate_pool"]["missing_spans"]


def test_offline_entry_guard_denies_socket_in_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'from scripts.phase5a2_shadow_ab import install_offline_guard; install_offline_guard(); import socket; socket.create_connection(("example.com",443))',
        ],
        cwd=bge.ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "OFFLINE_NETWORK_FORBIDDEN" in result.stderr


def test_contract_drift_rejected(contract, monkeypatch):
    bad = deepcopy(contract)
    bad["model"]["revision"] = "bad"
    with pytest.raises(bge.BgeRerankError, match="digest mismatch"):
        bge.verify_contract(bad, check_model=False)
