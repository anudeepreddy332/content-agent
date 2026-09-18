"""Offline reranker experiment gates; real inference is qualified by two saved runs."""

from copy import deepcopy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import phase5a2h_rerank as rerank
from scripts import phase5a2_shadow_ab as ab


@pytest.fixture
def contract():
    return rerank.read(rerank.CONTRACT)


@pytest.fixture
def tokenizer(contract):
    return rerank.load_tokenizer(rerank.TOKENIZER_FIXTURE, contract["model"])


@pytest.fixture
def inputs():
    oracle, evidence, units, control, chunks = rerank.input_data()
    return oracle, evidence, units, control, chunks


def test_pinned_identity_is_explicit_and_complete(contract):
    model = contract["model"]
    assert model["model_id"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert model["revision"] == "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    assert (
        model["file_sha256"]["model.safetensors"]
        == "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    )
    assert model["encoder_layers"] == list(range(6))
    assert model["classifier_shape"] == [1, 384]
    assert model["max_input_tokens"] == 512
    assert "Identity" in model["score"]
    assert "L-12" in model["config_name_or_path"]
    rerank.verify_contract(contract, check_model=False)


def test_missing_model_fails_without_download(tmp_path):
    with pytest.raises(rerank.RerankError, match="ACQUISITION_DEPENDENCY"):
        rerank.model_identity(tmp_path / rerank.REVISION)


def test_changed_tokenizer_fails_closed(tokenizer, contract, tmp_path):
    for name in rerank.FILES:
        if name != "model.safetensors":
            (tmp_path / name).write_bytes(
                (rerank.TOKENIZER_FIXTURE / name).read_bytes()
            )
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises(rerank.RerankError, match="identity mismatch"):
        rerank.load_tokenizer(tmp_path, contract["model"])


def test_contract_or_input_drift_is_rejected(contract, monkeypatch):
    bad = deepcopy(contract)
    bad["model"]["revision"] = "bad"
    with pytest.raises(rerank.RerankError, match="digest mismatch"):
        rerank.verify_contract(bad, check_model=False)
    monkeypatch.setattr(rerank, "file_hash", lambda path: "changed")
    with pytest.raises(rerank.RerankError, match="frozen input drift"):
        rerank.verify_contract(contract, check_model=False)


@pytest.mark.parametrize("candidate_words,total", [(508, 512), (509, 513)])
def test_actual_pair_exact_limit_and_one_token_overflow(
    tokenizer, candidate_words, total
):
    query, candidate = "hello", " ".join(["world"] * candidate_words)
    if total == 513:
        with pytest.raises(rerank.RerankError, match="PAIR_OVERFLOW"):
            rerank.pair_encoding(tokenizer, query, candidate)
    else:
        encoded, counts = rerank.pair_encoding(tokenizer, query, candidate)
        assert counts["query_tokens"] == 1
        assert counts["candidate_tokens"] == 508
        assert counts["special_tokens"] == 3
        assert counts["total_pair_tokens"] == len(encoded["input_ids"]) == 512
        assert counts["truncated_tokens"] == 0


def test_tokenizer_silent_truncation_is_detected(tokenizer):
    class TruncatingTokenizer:
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        num_special_tokens_to_add = tokenizer.num_special_tokens_to_add

        def __call__(self, *args, **kwargs):
            result = tokenizer(*args, **kwargs)
            if len(args) == 2:
                result["input_ids"] = result["input_ids"][:-1]
            return result

    with pytest.raises(rerank.RerankError, match="silently truncated"):
        rerank.pair_encoding(TruncatingTokenizer(), "hello", "world")


def test_full_union_and_pair_ledger_reproduce_without_hybrid_cutoff(
    inputs, tokenizer, contract
):
    _, _, units, control, _ = inputs
    rows = rerank.preflight_pairs(
        control["arms"]["CSWP"]["per_query"], units, tokenizer
    )
    assert rows == rerank.read(rerank.OUTPUT / "candidate_pools.json")
    assert ab.sha256_json(rows) == contract["candidate_pool_sha256"]
    assert sum(row["pool_size"] for row in rows) == 1046
    assert (
        max(
            c["pair_tokens"]["total_pair_tokens"]
            for row in rows
            for c in row["candidates"]
        )
        == 293
    )
    for query, row in zip(control["arms"]["CSWP"]["per_query"], rows, strict=True):
        expected = {
            c["chunk_id"]
            for channel in ["dense", "bm25"]
            for c in query["channels"][channel]["candidate_top20"]
        }
        assert {c["chunk_id"] for c in row["candidates"]} == expected
        assert row["pool_size"] <= 40
        if row["query_id"] in {"Q09", "Q16"}:
            assert any(c["h1_only"] for c in row["candidates"])
        for candidate in row["candidates"]:
            assert (
                candidate["provenance"]["source_spans"]
                == units[candidate["chunk_id"]]["source_spans"]
            )
    q13 = next(row for row in rows if row["query_id"] == "Q13")
    hybrid = next(
        q for q in control["arms"]["CSWP"]["per_query"] if q["query_id"] == "Q13"
    )["channels"]["hybrid"]["top10"]
    assert {r["chunk_id"] for r in q13["candidates"]} > {r["chunk_id"] for r in hybrid}


@pytest.mark.parametrize("damage", ["duplicate", "unknown", "rank", "short_dense"])
def test_bad_candidate_roster_rejected(inputs, damage):
    _, _, units, control, _ = inputs
    query = deepcopy(control["arms"]["CSWP"]["per_query"][0])
    rows = query["channels"]["dense"]["candidate_top20"]
    if damage == "duplicate":
        rows[1]["chunk_id"] = rows[0]["chunk_id"]
    elif damage == "unknown":
        rows[0]["chunk_id"] = "unknown"
    elif damage == "rank":
        rows[0]["rank"] = 7
    else:
        rows.pop()
    with pytest.raises(rerank.RerankError):
        rerank.union_pool(query, units)


def test_score_order_ignores_native_scores_and_breaks_ties_by_id():
    pool = [
        {
            "chunk_id": cid,
            "source": cid,
            "dense_score": 999 if cid == "b" else -9,
            "bm25_score": 999 if cid == "b" else -9,
        }
        for cid in ["b", "c", "a"]
    ]
    ordered = rerank.sort_scores(pool, [0.1, 2.0, 2.0])
    assert [r["chunk_id"] for r in ordered] == ["a", "c", "b"]
    assert [r["rank"] for r in ordered] == [1, 2, 3]
    assert ordered == rerank.sort_scores(pool, [0.1, 2.0, 2.0])


@pytest.mark.parametrize("scores", [[0.1], [float("nan"), 1.0], [float("inf"), 1.0]])
def test_nonfinite_or_missing_scores_fail_closed(scores):
    with pytest.raises(rerank.RerankError):
        rerank.sort_scores(
            [{"chunk_id": "a", "source": "a"}, {"chunk_id": "b", "source": "b"}], scores
        )


def test_score_batches_preserve_pair_lengths_and_full_roster(tokenizer):
    import torch

    units = {str(i): {"retrieval_text": "world " * (i + 1)} for i in range(11)}
    pool = [
        {
            "chunk_id": str(i),
            "source": "s",
            "pair_tokens": rerank.pair_encoding(
                tokenizer, "hello", units[str(i)]["retrieval_text"]
            )[1],
        }
        for i in range(11)
    ]
    calls = []

    class Model:
        def __call__(self, **inputs):
            assert not torch.is_grad_enabled()
            lengths = inputs["attention_mask"].sum(dim=1)
            calls.append(lengths.tolist())
            return SimpleNamespace(logits=lengths[:, None].float())

    first = rerank.score_pool(Model(), tokenizer, "hello", pool, units)
    second = rerank.score_pool(Model(), tokenizer, "hello", pool, units)
    assert first == second
    assert [len(c) for c in calls] == [8, 3, 8, 3]
    assert len(first) == len(pool)
    bad = deepcopy(pool)
    bad[0]["pair_tokens"]["total_pair_tokens"] -= 1
    with pytest.raises(rerank.RerankError, match="differs from preflight"):
        rerank.score_pool(Model(), tokenizer, "hello", bad, units)


def test_evidence_and_ranking_regressions_are_distinct_and_swaps_not_hidden():
    before = {
        f"{m}@{k}": 1.0
        for m in ["source_recall", "precision", "ndcg"]
        for k in ab.K_VALUES
    }
    before["mrr@10"] = 1.0
    after = {**before, "ndcg@5": 0.9}
    assert (
        rerank.classify_evidence({"span"}, {"span"})["classification"]
        == "EVIDENCE SAME"
    )
    assert (
        rerank.classify_ranking(before, after)["classification"]
        == "RANKING-METRIC REGRESSED"
    )
    swapped = rerank.classify_evidence({"span1"}, {"span2"})
    assert swapped["classification"] == "EVIDENCE REGRESSED"
    assert swapped["lost_spans"] == ["span1"] and swapped["gained_spans"] == ["span2"]


def test_occupancy_reports_duplicate_source_slots():
    rows = [{"source": s} for s in ["a", "a", "a", "b", "c"]]
    stats = rerank.occupancy(rows)["5"]
    assert stats == {
        "distinct_source_count": 3,
        "repeated_source_slots": 2,
        "max_same_document_slots": 3,
        "max_same_document_fraction": 0.6,
    }


def test_frozen_pool_recall_and_known_candidate_generation_misses(inputs):
    oracle, evidence, units, control, chunks = inputs
    pools = rerank.read(rerank.OUTPUT / "candidate_pools.json")
    # Synthetic full ordering: exact pool membership, not an experiment inference.
    ranked = {
        p["query_id"]: rerank.sort_scores(
            p["candidates"], [float(-i) for i in range(p["pool_size"])]
        )
        for p in pools
    }
    result = rerank.evaluate_rows(
        oracle, evidence, chunks, units, control, pools, ranked
    )
    assert result["candidate_pool"]["gating_33"]["evidence_recall"] == pytest.approx(
        0.9696969697
    )
    assert (
        result["aggregate_metrics"]["control"]["gating_33_excludes_q25_q26"][
            "evidence_span_recall@5"
        ]
        == 0.78787879
    )
    assert (
        result["aggregate_metrics"]["A"]["gating_33_excludes_q25_q26"][
            "evidence_span_recall@5"
        ]
        == 0.84848485
    )
    by_id = {r["query_id"]: r for r in result["per_query"]}
    assert by_id["Q19"]["candidate_pool"]["missing_spans"]
    assert by_id["Q30"]["candidate_pool"]["missing_spans"]
    for qid in ["Q09", "Q13", "Q16", "Q17", "Q21", "Q22"]:
        assert by_id[qid]["candidate_pool"]["evidence_recall"] == 1.0
    assert not by_id["Q25"]["gating_eligible"] and not by_id["Q26"]["gating_eligible"]
    assert result["absence_metrics"] is None


def test_h1_detection_keeps_h1_only_root_and_excludes_body():
    assert rerank.is_h1(
        {
            "structural_segments": [
                {"kind": "seam"},
                {"kind": "heading", "heading_level": 1},
            ]
        }
    )
    assert not rerank.is_h1(
        {
            "structural_segments": [
                {"kind": "heading", "heading_level": 1},
                {"kind": "paragraph"},
            ]
        }
    )


def test_offline_entry_guard_denies_socket_in_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'from scripts.phase5a2_shadow_ab import install_offline_guard; install_offline_guard(); import socket; socket.create_connection(("example.com",443))',
        ],
        cwd=rerank.ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "OFFLINE_NETWORK_FORBIDDEN" in result.stderr


def test_two_real_runs_are_identical_and_failure_is_preserved():
    first_path = rerank.OUTPUT / "run-1/results.json"
    second_path = rerank.OUTPUT / "run-2/results.json"
    assert first_path.read_bytes() == second_path.read_bytes()
    result = rerank.read(first_path)
    for run in ("run-1", "run-2"):
        runtime = rerank.read(rerank.OUTPUT / run / "runtime.json")
        assert runtime["result_sha256"] == ab.sha256_json(result)
    assert result["scoring_runner_sha256"] == rerank.file_hash(
        rerank.ROOT / "scripts/phase5a2h_rerank.py"
    )
    assert result["token_safety"] == {
        "candidate_pairs": 1046,
        "max_pair_tokens": 293,
        "pairs_truncated": 0,
        "pairs_exceeding_max_input": 0,
    }
    assert result["quality_gate"]["material_evidence_improvement"]
    assert result["quality_gate"]["critical_evidence_regressions"] == [
        "Q13",
        "Q20",
        "Q22",
    ]
    assert result["quality_gate"]["quality_pass"] is False
    by_id = {r["query_id"]: r for r in result["per_query"]}
    assert by_id["Q20"]["critical_spans_lost_vs_control"]
    assert not by_id["Q13"]["critical_spans_lost_vs_control"]
    assert by_id["Q13"]["critical_spans_lost_vs_A"]
    assert by_id["Q34"]["evidence_comparison"]["classification"] == "EVIDENCE SAME"
    assert (
        by_id["Q34"]["ranking_comparison"]["classification"]
        == "RANKING-METRIC IMPROVED"
    )
