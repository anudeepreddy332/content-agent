"""Adversarial structure, provenance and full-serialization packing boundaries."""

from copy import deepcopy
import json
import subprocess
import sys

import pytest

from scripts import phase5a1_shadow as shadow
from scripts import phase5a2_shadow_ab as ab
from scripts import phase5a2b_packed as packed


@pytest.fixture
def setup():
    return shadow.MiniLMTokenizer(), json.loads(packed.CONTRACT.read_text())


def fixture_manifest(tmp_path, text, tokenizer):
    (tmp_path / "fixture.md").write_bytes(text.encode())
    doc, blocks = shadow.parse_document(text.encode(), "fixture.md")
    children, errors = shadow.chunk_document(text.encode(), doc, blocks, tokenizer)
    shadow.validate_document(text.encode(), doc, blocks, children, errors, tokenizer)
    assert not errors
    return {"documents": [doc], "blocks": blocks, "children": children}


def test_packs_adjacent_prose_and_nested_list_without_mutating_B(tmp_path, setup):
    tokenizer, contract = setup
    text = "# Title\n\n## Section\n\nFirst paragraph.\n\n- item\n  - nested\n\nLast paragraph.\n"
    b = fixture_manifest(tmp_path, text, tokenizer)
    original = deepcopy(b)
    p, report = packed.pack(b, tokenizer, contract, tmp_path)
    assert b == original
    assert len(p["children"]) == 3
    unit = p["children"][-1]
    assert len(unit["constituent_child_ids"]) == 3
    assert unit["canonical_content"] == text[text.index("First") :]
    assert report["packing_groups"] == 1
    assert {u["chunk_id"] for u in p["children"]}.isdisjoint(
        c["chunk_id"] for c in b["children"]
    )


@pytest.mark.parametrize(
    "barrier",
    [
        "## New section\n",
        "```python\nx=1\n```\n",
        "| A |\n|---|\n| x |\n",
        "---\n",
        "> quoted\n",
        "    code\n",
        "<div>html</div>\n",
    ],
)
def test_protected_blocks_and_headings_are_never_crossed(tmp_path, setup, barrier):
    tokenizer, contract = setup
    b = fixture_manifest(
        tmp_path, "# Title\n\nBefore.\n\n" + barrier + "\nAfter.\n", tokenizer
    )
    p, _ = packed.pack(b, tokenizer, contract, tmp_path)
    assert all(len(u["constituent_child_ids"]) == 1 for u in p["children"])
    assert [u["retrieval_text"] for u in p["children"]] == [
        c["retrieval_text"] for c in b["children"]
    ]


@pytest.mark.parametrize("words,expected", [(253, 1), (254, 2)])
def test_254_full_tokens_pack_but_255_do_not(tmp_path, setup, words, expected):
    tokenizer, contract = setup
    text = "hello\n\n" + " ".join(["hello"] * (words - 1))
    assert tokenizer.count(shadow.serialize("fixture", [], text)) == words + 1
    b = fixture_manifest(tmp_path, text, tokenizer)
    p, _ = packed.pack(b, tokenizer, contract, tmp_path)
    assert len(p["children"]) == expected
    assert all(
        u["embedding_content_token_count"] <= 254
        and u["embedding_total_token_count"] <= 256
        for u in p["children"]
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_source_bridge_includes_exact_whitespace_unicode_and_bytes(
    tmp_path, setup, newline
):
    tokenizer, contract = setup
    text = newline.join(["# Résumé 😀", "", "é 漢字", "", "naïve e\u0301", ""])
    b = fixture_manifest(tmp_path, text, tokenizer)
    p, _ = packed.pack(b, tokenizer, contract, tmp_path)
    u = p["children"][-1]
    assert len(u["constituent_child_ids"]) == 2
    assert u["canonical_content"] == text[text.index("é 漢") :]
    span = u["source_spans"][0]
    assert (
        text.encode()[span["source_byte_start"] : span["source_byte_end"]].decode()
        == u["canonical_content"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "different_document",
        "different_parent",
        "different_heading",
        "gap",
        "overlap",
        "nonconsecutive_block",
    ],
)
def test_incompatible_adjacency_is_rejected(tmp_path, setup, mutation):
    tokenizer, contract = setup
    b = fixture_manifest(tmp_path, "one\n\ntwo\n", tokenizer)
    left, right = deepcopy(b["children"])
    blocks = {x["block_id"]: deepcopy(x) for x in b["blocks"]}
    raw = "one\n\ntwo\n"
    if mutation == "different_document":
        right["document_version"] = "other"
    elif mutation == "different_parent":
        right["parent_block_id"] = "other"
    elif mutation == "different_heading":
        right["heading_path"] = ["other"]
    elif mutation == "gap":
        raw = "one\nXtwo\n"
    elif mutation == "overlap":
        right["source_spans"][0]["source_char_start"] = 1
    else:
        blocks[right["origin_block_id"]]["ordinal"] += 1
    assert not packed.compatible(left, right, blocks, raw, contract)


@pytest.mark.parametrize(
    "tamper",
    [
        "source_span",
        "bytes",
        "content",
        "id",
        "membership",
        "token_count",
        "source_hash",
    ],
)
def test_validator_rejects_provenance_identity_and_membership_corruption(
    tmp_path, setup, tamper
):
    tokenizer, contract = setup
    text = "# Title\n\none\n\ntwo\n"
    b = fixture_manifest(tmp_path, text, tokenizer)
    p, _ = packed.pack(b, tokenizer, contract, tmp_path)
    u = p["children"][-1]
    if tamper == "source_span":
        u["source_spans"][0]["source_char_end"] -= 1
    elif tamper == "bytes":
        u["source_spans"][0]["source_byte_start"] += 1
    elif tamper == "content":
        u["canonical_content"] += "invented"
    elif tamper == "id":
        u["chunk_id"] = "bad"
    elif tamper == "membership":
        u["constituent_child_ids"].reverse()
    elif tamper == "source_hash":
        u["source_sha256"] = "bad"
    else:
        u["embedding_content_token_count"] = 1
    with pytest.raises(packed.PackingError):
        packed.validate_packed(
            b,
            p["children"],
            {"fixture.md": shadow.Source(text.encode())},
            tokenizer,
            contract,
        )


def test_gapped_repeated_fences_and_tables_stay_exact_singletons(tmp_path, setup):
    tokenizer, contract = setup
    text = (
        "# Title\n\n```python\n"
        + "print(1)\n" * 130
        + "```\n\n| A |\n|---|\n"
        + "| row |\n" * 150
    )
    b = fixture_manifest(tmp_path, text, tokenizer)
    p, _ = packed.pack(b, tokenizer, contract, tmp_path)
    assert len(p["children"]) == len(b["children"])
    for a, u in zip(b["children"], p["children"]):
        assert a["source_spans"] == u["source_spans"]
        assert a["retrieval_text"] == u["retrieval_text"]


def test_frozen_B_and_packed_builds_are_deterministic(setup):
    tokenizer, contract = setup
    b, _ = shadow.build_corpus(tokenizer)
    assert b == json.loads(ab.CANDIDATE_B_MANIFEST.read_text())
    assert (
        shadow.sha(shadow.canonical_json(b))
        == "a9402fea0c4fbbae6613346370c0759b0239d3d870a386526980e1c40815e852"
    )
    first = packed.pack(b, tokenizer, contract)
    second = packed.pack(b, tokenizer, contract)
    assert first == second
    assert len(first[0]["children"]) < len(b["children"])


def test_offline_socket_attempt_is_fatal_and_tracing_disabled():
    code = (
        "from scripts.phase5a2b_packed import ab; ab.install_offline_guard(); "
        "import os,socket; assert os.environ['LANGSMITH_TRACING']=='false'; "
        "socket.create_connection(('example.com',443))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=packed.ROOT
    )
    assert result.returncode != 0
    assert "OFFLINE_NETWORK_FORBIDDEN" in result.stderr


def test_frozen_classification_preserves_mixed_regressions():
    before = dict.fromkeys(ab.QUALITY_KEYS, 1.0)
    after = {**before, "evidence_span_recall@5": 0.5, "mrr@10": 2.0}
    assert ab._classification(before, after) == "REGRESSED"


def test_historical_control_check_reports_score_drift_but_rejects_rank_or_metric_drift():
    before = {
        "top10": [{"chunk_id": "a", "native_score": 0.5}],
        "metrics": {"recall@5": 1.0},
    }
    after = deepcopy(before)
    after["top10"][0]["native_score"] += 0.0000001
    result = packed.historical_control_check(after, before)
    assert result["native_score_differences"] == 1
    after["top10"][0]["chunk_id"] = "b"
    with pytest.raises(packed.PackingError, match="ranking/metric drift"):
        packed.historical_control_check(after, before)
    after = deepcopy(before)
    after["metrics"]["recall@5"] = 0.5
    with pytest.raises(packed.PackingError, match="ranking/metric drift"):
        packed.historical_control_check(after, before)


def test_artifacts_bind_frozen_contract_controls_and_unchanged_packing():
    import ast

    initial = json.loads((packed.OUTPUT / "experiment_manifest.json").read_text())
    effective = packed.freeze(packed.OUTPUT)
    correction = json.loads(
        (packed.OUTPUT / "control_check_correction.json").read_text()
    )
    assert (
        initial["packing_contract"]
        == effective["packing_contract"]
        == json.loads(packed.CONTRACT.read_text())
    )
    assert initial["held_constant"] == effective["held_constant"]
    before = ast.parse((packed.OUTPUT / "initial_runner.py.txt").read_text())
    after = ast.parse((packed.ROOT / "scripts/phase5a2b_packed.py").read_text())
    for tree in (before, after):
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        for name, digest in correction[
            "unchanged_packing_and_comparison_function_ast_sha256"
        ].items():
            assert (
                ab.sha256_bytes(
                    ast.dump(functions[name], include_attributes=False).encode()
                )
                == digest
            )


def test_two_full_reruns_include_identical_embeddings_rank_scores_metrics_and_packing():
    first = json.loads((packed.OUTPUT / "run-1/determinism_report.json").read_text())
    second = json.loads((packed.OUTPUT / "run-2/determinism_report.json").read_text())
    assert first == second
    result = json.loads((packed.OUTPUT / "results.json").read_text())
    assert ab.sha256_json(result) == first["result_sha256"]
    manifest = json.loads(
        (packed.OUTPUT / "candidate_b_packed_manifest.json").read_text()
    )
    assert ab.sha256_json(manifest) == first["packed_fingerprint"]
    assert (
        manifest["frozen_B_fingerprint"]
        == "a9402fea0c4fbbae6613346370c0759b0239d3d870a386526980e1c40815e852"
    )


def test_complete_three_arm_metrics_and_every_query_classification():
    result = json.loads((packed.OUTPUT / "results.json").read_text())
    assert result["absence_metrics"] is None
    assert (
        len([r for r in result["per_query_comparison"] if r["gating_eligible"]]) == 33
    )
    for arm in ("A", "B", "B-PACKED"):
        for channel in ("dense", "bm25", "hybrid"):
            metrics = result["arms"][arm]["aggregate_metrics"][channel][
                "gating_33_excludes_q25_q26"
            ]
            for k in ab.K_VALUES:
                for prefix in ("recall", "precision", "ndcg", "evidence_span_recall"):
                    assert f"{prefix}@{k}" in metrics
                assert "unique_sources" in result["diversity"][arm][channel][str(k)]
            assert "mrr@10" in metrics
    for row in result["per_query_comparison"]:
        for channel in row["channels"].values():
            for control in ("A", "B"):
                assert channel["vs_" + control]["classification"] == ab._classification(
                    channel["metrics"][control], channel["metrics"]["B-PACKED"]
                )
    assert result["B_PACKED_ready_to_advance"] is False
