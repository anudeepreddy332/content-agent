"""Candidate B source, structure, identity and actual MiniLM boundary qualification."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from scripts import phase5a1_shadow as shadow


@pytest.fixture
def tokenizer():
    return shadow.MiniLMTokenizer()


def build(text, tokenizer, **kwargs):
    raw = text.encode("utf-8")
    document, blocks = shadow.parse_document(raw, "fixture.md")
    children, errors = shadow.chunk_document(raw, document, blocks, tokenizer, **kwargs)
    shadow.validate_document(raw, document, blocks, children, errors, tokenizer)
    assert all(child["embedding_content_token_count"] <= 254 for child in children)
    return document, blocks, children, errors


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("A normal paragraph.\n", "paragraph"),
        ("- first\n  - nested\n- second\n", "list"),
        ("```python\nx = 1\n```\n", "fenced_code"),
        ("| A | B |\n|---|---|\n| 1 | 2 |\n", "table"),
        ("> quoted text\n> second line\n", "blockquote"),
        ("<div>html</div>\n", "html"),
        ("---\n", "thematic_break"),
        ("    indented code\n", "code"),
    ],
)
def test_fitting_structures_stay_byte_exact_and_intact(text, kind, tokenizer):
    _, blocks, children, errors = build(text, tokenizer)
    assert not errors
    assert len(blocks) == len(children) == 1
    assert blocks[0]["block_type"] == kind
    assert children[0]["canonical_content"] == text
    assert children[0]["split_policy"] == "intact"
    assert children[0]["retrieval_text"] == "fixture\n\n\n" + text


def test_nested_headings_parent_and_neighbor_boundaries(tokenizer):
    text = "# Title\n\nIntro.\n\n## A\n\nSame text.\n\n### Child\n\nNested.\n\n## B\n\nSame text.\n"
    _, blocks, children, errors = build(text, tokenizer)
    assert not errors
    by_text = {b["canonical_content"]: b for b in blocks}
    assert by_text["Nested.\n"]["heading_path"] == ["Title", "A", "Child"]
    assert by_text["Nested.\n"]["parent_block_id"] == by_text["### Child\n"]["block_id"]
    assert by_text["Nested.\n"]["next_block_id"] is None
    duplicate_text = [c for c in children if c["canonical_content"] == "Same text.\n"]
    assert len({c["chunk_id"] for c in duplicate_text}) == 2
    assert len({c["origin_block_id"] for c in duplicate_text}) == 2


def test_tiny_adjacent_blocks_are_preserved_without_cross_parent_packing(tokenizer):
    _, blocks, children, errors = build("# A\n\nx\n\ny\n\n## B\n\nz\n", tokenizer)
    assert not errors and len(blocks) == len(children) == 5
    assert all(len(c["ordered_block_ids"]) == 1 for c in children)


@pytest.mark.parametrize("words", [253, 254])
def test_exact_full_serialization_254_and_255_boundaries(words, tokenizer):
    # One title token + N payload tokens; empty breadcrumb contributes no tokens.
    text = " ".join(["hello"] * words)
    assert tokenizer.count(shadow.serialize("fixture", [], text)) == words + 1
    _, _, children, errors = build(text, tokenizer)
    assert not errors
    assert len(children) == (1 if words == 253 else 2)
    assert max(c["embedding_content_token_count"] for c in children) == 254
    assert all(c["embedding_total_token_count"] <= 256 for c in children)
    assert "".join(c["canonical_content"] for c in children) == text


@pytest.mark.parametrize(
    "text",
    [
        "This is a complete sentence. " * 240,
        "hello " * 900,
        "é 😀 naïve e\u0301 漢字. " * 240,
    ],
)
def test_oversized_prose_has_no_loss_overlap_or_unicode_corruption(text, tokenizer):
    _, _, children, errors = build(text, tokenizer)
    assert not errors and len(children) > 1
    assert "".join(c["canonical_content"] for c in children) == text
    assert all(not c["duplicated_context_spans"] for c in children)


def test_oversized_fence_repeats_only_exact_fences_and_complete_lines(tokenizer):
    body = "".join(f"print({i})\n" for i in range(220))
    text = "```python\n" + body + "```\n"
    _, blocks, children, errors = build(text, tokenizer)
    assert not errors and len(children) > 1
    assert len(blocks) == 1
    assert all(
        c["canonical_content"].startswith("```python\n")
        and c["canonical_content"].endswith("```\n")
        for c in children
    )
    assert "".join(c["canonical_content"][10:-4] for c in children) == body
    assert all(len(c["duplicated_context_spans"]) == 2 for c in children)


def test_table_rows_repeat_exact_header_and_never_split_cells(tokenizer):
    header = "| A | B |\n|---|---|\n"
    rows = "".join(f"| row {i} | value {i} |\n" for i in range(120))
    _, _, children, errors = build(header + rows, tokenizer)
    assert not errors and len(children) > 1
    assert all(c["canonical_content"].startswith(header) for c in children)
    assert "".join(c["canonical_content"][len(header) :] for c in children) == rows
    assert all(len(c["duplicated_context_spans"]) == 1 for c in children)


def test_list_splits_only_complete_items_including_nested_items(tokenizer):
    items = ["- item " + str(i) + "\n  - nested detail\n" for i in range(100)]
    _, blocks, children, errors = build("".join(items), tokenizer)
    assert not errors and len(children) > 1
    assert len(blocks) == 1 and blocks[0]["block_type"] == "list"
    assert "".join(c["canonical_content"] for c in children) == "".join(items)
    assert all(
        c["canonical_content"].startswith("- item")
        and c["canonical_content"].endswith("  - nested detail\n")
        for c in children
    )


@pytest.mark.parametrize(
    "text",
    [
        "```python\n" + "hello " * 300 + "\n```\n",
        "| A |\n|---|\n| " + "hello " * 300 + "|\n",
        "- " + "hello " * 300 + "\n",
        "# " + "hello " * 300 + "\n\nshort body\n",
        "```python\n" + "print(1)\n" * 300,
    ],
)
def test_indivisible_overflow_is_explicit_and_never_emitted_as_unsafe_child(
    text, tokenizer
):
    _, _, children, errors = build(text, tokenizer)
    assert errors
    assert all(e["status"] == "CHUNK_OVERFLOW" and e["source_spans"] for e in errors)
    assert all(c["embedding_content_token_count"] <= 254 for c in children)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_unicode_exact_byte_character_and_line_spans(newline, tokenizer):
    text = newline.join(
        ["# Résumé 😀", "", "é e\u0301 中文\u2028same line", "", "- item", ""]
    )
    document, blocks, children, errors = build(text, tokenizer)
    assert not errors
    paragraph = next(b for b in blocks if b["block_type"] == "paragraph")
    assert paragraph["start_line"] == paragraph["end_line"] == 3
    assert paragraph["source_byte_start"] > paragraph["source_char_start"]
    assert paragraph["canonical_content"] == "é e\u0301 中文\u2028same line" + newline
    assert document["title"] == "Résumé 😀"


def test_reference_definitions_and_setext_heading_preserve_source(tokenizer):
    text = "Title\n=====\n\n[link]: https://example.com\n\nSee [link].\n"
    _, blocks, _, errors = build(text, tokenizer)
    assert not errors
    assert blocks[0]["block_type"] == "heading"
    assert any(
        b["block_type"] == "source_gap" and "[link]:" in b["canonical_content"]
        for b in blocks
    )


def test_same_bytes_are_deterministic_and_source_change_versions_derived_ids(tokenizer):
    first = build("# Title\n\nbody", tokenizer)
    assert first == build("# Title\n\nbody", tokenizer)
    changed = build("# Title\n\nbody!", tokenizer)
    assert first[0]["document_id"] == changed[0]["document_id"]
    assert first[0]["document_version"] != changed[0]["document_version"]
    assert set(b["block_id"] for b in first[1]).isdisjoint(
        b["block_id"] for b in changed[1]
    )
    assert set(c["chunk_id"] for c in first[2]).isdisjoint(
        c["chunk_id"] for c in changed[2]
    )


def test_parser_and_chunker_version_changes_have_distinct_derived_identity(
    tokenizer, monkeypatch
):
    raw = b"# Title\n\nbody"
    document, blocks, children, _ = build(raw.decode(), tokenizer)
    changed, _ = shadow.chunk_document(
        raw, document, blocks, tokenizer, chunker_version="next-chunker"
    )
    assert {c["chunk_id"] for c in children}.isdisjoint(c["chunk_id"] for c in changed)
    # Simulate installation of a new parser version solely to exercise identity.
    monkeypatch.setattr(
        shadow.importlib.metadata, "version", lambda name: "next-parser"
    )
    next_doc, next_blocks = shadow.parse_document(
        raw, "fixture.md", parser_version="next-parser"
    )
    assert next_doc["document_version"] == document["document_version"]
    assert {b["block_id"] for b in blocks}.isdisjoint(
        b["block_id"] for b in next_blocks
    )
    next_children, _ = shadow.chunk_document(raw, next_doc, next_blocks, tokenizer)
    assert {c["chunk_id"] for c in children}.isdisjoint(
        c["chunk_id"] for c in next_children
    )


@pytest.mark.parametrize(
    "field", ["source_byte_start", "source_char_start", "start_line"]
)
def test_provenance_tampering_is_rejected(field, tokenizer):
    raw = b"# Title\n\nbody\n"
    document, blocks, children, errors = build(raw.decode(), tokenizer)
    children = copy.deepcopy(children)
    children[-1]["source_spans"][0][field] += 1
    with pytest.raises(shadow.ShadowError):
        shadow.validate_document(raw, document, blocks, children, errors, tokenizer)


def test_omitted_child_wrong_parent_and_wrong_source_identity_are_rejected(tokenizer):
    raw = b"# A\n\nalpha\n\n# B\n\nbeta\n"
    document, blocks, children, errors = build(raw.decode(), tokenizer)
    with pytest.raises(shadow.ShadowError, match="unrepresented"):
        shadow.validate_document(
            raw, document, blocks, children[:-1], errors, tokenizer
        )
    wrong = copy.deepcopy(children)
    wrong[-1]["parent_block_id"] = blocks[0]["block_id"]
    with pytest.raises(shadow.ShadowError, match="structural boundary"):
        shadow.validate_document(raw, document, blocks, wrong, errors, tokenizer)
    wrong = copy.deepcopy(children)
    wrong[-1]["source_sha256"] = "0" * 64
    with pytest.raises(shadow.ShadowError, match="provenance identity"):
        shadow.validate_document(raw, document, blocks, wrong, errors, tokenizer)


def test_missing_or_changed_frozen_tokenizer_fails_closed(tmp_path):
    with pytest.raises(shadow.ShadowError, match="unavailable or changed"):
        shadow.MiniLMTokenizer(tmp_path)
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(shadow.ShadowError, match="unavailable or changed"):
        shadow.MiniLMTokenizer(tmp_path)


def test_actual_transformers_tokenizer_matches_raw_backend(tokenizer):
    from transformers import AutoTokenizer

    actual = AutoTokenizer.from_pretrained(
        str(shadow.ROOT / "evals/fixtures/phase5a1_minilm_tokenizer"),
        local_files_only=True,
    )
    for text in [
        "hello " * 300,
        "café 😀\n```python\nx = 12\n```",
        "A\nA > B\n\n|x|y|",
    ]:
        assert len(
            actual(text, add_special_tokens=False, truncation=False)["input_ids"]
        ) == tokenizer.count(text)
        assert len(
            actual(text, add_special_tokens=True, truncation=False)["input_ids"]
        ) == tokenizer.count(text, True)


def test_two_run_frozen_corpus_manifest_and_all_source_spans(tokenizer):
    first, report1 = shadow.build_corpus(tokenizer)
    second, report2 = shadow.build_corpus(tokenizer)
    assert (
        shadow.canonical_json(first).encode() == shadow.canonical_json(second).encode()
    )
    assert report1 == report2
    assert report1["source_count"] == 20
    assert report1["children_above_254"] == report1["overflow_count"] == 0
    assert report1["duplicate_logical_ids"] == report1["provenance_failures"] == 0
    assert report1["structural_boundary_violations"] == 0
    assert set(first["children"][0]).issuperset(
        json.loads(shadow.CONTRACT.read_text())["canonical_schema"]["child"]["required"]
    )
    # Stored evidence must agree with an independent current reconstruction.
    checked = json.loads(
        (shadow.ROOT / "reports/phase5/phase5a1/candidate_b_manifest.json").read_text()
    )
    assert first == checked


def test_frozen_source_mutation_is_rejected(tokenizer, tmp_path):
    source_path = tmp_path / "kb/seed_docs/agentic-ai-production.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("changed source")
    with pytest.raises(shadow.ShadowError, match="frozen source changed"):
        shadow.build_corpus(tokenizer, root=tmp_path)


def test_process_guard_blocks_socket_before_optional_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'from scripts.phase5a1_shadow import install_offline_guard; install_offline_guard(); import socket; socket.create_connection(("example.com",443))',
        ],
        cwd=shadow.ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "OFFLINE_NETWORK_FORBIDDEN" in result.stderr
