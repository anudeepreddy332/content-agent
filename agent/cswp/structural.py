"""Production structural Markdown parser and Candidate-B child chunker."""

from __future__ import annotations

import argparse
import bisect
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import re
import sys
from collections import Counter

from agent.cswp.constants import (
    ABC_CONTRACT as CONTRACT,
    BASELINE_MANIFEST as BASELINE,
    CHUNKER_VERSION,
    CONTENT_TOKEN_LIMIT,
    PARSER_NAME,
    PARSER_VERSION,
    ROOT,
    SERIALIZATION_VERSION,
)
from agent.cswp.errors import StructuralError
from agent.cswp.identity import canonical_json, identity, sha
from agent.cswp.offline import install_offline_guard
from agent.cswp.source import Source
from agent.cswp.tokenizer import MiniLMTokenizer

LIMIT = CONTENT_TOKEN_LIMIT
ShadowError = StructuralError


def parse_document(
    raw, source_uri, tenant_scope="phase5-shadow", parser_version=PARSER_VERSION
):
    from markdown_it import MarkdownIt

    if importlib.metadata.version(PARSER_NAME) != parser_version:
        raise ShadowError("parser version differs from declared identity")
    path = PurePosixPath(source_uri)
    if path.is_absolute() or ".." in path.parts or "\\" in source_uri:
        raise ShadowError("source URI must be a normalized repository-relative path")
    source = Source(raw)
    document_id = identity(
        "document", {"tenant_scope": tenant_scope, "normalized_source_uri": str(path)}
    )
    document_version = identity(
        "document-version", {"document_id": document_id, "source_sha256": sha(raw)}
    )
    tokens = MarkdownIt("commonmark", {"html": True}).enable("table").parse(source.text)
    top = [
        (i, token)
        for i, token in enumerate(tokens)
        if token.level == 0 and token.map and token.nesting != -1
    ]
    title = next(
        (
            tokens[i + 1].content
            for i, token in top
            if token.type == "heading_open" and token.tag == "h1"
        ),
        path.stem,
    )
    document = {
        "document_id": document_id,
        "document_version": document_version,
        "source_uri": str(path),
        "source_path": str(path),
        "source_sha256": sha(raw),
        "document_type": "markdown",
        "tenant_scope": tenant_scope,
        "parser_name": PARSER_NAME,
        "parser_version": parser_version,
        "title": title,
        "parser_preset": "commonmark+table;html=true",
    }
    blocks, stack = [], []
    cursor = 0

    def add(kind, start, end, token_index=None, heading=None):
        nonlocal stack
        if heading:
            level, label = heading
            stack = [entry for entry in stack if entry[0] < level]
        parent = stack[-1][2] if stack else None
        heading_path = [entry[1] for entry in stack] + ([heading[1]] if heading else [])
        span = source.span(start, end)
        text = source.extract(span)
        block = {
            **span,
            "document_id": document_id,
            "document_version": document_version,
            "source_path": str(path),
            "source_sha256": sha(raw),
            "parent_block_id": parent,
            "heading_path": heading_path,
            "block_type": kind,
            "ordinal": len(blocks),
            "structural_path": [entry[3] for entry in stack] + [len(blocks)],
            "canonical_content": text,
            "canonical_content_sha256": sha(text),
            "parser_name": PARSER_NAME,
            "parser_version": parser_version,
            "legal_units": [],
            "context_before": [],
            "context_after": [],
        }
        fields = json.loads(CONTRACT.read_text())["identity_contract"][
            "block_id_payload"
        ]
        block["block_id"] = identity("block", {key: block[key] for key in fields})
        if token_index is not None:
            token = tokens[token_index]
            lo, hi = token.map
            if kind == "list":
                items = [
                    t.map
                    for t in tokens[token_index + 1 :]
                    if t.type == "list_item_open"
                    and t.level == 1
                    and t.map
                    and lo <= t.map[0] < hi
                ]
                starts = [item[0] for item in items]
                block["legal_units"] = [
                    source.line_span(a, b) for a, b in zip(starts, starts[1:] + [hi])
                ]
            elif kind == "fenced_code":
                # An oversized unclosed fence fails closed; no invented source text.
                closing = source.lines[hi - 1].strip()
                if hi - lo >= 2 and re.fullmatch(
                    re.escape(token.markup[0]) + "{" + str(len(token.markup)) + ",}",
                    closing,
                ):
                    block["context_before"] = [
                        source.line_span(lo, lo + 1, "repeated_fence")
                    ]
                    block["context_after"] = [
                        source.line_span(hi - 1, hi, "repeated_fence")
                    ]
                    block["legal_units"] = [
                        source.line_span(line, line + 1)
                        for line in range(lo + 1, hi - 1)
                    ]
            elif kind == "table":
                block["context_before"] = [
                    source.line_span(lo, lo + 2, "repeated_table_header")
                ]
                block["legal_units"] = [
                    source.line_span(line, line + 1) for line in range(lo + 2, hi)
                ]
            elif kind == "code":
                block["legal_units"] = [
                    source.line_span(line, line + 1) for line in range(lo, hi)
                ]
        blocks.append(block)
        if heading:
            stack.append((heading[0], heading[1], block["block_id"], block["ordinal"]))

    types = {
        "heading_open": "heading",
        "paragraph_open": "paragraph",
        "bullet_list_open": "list",
        "ordered_list_open": "list",
        "fence": "fenced_code",
        "code_block": "code",
        "table_open": "table",
        "blockquote_open": "blockquote",
        "html_block": "html",
        "hr": "thematic_break",
    }
    for i, token in top:
        start, end = (source.starts[line] for line in token.map)
        if start < cursor:
            raise ShadowError("overlapping top-level parser source maps")
        if source.text[cursor:start].strip():
            add("source_gap", cursor, start)  # e.g. link-reference definitions
        heading = (
            (int(token.tag[1:]), tokens[i + 1].content)
            if token.type == "heading_open"
            else None
        )
        add(types.get(token.type, token.type), start, end, i, heading)
        cursor = end
    if source.text[cursor:].strip():
        add("source_gap", cursor, len(source.text))
    siblings = {}
    for block in blocks:
        siblings.setdefault(block["parent_block_id"], []).append(block)
    for group in siblings.values():
        for i, block in enumerate(group):
            block["previous_block_id"] = group[i - 1]["block_id"] if i else None
            block["next_block_id"] = (
                group[i + 1]["block_id"] if i + 1 < len(group) else None
            )
    return document, blocks


def serialize(title, heading_path, content):
    """Literal LF separators, exact title/labels, no stripping of child content."""
    return title + "\n" + " > ".join(heading_path) + "\n\n" + content


def chunk_document(raw, document, blocks, tokenizer, chunker_version=CHUNKER_VERSION):
    source = Source(raw)
    children, overflows = [], []
    id_fields = json.loads(CONTRACT.read_text())["identity_contract"][
        "chunk_id_payload"
    ]
    for block in blocks:

        def rendered(spans):
            content = "".join(source.extract(span) for span in spans)
            return content, serialize(document["title"], block["heading_path"], content)

        def fits(spans):
            return tokenizer.count(rendered(spans)[1]) <= LIMIT

        def emit(spans, split):
            content, retrieval = rendered(spans)
            count, total = (
                tokenizer.count(retrieval),
                tokenizer.count(retrieval, specials=True),
            )
            if count > LIMIT or total != count + 2 or total > 256:
                raise ShadowError("CHUNK_OVERFLOW: final serialized child invariant")
            child = {
                **{
                    key: document[key]
                    for key in (
                        "document_id",
                        "document_version",
                        "source_path",
                        "source_sha256",
                        "parser_name",
                        "parser_version",
                    )
                },
                "parent_block_id": block["parent_block_id"],
                "heading_path": block["heading_path"],
                "origin_block_id": block["block_id"],
                "ordered_block_ids": [block["block_id"]],
                "chunker_version": chunker_version,
                "chunk_index": len(children),
                "retrieval_serialization_version": SERIALIZATION_VERSION,
                "source_spans": spans,
                "canonical_content": content,
                "canonical_content_sha256": sha(content),
                "retrieval_text": retrieval,
                "retrieval_text_sha256": sha(retrieval),
                "embedding_tokenizer_id": tokenizer.tokenizer_id,
                "embedding_content_token_count": count,
                "embedding_special_token_count": total - count,
                "embedding_total_token_count": total,
                "split_policy": split,
                "duplicated_context_spans": [
                    span for span in spans if span["role"] != "content"
                ],
            }
            child["chunk_id"] = identity(
                "chunk", {key: child[key] for key in id_fields}
            )
            children.append(child)

        def overflow(spans, reason):
            overflows.append(
                {
                    "status": "CHUNK_OVERFLOW",
                    "reason": reason,
                    "document_version": document["document_version"],
                    "block_id": block["block_id"],
                    "source_path": document["source_path"],
                    "source_spans": spans,
                    "embedding_content_token_count": tokenizer.count(
                        rendered(spans)[1]
                    ),
                }
            )

        whole = [source.span(block["source_char_start"], block["source_char_end"])]
        if fits(whole):
            emit(whole, "intact")
            continue
        before, after = block["context_before"], block["context_after"]
        kind = block["block_type"]
        units = block["legal_units"]
        if kind == "paragraph":
            start, end = block["source_char_start"], block["source_char_end"]
            content = source.text[start:end]
            # Paragraph is already atomic; prefer sentence endings before offsets.
            boundaries = (
                [start]
                + [start + m.end() for m in re.finditer(r"(?<=[.!?])\s+", content)]
                + [end]
            )
            boundaries = sorted(set(boundaries))
            units = [source.span(a, b) for a, b in zip(boundaries, boundaries[1:])]
        if not units or not fits(before + after):
            overflow(whole, "indivisible block or metadata/context exhausts capacity")
            continue
        pending = []
        for unit in units:
            if fits(before + pending + [unit] + after):
                pending.append(unit)
                continue
            if pending:
                emit(before + pending + after, kind + "-units")
                pending = []
            if fits(before + [unit] + after):
                pending = [unit]
            elif kind == "paragraph":
                start, end = unit["source_char_start"], unit["source_char_end"]
                offsets = tokenizer.offsets(source.text[start:end])
                ends = sorted({start + b for a, b in offsets if b > 0} | {end})
                cursor, best = start, None
                for boundary in ends:
                    candidate = source.span(cursor, boundary)
                    if fits([candidate]):
                        best = boundary
                    else:
                        if best is None:
                            overflow(
                                [candidate],
                                "smallest tokenizer-offset unit exceeds capacity",
                            )
                            cursor = boundary
                        else:
                            emit(
                                [source.span(cursor, best)],
                                "paragraph-tokenizer-offsets",
                            )
                            cursor = best
                            if fits([source.span(cursor, boundary)]):
                                best = boundary
                                continue
                            overflow(
                                [source.span(cursor, boundary)],
                                "smallest tokenizer-offset unit exceeds capacity",
                            )
                            cursor = boundary
                        best = None
                if cursor < end:
                    tail = source.span(cursor, end)
                    if fits([tail]):
                        pending = [tail]
                    else:
                        overflow(
                            [tail], "remaining tokenizer-offset unit exceeds capacity"
                        )
            else:
                overflow(
                    before + [unit] + after,
                    "smallest legal " + kind + " unit exceeds capacity",
                )
        if pending:
            emit(before + pending + after, kind + "-units")
    return children, overflows


def validate_document(raw, document, blocks, children, overflows, tokenizer):
    """Reconstruct bytes, IDs, tokens, edges and coverage; reject damaged artifacts."""
    source = Source(raw)
    failures = []
    contract = json.loads(CONTRACT.read_text())["identity_contract"]
    by_id = {block["block_id"]: block for block in blocks}
    coverage = {key: [] for key in by_id}
    expected_document_id = identity(
        "document",
        {
            "tenant_scope": document["tenant_scope"],
            "normalized_source_uri": document["source_uri"],
        },
    )
    expected_version = identity(
        "document-version",
        {
            "document_id": expected_document_id,
            "source_sha256": sha(raw),
        },
    )
    if (
        document["document_id"] != expected_document_id
        or document["document_version"] != expected_version
        or document["source_sha256"] != sha(raw)
    ):
        failures.append("document identity mismatch")
    if len(by_id) != len(blocks) or len({c["chunk_id"] for c in children}) != len(
        children
    ):
        failures.append("duplicate logical IDs")
    cursor = 0
    for ordinal, block in enumerate(blocks):
        if (
            block["ordinal"] != ordinal
            or block["source_char_start"] < cursor
            or source.text[cursor : block["source_char_start"]].strip()
        ):
            failures.append("block source ordering or coverage mismatch")
        cursor = block["source_char_end"]
    if source.text[cursor:].strip():
        failures.append("unrepresented source tail")
    for block in blocks:
        span = source.span(block["source_char_start"], block["source_char_end"])
        if any(block[k] != v for k, v in span.items()):
            failures.append("block span mismatch")
        if (
            source.extract(span) != block["canonical_content"]
            or sha(source.extract(span)) != block["canonical_content_sha256"]
        ):
            failures.append("block source mismatch")
        if (
            identity("block", {k: block[k] for k in contract["block_id_payload"]})
            != block["block_id"]
        ):
            failures.append("block ID mismatch")
        parent = by_id.get(block["parent_block_id"])
        if block["parent_block_id"] and (
            parent is None
            or parent["block_type"] != "heading"
            or parent["ordinal"] >= block["ordinal"]
        ):
            failures.append("invalid structural parent")
        siblings = [
            b for b in blocks if b["parent_block_id"] == block["parent_block_id"]
        ]
        position = siblings.index(block)
        if block["previous_block_id"] != (
            siblings[position - 1]["block_id"] if position else None
        ) or block["next_block_id"] != (
            siblings[position + 1]["block_id"] if position + 1 < len(siblings) else None
        ):
            failures.append("neighbor reading order mismatch")
        for edge in ("previous_block_id", "next_block_id"):
            other = by_id.get(block[edge])
            if block[edge] and (
                other is None or other["parent_block_id"] != block["parent_block_id"]
            ):
                failures.append("neighbor crosses parent")
    for index, child in enumerate(children):
        block = by_id.get(child["origin_block_id"])
        if block is None:
            failures.append("unknown origin block")
            continue
        for key in (
            "document_id",
            "document_version",
            "source_path",
            "source_sha256",
            "parser_name",
            "parser_version",
        ):
            if child[key] != document[key] or block[key] != document[key]:
                failures.append("provenance identity mismatch")
        if (
            child["chunk_index"] != index
            or child["heading_path"] != block["heading_path"]
            or child["embedding_tokenizer_id"] != tokenizer.tokenizer_id
        ):
            failures.append("child metadata mismatch")
        if (
            child["embedding_special_token_count"] != 2
            or child["retrieval_serialization_version"] != SERIALIZATION_VERSION
        ):
            failures.append("child serialization identity mismatch")
        spans = child["source_spans"]
        contexts = block["context_before"] + block["context_after"]
        if child["duplicated_context_spans"] != [
            s for s in spans if s["role"] != "content"
        ]:
            failures.append("duplicated context accounting mismatch")
        if child["split_policy"] == "intact":
            if spans != [
                source.span(block["source_char_start"], block["source_char_end"])
            ]:
                failures.append("intact block was changed")
        elif block["block_type"] != "paragraph":
            if [s for s in spans if s["role"] != "content"] != contexts or any(
                s not in block["legal_units"] for s in spans if s["role"] == "content"
            ):
                failures.append("illegal structural split")
        text = ""
        for span in child["source_spans"]:
            expected = source.span(
                span["source_char_start"], span["source_char_end"], span["role"]
            )
            if span != expected or source.raw[
                span["source_byte_start"] : span["source_byte_end"]
            ].decode("utf-8") != source.extract(span):
                failures.append("child source span mismatch")
            if (
                not block["source_char_start"]
                <= span["source_char_start"]
                < span["source_char_end"]
                <= block["source_char_end"]
            ):
                failures.append("structural boundary violation")
            text += source.extract(span)
            coverage[block["block_id"]].append(
                (span["source_char_start"], span["source_char_end"])
            )
        retrieval = serialize(document["title"], block["heading_path"], text)
        if (
            child["canonical_content"] != text
            or child["canonical_content_sha256"] != sha(text)
            or child["retrieval_text"] != retrieval
            or child["retrieval_text_sha256"] != sha(retrieval)
        ):
            failures.append("child reconstruction mismatch")
        if child["parent_block_id"] != block["parent_block_id"] or child[
            "ordered_block_ids"
        ] != [block["block_id"]]:
            failures.append("structural boundary violation")
        count, total = tokenizer.count(retrieval), tokenizer.count(retrieval, True)
        if (
            count > LIMIT
            or total != count + 2
            or child["embedding_content_token_count"] != count
            or child["embedding_total_token_count"] != total
        ):
            failures.append("token count mismatch")
        if (
            identity("chunk", {k: child[k] for k in contract["chunk_id_payload"]})
            != child["chunk_id"]
        ):
            failures.append("chunk ID mismatch")
    for failure in overflows:
        coverage[failure["block_id"]].extend(
            (s["source_char_start"], s["source_char_end"])
            for s in failure["source_spans"]
        )
    for block in blocks:
        cursor = block["source_char_start"]
        for start, end in sorted(coverage[block["block_id"]]):
            if start > cursor:
                failures.append("unrepresented block content")
            cursor = max(cursor, end)
        if cursor != block["source_char_end"]:
            failures.append("unrepresented block tail")
    if failures:
        raise ShadowError("; ".join(sorted(set(failures))))


def build_corpus(tokenizer, root=ROOT):
    baseline = json.loads(BASELINE.read_text())
    sources = baseline["corpus"]["ordered_source_manifest"]
    if sha(canonical_json(sources)) != baseline["corpus"]["aggregate_fingerprint"]:
        raise ShadowError("frozen corpus manifest fingerprint differs")
    documents, blocks, children, overflows = [], [], [], []
    for row in sources:
        raw = (root / row["path"]).read_bytes()
        if sha(raw) != row["source_sha256"]:
            raise ShadowError("frozen source changed: " + row["path"])
        document, parsed = parse_document(raw, row["path"])
        chunks, errors = chunk_document(raw, document, parsed, tokenizer)
        validate_document(raw, document, parsed, chunks, errors, tokenizer)
        documents.append(document)
        blocks.extend(parsed)
        children.extend(chunks)
        overflows.extend(errors)
    ids = [
        row[key]
        for rows, key in (
            (documents, "document_id"),
            (blocks, "block_id"),
            (children, "chunk_id"),
        )
        for row in rows
    ]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise ShadowError("duplicate logical IDs")
    counts = sorted(child["embedding_content_token_count"] for child in children)
    manifest = {
        "schema_version": "phase5a1_shadow_v1",
        "candidate": "B",
        "baseline_corpus_fingerprint": baseline["corpus"]["aggregate_fingerprint"],
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "serialization_version": SERIALIZATION_VERSION,
        "retrieval_text_grammar": 'title + LF + join(heading_path, " > ") + LF + LF + exact_child_content',
        "tokenizer": tokenizer.manifest,
        "documents": documents,
        "blocks": blocks,
        "children": children,
        "overflows": overflows,
    }
    report = {
        "source_count": len(documents),
        "block_count": len(blocks),
        "child_count": len(children),
        "block_types": dict(
            sorted(Counter(block["block_type"] for block in blocks).items())
        ),
        "token_distribution": {
            "min": min(counts, default=0),
            "p50": counts[len(counts) // 2] if counts else 0,
            "p95": counts[min(len(counts) - 1, (95 * len(counts) + 99) // 100 - 1)]
            if counts
            else 0,
            "max": max(counts, default=0),
            "histogram": dict(sorted(Counter(counts).items())),
        },
        "children_above_254": sum(n > LIMIT for n in counts),
        "overflow_count": len(overflows),
        "structural_boundary_violations": 0,
        "provenance_failures": 0,
        "duplicate_logical_ids": duplicates,
        "deterministic_fingerprint": sha(canonical_json(manifest)),
        "embeddings_generated": 0,
        "provider_calls": 0,
        "network_calls": 0,
        "production_retrieval_changed": False,
        "qdrant_mutated": False,
    }
    return manifest, report


def main():
    install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports/phase5/phase5a1"
    )
    args = parser.parse_args()
    # Restrict writes to the dedicated artifact subtree; never overwrite baseline A.
    destination = args.output_dir.resolve()
    if not destination.is_relative_to((ROOT / "reports/phase5/phase5a1").resolve()):
        raise ShadowError("output must remain in reports/phase5/phase5a1")
    destination.mkdir(parents=True, exist_ok=True)
    manifest, report = build_corpus(MiniLMTokenizer())
    manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
    (destination / "candidate_b_manifest.json").write_bytes(manifest_bytes)
    report["manifest_file_sha256"] = sha(manifest_bytes)
    (destination / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["overflow_count"]:
        raise SystemExit("CHUNK_OVERFLOW: shadow representation is not ready")


if __name__ == "__main__":
    main()
