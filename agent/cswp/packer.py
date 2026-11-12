"""Production CSWP-v1 contiguous structural window packer."""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from agent.cswp.constants import CANDIDATE, ROOT
from agent.cswp.errors import CSWPError
from agent.cswp.identity import canonical_json, sha, sha256_json
from agent.cswp.source import Source

PACKABLE_BLOCKS = {"paragraph", "list"}
BARRIER_BLOCKS = {
    "fenced_code",
    "code",
    "table",
    "blockquote",
    "html",
    "thematic_break",
    "source_gap",
}


def content_interval(child):
    spans = child["source_spans"]
    if not spans or any(s["role"] != "content" for s in spans):
        return None
    if any(
        a["source_char_end"] != b["source_char_start"] for a, b in zip(spans, spans[1:])
    ):
        return None
    return spans[0]["source_char_start"], spans[-1]["source_char_end"]


def heading_level(block):
    if block["block_type"] != "heading":
        return None
    text = block["canonical_content"].lstrip("\r\n")
    if text.startswith("#"):
        count = 0
        while count < len(text) and text[count] == "#":
            count += 1
        if 1 <= count <= 6 and (count == len(text) or text[count] in " \t\r\n"):
            return count
    lines = block["canonical_content"].rstrip("\r\n").split("\n")
    if len(lines) >= 2:
        under = lines[-1].strip()
        if under and set(under) <= {"="}:
            return 1
        if under and set(under) <= {"-"}:
            return 2
    raise CSWPError("unclassified heading markup")


def packable_block(block):
    kind = block["block_type"]
    if kind in PACKABLE_BLOCKS:
        return True
    if kind == "heading":
        return 2 <= heading_level(block) <= 6
    return False


def child_cover(child, blocks):
    interval = content_interval(child)
    if interval is not None:
        return interval
    block = blocks[child["origin_block_id"]]
    return block["source_char_start"], block["source_char_end"]


def breadcrumb_for(title, segments, start, end):
    first = next(
        (
            segment
            for segment in segments
            if segment["kind"] != "seam"
            and segment["start"] < end
            and segment["end"] > start
        ),
        None,
    )
    if first is None:
        return []
    present = {
        segment["heading_path"][-1]
        for segment in segments
        if segment["kind"] == "heading"
        and start <= segment["start"]
        and segment["end"] <= end
        and segment["heading_path"]
    }
    path = [label for label in first["heading_path"] if label not in present]
    return [label for label in path if label != title]


def serialize_slice(title, breadcrumb, slice_text):
    return title + "\n" + " > ".join(breadcrumb) + "\n\n" + slice_text


def token_counts(tokenizer, text):
    count = tokenizer.count(text)
    total = tokenizer.count(text, specials=True)
    return count, total


def fits(tokenizer, title, segments, start, end, source_text):
    if start >= end:
        return True
    retrieval = serialize_slice(
        title, breadcrumb_for(title, segments, start, end), source_text[start:end]
    )
    count, total = token_counts(tokenizer, retrieval)
    return count <= 254 and total == count + 2 and total <= 256


def make_segment(kind, start, end, block=None, child=None, packable=False):
    heading_path = list((block or child or {}).get("heading_path") or [])
    return {
        "kind": kind,
        "start": start,
        "end": end,
        "packable": packable,
        "heading_path": heading_path,
        "heading_level": heading_level(block) if block and kind == "heading" else None,
        "origin_block_id": None if block is None else block["block_id"],
        "constituent_child_ids": [] if child is None else [child["chunk_id"]],
        "block_type": None if block is None else block["block_type"],
    }


def build_ledger(children, blocks, source_text):
    ordered = sorted(
        children,
        key=lambda child: (child_cover(child, blocks)[0], child["chunk_index"]),
    )
    segments = []
    cursor = 0
    for child in ordered:
        start, end = child_cover(child, blocks)
        if start < cursor:
            if not packable_block(blocks[child["origin_block_id"]]) and segments:
                segments[-1]["constituent_child_ids"].append(child["chunk_id"])
                continue
            raise CSWPError("unlabelled packable overlap in B children")
        if cursor < start:
            gap = source_text[cursor:start]
            if gap.strip():
                raise CSWPError("non-whitespace source gap missing from B children")
            segments.append(
                make_segment("seam", cursor, start, packable=True)
            )
        block = blocks[child["origin_block_id"]]
        kind = block["block_type"]
        if kind == "heading":
            kind = "heading"
        packable = packable_block(block)
        if kind in BARRIER_BLOCKS:
            packable = False
        if kind == "heading" and heading_level(block) == 1:
            packable = False
        segments.append(
            make_segment(kind, start, end, block=block, child=child, packable=packable)
        )
        cursor = max(cursor, end)
    if cursor < len(source_text):
        gap = source_text[cursor:]
        if gap.strip():
            raise CSWPError("trailing non-whitespace uncovered")
        segments.append(make_segment("seam", cursor, len(source_text), packable=True))
    covered = [0] * (len(source_text) + 1)
    for segment in segments:
        for index in range(segment["start"], segment["end"]):
            covered[index] += 1
    if any(count != 1 for count in covered[: len(source_text)]):
        raise CSWPError("source coverage gap or unlabelled overlap")
    return segments


def core_groups(segments):
    groups, pending = [], []
    for segment in segments:
        if segment["packable"]:
            pending.append(segment)
            continue
        if pending and all(item["kind"] == "seam" for item in pending):
            groups.append(("barrier", pending + [segment]))
            pending = []
            continue
        if pending:
            groups.append(("packable", pending))
            pending = []
        groups.append(("barrier", [segment]))
    if pending:
        groups.append(("packable", pending))
    return groups


def split_packable(title, segments, source_text, tokenizer):
    cores, pending = [], []
    for segment in segments:
        candidate = pending + [segment]
        start, end = candidate[0]["start"], candidate[-1]["end"]
        if fits(tokenizer, title, candidate, start, end, source_text):
            pending = candidate
            continue
        if not pending:
            raise CSWPError("smallest legal unit exceeds 254")
        cores.append(pending)
        if not fits(
            tokenizer, title, [segment], segment["start"], segment["end"], source_text
        ):
            raise CSWPError("smallest legal unit exceeds 254")
        pending = [segment]
    if pending:
        cores.append(pending)
    return cores


def left_bound(core_start, segments):
    bound = 0
    for segment in segments:
        if segment["end"] <= core_start and not segment["packable"]:
            bound = segment["end"]
    return bound


def snap_overlap_start(start, core_start, core_end, title, segments, source_text, tokenizer):
    for segment in segments:
        if segment["kind"] != "heading" or not (
            segment["start"] < start < segment["end"]
        ):
            continue
        if fits(
            tokenizer, title, segments, segment["start"], core_end, source_text
        ):
            return segment["start"]
        return min(segment["end"], core_start)
    return start


def left_overlap_start(title, segments, core_start, core_end, source_text, tokenizer):
    lo = left_bound(core_start, segments)
    hi = core_start
    if not fits(tokenizer, title, segments, hi, core_end, source_text):
        raise CSWPError("core serialization overflow")
    while lo < hi:
        mid = (lo + hi) // 2
        if fits(tokenizer, title, segments, mid, core_end, source_text):
            hi = mid
        else:
            lo = mid + 1
    return snap_overlap_start(
        lo, core_start, core_end, title, segments, source_text, tokenizer
    )


def structural_segments(segments, retrieval_start, core_start, core_end):
    rows = []
    for segment in segments:
        start = max(segment["start"], retrieval_start)
        end = min(segment["end"], core_end)
        if start >= end:
            continue
        if end <= core_start:
            role = "left_overlap"
        elif segment["kind"] == "seam":
            role = "seam"
        else:
            role = "core"
        rows.append(
            {
                "role": role,
                "kind": segment["kind"],
                "source_char_start": start,
                "source_char_end": end,
                "heading_path": segment["heading_path"],
                "heading_level": segment["heading_level"],
                "origin_block_id": segment["origin_block_id"],
                "constituent_child_ids": segment["constituent_child_ids"],
            }
        )
    return rows


def make_unit(
    group,
    all_segments,
    document,
    source,
    tokenizer,
    contract,
    retrieval_start,
    core_start,
    core_end,
):
    title = document["title"]
    slice_text = source.text[retrieval_start:core_end]
    breadcrumb = breadcrumb_for(title, all_segments, retrieval_start, core_end)
    retrieval = serialize_slice(title, breadcrumb, slice_text)
    count, total = token_counts(tokenizer, retrieval)
    if count > 254 or total != count + 2 or total > 256:
        raise CSWPError("full serialization overflow")
    core_span = source.span(core_start, core_end, "core")
    spans = []
    if retrieval_start < core_start:
        spans.append(source.span(retrieval_start, core_start, "left_overlap"))
    spans.append(core_span)
    constituents = [
        cid
        for segment in group
        for cid in segment["constituent_child_ids"]
    ]
    payload = {
        "candidate": CANDIDATE,
        "packing_version": contract["packing_version"],
        "contract_sha256": sha256_json(contract),
        "document_id": document["document_id"],
        "document_version": document["document_version"],
        "source_path": document["source_path"],
        "source_sha256": document["source_sha256"],
        "parser_name": document["parser_name"],
        "parser_version": document["parser_version"],
        "core_char_start": core_start,
        "core_char_end": core_end,
        "retrieval_char_start": retrieval_start,
        "retrieval_char_end": core_end,
        "source_spans": spans,
        "constituent_child_ids": constituents,
        "ordered_block_ids": list(
            dict.fromkeys(
                segment["origin_block_id"]
                for segment in group
                if segment["origin_block_id"]
            )
        ),
        "retrieval_text_sha256": sha(retrieval),
        "embedding_tokenizer_id": tokenizer.tokenizer_id,
    }
    return {
        **payload,
        "chunk_id": "ca:cswp:" + sha256_json(payload),
        "heading_path": breadcrumb,
        "parent_block_id": next(
            (
                segment["origin_block_id"]
                for segment in group
                if segment["origin_block_id"]
            ),
            None,
        ),
        "canonical_content": source.text[core_start:core_end],
        "retrieval_text": retrieval,
        "structural_segments": structural_segments(
            all_segments, retrieval_start, core_start, core_end
        ),
        "embedding_content_token_count": count,
        "embedding_total_token_count": total,
    }


def pack_document(document, children, blocks, tokenizer, contract, root=ROOT):
    source = Source((root / document["source_path"]).read_bytes())
    if sha(source.raw) != document["source_sha256"]:
        raise CSWPError("source hash drift")
    segments = build_ledger(children, blocks, source.text)
    units = []
    for kind, group in core_groups(segments):
        cores = [group] if kind == "barrier" else split_packable(
            document["title"], group, source.text, tokenizer
        )
        for core in cores:
            core_start, core_end = core[0]["start"], core[-1]["end"]
            if core_start >= core_end:
                continue
            retrieval_start = (
                left_overlap_start(
                    document["title"],
                    segments,
                    core_start,
                    core_end,
                    source.text,
                    tokenizer,
                )
                if kind == "packable"
                else core_start
            )
            units.append(
                make_unit(
                    core,
                    segments,
                    document,
                    source,
                    tokenizer,
                    contract,
                    retrieval_start,
                    core_start,
                    core_end,
                )
            )
    return units, segments, source


def validate_corpus(manifest_b, units, tokenizer, contract, root=ROOT):
    children = {child["chunk_id"]: child for child in manifest_b["children"]}
    blocks = {block["block_id"]: block for block in manifest_b["blocks"]}
    by_doc = {}
    for child in manifest_b["children"]:
        by_doc.setdefault(child["document_version"], []).append(child)
    if len({unit["chunk_id"] for unit in units}) != len(units):
        raise CSWPError("duplicate logical IDs")
    for document in manifest_b["documents"]:
        doc_units = [unit for unit in units if unit["document_version"] == document["document_version"]]
        source = Source((root / document["source_path"]).read_bytes())
        rebuilt, _, _ = pack_document(
            document,
            by_doc[document["document_version"]],
            blocks,
            tokenizer,
            contract,
            root,
        )
        if rebuilt != doc_units:
            raise CSWPError("serialization/identity/provenance mismatch")
        coverage = [0] * len(source.text)
        overlap = [0] * len(source.text)
        for unit in doc_units:
            for index in range(unit["core_char_start"], unit["core_char_end"]):
                coverage[index] += 1
            for index in range(unit["retrieval_char_start"], unit["core_char_start"]):
                overlap[index] += 1
            if unit["embedding_content_token_count"] > 254:
                raise CSWPError("full serialization overflow")
            extracted = source.text[unit["core_char_start"] : unit["core_char_end"]]
            if extracted != unit["canonical_content"]:
                raise CSWPError("source content mismatch")
            for span in unit["source_spans"]:
                expected = source.span(
                    span["source_char_start"], span["source_char_end"], span["role"]
                )
                if span != expected:
                    raise CSWPError("byte/character/line provenance mismatch")
        if any(count != 1 for count in coverage):
            raise CSWPError("source coverage gap or unlabelled overlap")
        for index, count in enumerate(overlap):
            if count and coverage[index] != 1:
                raise CSWPError("overlap without a unique core owner")
    flattened = [
        cid
        for unit in units
        for cid in unit["constituent_child_ids"]
        if cid in children
    ]
    original = [child["chunk_id"] for child in manifest_b["children"]]
    if sorted(flattened) != sorted(set(flattened)):
        raise CSWPError("child membership duplication")
    if set(flattened) != set(original):
        raise CSWPError("child membership loss")


def pack(manifest_b, tokenizer, contract, root=ROOT):
    blocks = {block["block_id"]: block for block in manifest_b["blocks"]}
    by_doc = {}
    for child in manifest_b["children"]:
        by_doc.setdefault(child["document_version"], []).append(child)
    units = []
    for document in manifest_b["documents"]:
        packed, _, _ = pack_document(
            document,
            by_doc[document["document_version"]],
            blocks,
            tokenizer,
            contract,
            root,
        )
        units.extend(packed)
    validate_corpus(manifest_b, units, tokenizer, contract, root)
    manifest = {
        "candidate": CANDIDATE,
        "schema_version": "phase5a2e_representation_v1",
        "packing_version": contract["packing_version"],
        "contract_sha256": sha256_json(contract),
        "frozen_B_fingerprint": sha(canonical_json(manifest_b)),
        "children": units,
    }
    counts = sorted(unit["embedding_content_token_count"] for unit in units)
    report = {
        "retrieval_chunks": len(units),
        "token_distribution": {
            "min": min(counts),
            "median": statistics.median(counts),
            "p95": counts[math.ceil(0.95 * len(counts)) - 1],
            "max": max(counts),
        },
        "children_above_254": sum(count > 254 for count in counts),
        "source_coverage_gaps": 0,
        "unlabelled_overlaps": 0,
        "provenance_failures": 0,
        "duplicate_logical_ids": 0,
        "left_overlap_units": sum(
            unit["retrieval_char_start"] < unit["core_char_start"] for unit in units
        ),
        "heading_h1_units": sum(
            any(
                segment["kind"] == "heading" and segment.get("heading_level") == 1
                for segment in unit["structural_segments"]
            )
            for unit in units
        ),
        "deterministic_fingerprint": sha256_json(manifest),
    }
    return manifest, report

