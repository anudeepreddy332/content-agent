# Phase 5A1 — Candidate B shadow representation

Candidate B is an offline Markdown representation build. It does not import or
call the production ingestion, embedding, Qdrant, retrieval, BM25 or fusion
paths. The required starting commit is
`260dfd7450de30f4f5aa333c4fcd390d3eff8b7a` on
`feature/phase5-rag-hardening`. Baseline A and the 5A0 contracts remain unchanged.

## Parser and canonical source

`scripts/phase5a1_shadow.py` uses the frozen `markdown-it-py==4.2.0`, CommonMark
preset, HTML blocks enabled, and the built-in table rule enabled. Top-level
source maps produce ordered atomic blocks. Nested lists remain whole list
groups; their complete direct items are the legal subdivision units. Headings
form a level-aware section tree. Each block points to its nearest enclosing
heading; heading blocks themselves point to the preceding lower-level heading.
Sibling edges remain within one document version and structural parent.

Canonical text is sliced from the original UTF-8 source bytes, not the parser's
rendered text. Both character and byte offsets are zero-based and half-open;
line numbers are one-based and inclusive. CRLF and CR source line endings and
Unicode are preserved. Blank separators between blocks are not child content.
Non-whitespace source gaps, including reference definitions omitted from the
parser's token stream, become explicit `source_gap` blocks. Coverage validation
rejects lost non-whitespace source or any uncovered block content.

## Retrieval grammar frozen for this implementation

5A0 froze the three fields and their order, but did **not** specify literal
separators. This implementation freezes that previously unspecified detail as
`title-breadcrumb-exact-child-v1`:

```python
title + "\n" + " > ".join(heading_path) + "\n\n" + exact_child_content
```

The title is the first top-level H1's parser inline content, otherwise the source
filename stem. Breadcrumb labels use parser inline heading content and include
the current heading for heading blocks. No stripping or normalization is applied
to child content. The separators are always LF; original child line endings
remain unchanged. Empty breadcrumbs still leave their separators in place.
Hashes, IDs, timestamps, or generated descriptions are never added to retrieval
text. Metadata is separate. Future changes to this grammar require a new
serialization and chunker version.

## Structural subdivision

A fitting block remains one intact child, with no overlap. This version does
not combine adjacent blocks; tiny blocks retain their own identity. There is no
soft size target and no token size sweep.

Only oversized blocks are subdivided:

- Paragraphs use sentence-ending whitespace, then actual tokenizer offsets
  for an oversized sentence. Each emitted substring is re-tokenized in its
  complete retrieval serialization. There is no assumed monotonic token count,
  token decoding, partial UTF-8 decoding, or overlap.
- Lists use complete direct items including nested content and source markers.
  Every child retains the list's origin block ID.
- Fenced code uses complete source lines, repeating the exact opening and
  closing fence source spans. Fence language is retained in the opener. An
  oversized unclosed fence fails closed. Indented code uses complete lines.
- Tables use complete rows and repeat the exact header and delimiter lines.
- Other oversized atomic structures fail closed rather than acquire an
  unqualified splitting policy.

Repeated fences and table headers are separately marked context spans and
listed in `duplicated_context_spans`. The canonical child is the concatenation
of its exact ordered source slices, including these declared repetitions. An
oversized indivisible item, code line, table row, or metadata/context prefix
produces `CHUNK_OVERFLOW`, with the failed source spans and actual token count.
Other safe children may be recorded diagnostically, but any overflow makes the
whole corpus ineligible and the CLI exits nonzero after writing the artifacts.

## Tokenizer and identity

Only the frozen tokenizer assets are loaded. The three files under
`evals/fixtures/phase5a1_minilm_tokenizer/` were copied from the existing local
MiniLM snapshot and match 5A0's SHA-256 values. This makes the tests independent
of model downloads and machine cache contents. No model weights are loaded.

The actual `tokenizers` backend disables truncation and padding. The complete
retrieval text is measured both without and with special tokens. Each child
must have at most 254 content WordPieces and exactly two additional specials,
for at most 256 total tokens. Tests independently compare the backend counts
with `transformers.AutoTokenizer` loaded from the same frozen local assets.

IDs use 5A0's SHA-256 canonical JSON field lists (sorted keys, compact separators,
UTF-8, no random IDs). The document ID tracks tenant scope and normalized source
URI; document version tracks exact source bytes. Parser version changes block
and child IDs; chunker version changes child IDs. Document identity intentionally
remains stable across parser/chunker changes, as required by the frozen schema.
Duplicate text in different sections retains separate span-based identities.

Validation reconstructs source slices and hashes, re-computes IDs and full token
counts, verifies parent and sibling boundaries, and checks complete block
coverage by children or explicit overflow records. The manifest retains parser,
chunker, serialization, tokenizer revision/file hashes and runtime identity.

## Evidence and reproduction

The checked-in `reports/phase5/phase5a1/` directory contains one canonical
manifest, the corpus report, independent run reports, and validation evidence.
Two fresh CLI processes generated and compared the complete manifest bytes.
Redundant copies of the manifest are omitted after comparison; each run report
retains both the canonical-JSON fingerprint and the actual file SHA-256.

```sh
.venv/bin/python scripts/phase5a1_shadow.py --output-dir reports/phase5/phase5a1/run-1
.venv/bin/python scripts/phase5a1_shadow.py --output-dir reports/phase5/phase5a1/run-2
cmp reports/phase5/phase5a1/run-1/candidate_b_manifest.json reports/phase5/phase5a1/run-2/candidate_b_manifest.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LANGSMITH_TRACING=false LANGCHAIN_TRACING=false LANGCHAIN_TRACING_V2=false .venv/bin/python -m pytest tests/test_phase5a1_shadow.py tests/test_phase5a0_baseline.py tests/test_retrieval_golden_v2.py -q
```

The build installs a process-wide socket/DNS guard before optional imports and
allows no network connections, including loopback. LangSmith is disabled. The
existing regression suite separately uses local browser/loopback harnesses;
external Python socket events are forbidden during full-suite qualification.
The validation report distinguishes local harness traffic from external calls.

## Qualification boundary

The frozen 20-document corpus produces 498 blocks and 510 children. Content
WordPieces: minimum 7, median 46, nearest-rank p95 175, maximum 251. All overflow,
provenance, duplicate-ID and structural-boundary counts are zero. Synthetic
fixtures additionally exercise tables, Unicode/CRLF, exact 254/255 boundaries,
indivisible overflow, and oversized code/list/table splitting.

This qualifies representation readiness for 5A2, not retrieval quality or
production promotion. No embeddings, shadow indexes, retrieval A/B evaluation,
Candidate C expansion, semantic chunking, or production changes occur here.
The next independently authorized slice may embed B, build isolated immutable
shadow indexes and compare A/B under the existing frozen evaluation contract.
