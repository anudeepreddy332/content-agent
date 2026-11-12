# Phase 5A0 — Production RAG Hardening Baseline and Contract

_Frozen at starting HEAD `f269ab760fc78f0b3a65618ae0c744649d1a0a2e` on 2026-09-17._

## Disposition

**READY FOR PHASE 5A1 SHADOW IMPLEMENTATION.** Baseline A is reproducible from
checked-in source bytes without an operational Qdrant volume. The document,
provenance, identity, A/B/C, evaluation, promotion, and rollback contracts are
frozen in this document and in
`evals/fixtures/phase5a0_abc_contract.json`.

This disposition authorizes only the 5A1 shadow parser/provenance/identity and
structural-child implementation. It does not authorize a serving-index change,
production collection mutation, embedding-model replacement, provider call,
merge, push, or deployment.

## Frozen measured artifacts

- `reports/phase5/phase5a0/baseline_a_manifest.json`: ordered 20-source
  manifest, source/corpus hashes, exact 73-chunk identity, local model and
  tokenizer identity, ranker/fusion parameters, and current exposure limits.
- `reports/phase5/phase5a0/legacy_chunk_truncation.json`: per-chunk cl100k and
  WordPiece counts, tail positions, severity, and adjudicated evidence-tail
  intersections.
- `reports/phase5/phase5a0/baseline_a_report.json`: dense, BM25, and hybrid
  aggregate and per-query results, exact evidence-span recall, evidence
  exposure, channel wins, fusion harm, and failure classifications.
- `scripts/phase5a0_baseline.py`: deterministic source-only reconstruction and
  evaluator. It refuses the wrong starting HEAD and resolves only the frozen
  local MiniLM snapshot.

## Measured current architecture

| Stage | Implementation and status | Representation / identity | Truncation and failure behavior | Observability and coverage |
| --- | --- | --- | --- | --- |
| Source | `scripts/ingest.py`; live for `.md`/`.txt`. Declared Office/PDF/HTML route is dead because `tools.document_ingest` is absent. | UTF-8 file; filename stem becomes `source`. No version identity in production payload. | `read_text(...).strip()`; empty file skips. Non-text formats skip when the missing adapter cannot import. | Console counts only. Phase 5A0 freezes 20 exact source hashes and one corpus fingerprint. |
| Parser | Direct text read; live. No Markdown parser in production ingestion. | One stripped string; headings, code, lists and tables are not represented structurally. | No parser validation or protected-block policy. | No parser manifest or parser tests on the live route. Golden v2 validates exact source line spans offline. |
| Canonical text | The stripped source string is implicitly canonical; live. | No distinct canonical-content artifact or byte/character span map. | One legacy chunk ends on a partial UTF-8 token boundary and decodes non-exactly; the measurement records it instead of hiding it. | Phase 5A0 reconstructs exact source and chunk intervals. |
| Chunker | `tools/save_to_kb.py::_chunk_text`; live. | `cl100k_base`, 400 tokens, 50 overlap, stride 350; 73 chunks. Operational point UUID is random. | Splits without Markdown structure. No embedding-token assertion. | Existing tests cover retrieval metrics, not full chunk/provenance identity. New baseline tests freeze counts and fingerprints. |
| Embeddings | `SentenceTransformer("all-MiniLM-L6-v2")`; live, but production revision is not pinned in code. | 384-D normalized vectors. Local evaluated revision is `1110a243…`; historical production revision is `NOT_VERIFIABLE`. | Model accepts 256 total WordPieces: 254 content tokens plus two specials. Longer inputs silently truncate. 64/73 legacy chunks are exposed. | Live code emits no truncation telemetry. Phase 5A0 records every count/tail. |
| Qdrant | `machinist_evergreen`, intended Qdrant 1.9.2, 384-D cosine; live serving path. | UUID4 point ID; payload is text/source/chunk index plus basic filename/type. | Ingest exceptions return `False`; writes are batched. Re-ingestion can duplicate logical chunks. | Persisted local config was previously observed, but live/historical production identity is `NOT_VERIFIABLE`. Phase 5A0 did not read or write it. |
| BM25 corpus/index | `tools/query_kb.py::_build_bm25`; live, lazy, rebuilt from Qdrant payload text. | Same chunk text as dense, but no canonical child ID. Corpus order comes from Qdrant scroll. | Empty/missing collection yields no index. Import/error falls back to dense. | No index manifest. Phase 5A0 reproduces `lower().split()` plus `BM25Okapi` and preserves ranker-native scores. |
| Dense search | Qdrant cosine search; live. | Returns text, source, chunk index and converted distance; point ID is discarded. | Search error returns `None`, which collapses retrieval to empty rather than BM25-only. | Logger records errors but not a stable index/model/chunk identity. Phase 5A0 uses exhaustive cosine as the small-corpus reference. |
| BM25 search | `_bm25_query`; live. | Lowercase whitespace tokens; returns payload text/source/chunk index and optional native score. | Zero-score rows are omitted. Punctuation-heavy technical identifiers are not normalized. | Diagnostic API exposes top score. Phase 5A0 records punctuation cases and channel outcomes; no redesign is performed. |
| RRF | `_reciprocal_rank_fusion(k=60)`; live. | Uses raw chunk text as identity and zero-origin ranks. | A Python `set` supplies fusion membership, so equal-score ordering is not explicitly stable. Native dense/BM25 scores are correctly not averaged. | RRF score is logged, but component rank/ID lineage is not in the public result. Phase 5A0 uses stable `chunk_id` and a declared tie-break only in the evaluator. |
| Retrieve node | `query_kb(..., n_results=5)`; live. | Five fused KB result dictionaries stored in state. | Empty KB and search failures are tolerated; no typed failure state distinguishes no evidence from retrieval failure. | Retrieval latency and result count are logged; exact index identity and stage manifests are absent. |
| Context assembly / drafter | `_build_source_context(web_sources[:6], kb_results[:3])`; live. The helper then clips each KB text to 2,000 characters. | At most three ranked KB chunks, prefix only. | Evidence in ranks 4–5 is not shown to the drafter; long-chunk tails can be clipped. | Phase 5A0 measures drafter-exposed span recall separately and finds two lost-after-retrieval queries (Q07, Q23). |
| Verifier exposure | `build_evidence_manifest`; live Call-B path. | First five KB results, full stored text, source text hash and limited provenance. | No character clip, but no canonical source-span/index identity. | Verifier-visible content is traceable by evidence hash, not yet to exact source spans. Phase 5A0 finds zero additional loss from retrieved top-5 to verifier exposure. |

## Baseline A identity

The control is a clean reconstruction from the 20 ordered Markdown documents,
not a snapshot of UUID points:

- corpus fingerprint:
  `724a8c3486cb482eadbd592a93918a6e2954bebdbdfa12a39753556b1c3d7c02`;
- ordered legacy chunk fingerprint:
  `c3976d83cc478d5f4352e680b59aac98ed314c5bbeb9f9ffee56e0ea51fbecbb`;
- 73 chunks from `cl100k_base` 400/50 windows;
- MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- fast BERT WordPiece tokenizer, 254 content-token limit, 384 dimensions;
- exhaustive normalized cosine reference; dense candidate K=20;
- `BM25Okapi` defaults over `text.lower().split()`; BM25 candidate K=20;
- zero-origin RRF with `k=60`; final K=10;
- stable evaluator tie break by canonical `chunk_id` where production is not
  explicitly deterministic.

Exact/plain cosine is the experimental reference at 73 chunks. HNSW is not
tuned in Phase 5A0; any future ANN experiment must report recall against this
exact reference.

## Tokenization and silent truncation

The local runtime independently reproduces the known exposure:

- 64/73 chunks, or 87.6712%, exceed 254 content WordPieces;
- severity: 8 chunks lose at most 25% of content tokens and 56 lose between
  25% and 50%; none lose more than 50%;
- 24/35 queries have at least one adjudicated evidence span intersecting a
  truncated region of at least one overlapping legacy chunk;
- one legacy cl100k boundary produces a non-exact replacement character at a
  split UTF-8 boundary; it is retained as measured legacy behavior.

Tail intersection is an exposure diagnosis, not proof that truncation caused a
ranking miss: overlapping legacy chunks may carry a safe copy. The report marks
truncation as a `plausible-contributor-not-causally-isolated` unless an isolated
experiment proves causality.

The future invariant is measured after the complete retrieval serialization:

> actual embedding tokenizer count, with truncation disabled, must fit the
> model limit before encoding; otherwise emit durable `CHUNK_OVERFLOW`.

Zero truncation is necessary but is not by itself a quality win.

## Retrieval baseline metrics

The authoritative gate slice is 33 cases and excludes Q25/Q26. All 35 remain
reported diagnostically. Values below are macro averages over the gating slice.

| Channel | Source recall @1 / @3 / @5 / @10 | Precision @1 / @3 / @5 / @10 | MRR@10 | Graded nDCG @1 / @3 / @5 / @10 | Exact evidence-span recall @1 / @3 / @5 / @10 |
| --- | --- | --- | --- | --- | --- |
| Dense | 0.8939 / 0.9545 / 0.9848 / 1.0000 | 0.9394 / 0.7677 / 0.5758 / 0.3485 | 0.9697 | 0.9394 / 0.9554 / 0.9626 / 0.9679 | 0.5000 / 0.7576 / 0.7879 / 0.9242 |
| BM25 | 0.8333 / 0.9242 / 0.9545 / 0.9545 | 0.8788 / 0.6667 / 0.4909 / 0.3030 | 0.9116 | 0.8586 / 0.8919 / 0.9049 / 0.9049 | 0.4697 / 0.6818 / 0.7727 / 0.8636 |
| Hybrid RRF | 0.9242 / 0.9848 / 0.9848 / 0.9848 | 0.9697 / 0.7980 / 0.5697 / 0.3273 | 0.9798 | 0.9697 / 0.9709 / 0.9709 / 0.9709 | 0.5606 / 0.7727 / 0.8333 / 0.8939 |

`Precision@K` is relevant chunk slots divided by K; repeated relevant-source
chunks count as relevant slots. `Source recall@K` credits each relevant source
once. Graded nDCG credits only the first occurrence of a source and uses v2
grades 2/1. Evidence-span recall requires the union of returned exact source
intervals to cover the full adjudicated span.

There are no ABSENT metrics: v2 has zero genuine ABSENT queries. All results
are development/diagnostic, not holdout evidence.

## Demonstrated failure modes

Source-level success hides exact-evidence misses:

- At hybrid K=5, gating source recall is 0.9848 while exact evidence-span
  recall is 0.8333.
- The critical grade-2 span misses on gating queries are Q15, Q19, Q21, and
  Q22. Q25 and Q26 also miss grade-2 spans but remain non-gating diagnostics.
- Q30 retrieves only one of two grade-1 sources at K=5. Q12, Q31, Q34, and
  Q35 have additional grade-1 span misses.
- Dense-only diagnostic wins at the declared K=5 comparison occur on Q21,
  Q25, Q31, and Q34; BM25-only wins on Q11 and Q35; hybrid wins on Q07, Q12,
  Q23, and Q33.
- Fusion is worse than the best component, under the frozen lexicographic
  source-recall / span-recall / nDCG comparison, on Q11, Q21, Q25, Q31, Q34,
  and Q35. This is a measurement, not authorization to tune RRF.
- Q07 and Q23 have evidence in retrieved top-5 but lose it when the drafter is
  limited to top-3. Verifier exposure retains the retrieved top-5 evidence.
- BM25's `lower().split()` leaves `C++`, dotted identifiers, function names,
  paths, hyphenated terms, versions, and punctuation attached. V2 shows BM25
  channel misses but does not isolate tokenizer redesign as the causal remedy.

The per-query report separates parser/structure, chunk boundary, embedding
truncation, embedding semantic, lexical/BM25, candidate-K, fusion/ranking,
identity/dedup, evidence exposure, evaluator ambiguity, and unknown. A
truncation intersection is not upgraded from plausible to demonstrated cause
without an isolated candidate comparison. Reranking is never blamed for
evidence absent from its candidate pool.

## Canonical document and provenance schema

The machine-readable required fields live in the A/B/C contract. The governing
rules are:

1. Canonical content is exact parser-derived source content. Metadata and
   retrieval text are separate representations.
2. Markdown provenance includes source path, source SHA, heading path, exact
   byte and character spans, and start/end lines.
3. Blocks carry parser-derived parent and reading-order identity. Neighbor
   edges never cross a document version or structural parent.
4. Children carry ordered block IDs, exact source spans, chunker version, full
   retrieval-text hash, and actual embedding-token counts.
5. Future PDF provenance must retain page, reading-order block, bounding box,
   and extracted span. Display line numbers may be derived but are not source
   authority.
6. LLM-generated provenance is forbidden.

Filter-compatible payload fields include tenant, ACL principals, source,
document version, document type, content type, and deletion/current-version
state. Tenant and ACL enforcement are explicitly outside Phase 5, but future
hard correctness/security constraints must be pre-filters. Soft preferences
may be post-filters only with measured justification.

## Deterministic identity and immutable indexes

IDs are SHA-256 of canonical JSON inputs:

- `document_id`: tenant scope + normalized source URI;
- `document_version`: document ID + exact source SHA;
- `block_id`: document version + parser identity + structural path + block
  type + exact byte span + content hash;
- `chunk_id`: document version + parser/chunker identity + ordered blocks +
  exact spans + retrieval-text hash;
- `index_id`: ordered corpus digest + representation, embedding, dense, BM25,
  and fusion identities.

Same bytes and parser/chunker versions reproduce all derived IDs. A byte,
parser, chunker, serialization, tokenizer, model, or index-configuration change
creates a different version. Re-ingestion is idempotent. Random UUIDs are not
logical identity.

Every experimental index is immutable and digest-named. Phase 5A1 may produce
manifests only; Phase 5A2 may create a separate shadow collection. Neither may
overwrite `machinist_evergreen` or activate a serving alias. Rollback retains
the last qualified immutable index and requires a separately authorized alias
or configuration switch.

## Candidate A / B / C experiment

### A — legacy control

Use the exact frozen source order, 400/50 chunks, local MiniLM revision,
lowercase-whitespace BM25, candidate Ks, RRF policy, and exposure behavior in
the manifest. A never depends on UUIDs or an operational Qdrant volume.

### B — structure-aware, embedding-safe children

Parse Markdown with `markdown-it-py` into a heading tree and ordered atomic
blocks. Preserve paragraphs, list groups, fenced code, tables, blockquotes,
HTML blocks, headings, and thematic breaks. Pack only adjacent compatible
blocks within one parent. Recursive splitting is an oversized-block fallback,
not the primary parser.

There is no evidence-backed magic soft size in 5A0. The only frozen numeric
limit is the model's hard 254-content-token capacity, measured after serializing
`document title + heading breadcrumb + exact child`. Intact blocks have no
overlap. Forced prose subdivision may duplicate at most one deterministic
boundary sentence/paragraph and must record duplicate spans. Code and tables
never receive arbitrary token overlap. A smallest legal unit that still does
not fit becomes `CHUNK_OVERFLOW`.

B changes parsing, chunks, provenance, IDs, and deterministic structural
retrieval text only. The embedding model, exact dense reference, BM25
tokenization/parameters, candidate K, RRF, final K, and v2 evaluator remain
fixed.

### C — B retrieval plus deterministic context expansion

C reuses B's bit-identical child index and rank manifests. A child hit expands
only inside its nearest parser-derived heading parent. Prefer the complete
parent when it fits; otherwise use the seed plus at most one immediate sibling
on each side. Never cross document version, parent, tenant, or ACL boundary.

Order by seed rank, then source reading order, then chunk ID. Deduplicate by
chunk ID and exact source span. The primary C budget for each query/consumer is
exactly the actual-tokenizer token count of A's exposed evidence pack at the
same K. This matches capacity causally instead of inventing a universal budget.
No partial atomic unit is packed; budget exhaustion is recorded.

## Evaluation and promotion contract

Each arm emits separate dense, BM25, and fused results at retrieval, expanded,
packed, drafter-exposed, and verifier-exposed stages. Every row binds query,
index, chunk, rank, source spans, serialized/exposed hashes, consumer tokenizer,
and token count.

Every comparison reports all 35 diagnostic cases, the 33-case gating slice,
aggregate metrics, per-query deltas, fixed queries, regressions, and critical
misses. A critical regression cannot be hidden by an average gain. Retrieval,
evidence exposure, and end-to-end behavior remain independent layers; there is
no single quality score.

B/C cannot be promoted with silent truncation, provenance failure,
nondeterministic manifests/ranks, critical evidence regression, unexplained
per-query regression, materially worse exposure, or unbounded latency/storage.
Zero truncation, more chunks, or one improved aggregate is not sufficient.

## Conditional improvements

- A cross-encoder is eligible only when candidate top-N contains the gold
  evidence reliably but final top-K remains inadequate. A future arm must
  measure candidate recall, post-rerank recall, MRR, nDCG, exact evidence rank,
  p50/p95 latency, and compute/cost.
- Query rewrite, HyDE, decomposition, and step-back prompting remain out until
  failures demonstrate vocabulary mismatch, underspecification, or multi-part
  need. Raw-query results remain the comparator.
- BM25 tokenization changes require frozen lexical fixtures and causal evidence.
- Generated contextual descriptions, semantic chunking, embedding replacement,
  HNSW tuning, ColBERT, and GraphRAG remain separate future ablations.

## Phase 5 execution plan

1. **5A0 — complete:** baseline, schema, IDs, A/B/C, evaluation, promotion,
   rollback, and roadmap frozen.
2. **5A1 — next authorized slice:** implement the shadow Markdown parser,
   provenance, deterministic IDs, Candidate B manifests, tokenizer assertion,
   overflow states, protected code/table fixtures, and two-run fingerprint
   comparison. Stop before embeddings or Qdrant writes unless separately
   authorized.
3. **5A2:** build immutable shadow dense/BM25 indexes and measure A versus B.
4. **5B:** test C expansion and evidence packing on the selected B ranks.
5. **5C:** test only improvements triggered by measured failures.
6. **5D:** qualify one real heterogeneous format at a time; do not restore the
   archived multi-format adapter unchanged.
7. **5E:** final qualification with expanded independently adjudicated gold,
   genuine ABSENT cases, hard negatives, a hidden holdout, and shadow validation.

## Remaining qualification gaps

V2 is only a 35-case development/diagnostic nucleus. Final Phase 5 requires a
genuine ABSENT/OOS population, hard negatives, an untouched independently
adjudicated holdout, and growth toward 100–300 representative cases. Pilot and
production failures enter the permanent regression corpus only after
adjudication. Tenant/ACL enforcement, scale, latency, storage, recovery, and
operational cutover also remain unqualified.

No demonstrated 5A0 blocker remains. The operational production revision and
live collection identity remain `NOT_VERIFIABLE`, but Baseline A does not depend
on them and therefore stays reproducible.
