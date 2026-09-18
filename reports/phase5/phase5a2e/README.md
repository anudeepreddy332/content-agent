# Phase 5A2E: CSWP-v1 contiguous structural window packing

**Evaluation complete; CSWP-v1 must not advance.** Lossless H2–H6 contiguous packing recovered hybrid exact-span Recall@5 from B `0.51515152` / B-PACKED `0.50000000` to `0.78787879`, including Q07/Q14/Q23 matching A, but it remains below A `0.84848485` and hides six conservative gating regressions versus A. Beating B is not the promotion bar. No overlap or packing retuning followed the results.

## Frozen scope and integrity

Required starting HEAD and commit parent: `fa7eefbc0eeb6eb8570577a8e19497bfbd2a21b8`. Branch: `feature/phase5-rag-hardening`. Initial worktree clean; parent chain verified. The exact 5A2B checkpoint was pushed before implementation. The experiment commit remains local; no merge or PR.

CSWP-v1 is a new shadow representation. Frozen A, B, and B-PACKED manifests/fingerprints are unchanged. The packer builds a lossless source-segment ledger (prose, headings, lists, whitespace seams, code, tables, protected blocks, parser gaps), greedily packs source-adjacent paragraphs, complete lists, whitespace seams, and complete H2–H6 headings, treats H1/code/table/blockquote/HTML/thematic-break/source-gap as barriers, measures complete MiniLM serialization at 254/256 with `truncation=False`, serializes `title + LF + breadcrumb-excluding-in-slice-headings + LF + LF + verbatim contiguous source`, and prepends maximal labelled left overlap after each packable core. No semantic chunking, Candidate C, reranker, query rewrite, or embedding change.

Provider calls: **0**. External network calls during deterministic evaluation: **0**. Tracing disabled and socket/DNS attempts fail closed. Production retrieval and Qdrant were not read or written.

## Representation

| Measure | CSWP-v1 |
|---|---:|
| Retrieval chunks | 159 |
| Tokens min / median / p95 / max | 5 / 254 / 254 / 254 |
| Units at exactly 254 content tokens | 86 |
| Left-overlap units | 85 |
| H1 root units | 20 |
| Above 254 / coverage gaps / unlabelled overlaps / provenance failures | 0 / 0 / 0 / 0 |

Fingerprint (both builds): `439ccdf81e2aff1bc0bf3448734cb71f774bc8d3e7619dd1e4b51b2685b79bb3`.
B fingerprint remains `a9402fea0c4fbbae6613346370c0759b0239d3d870a386526980e1c40815e852`. A remains 73 vectors, B 510, B-PACKED 459. Q07/Q09/Q14/Q23 gold spans are coverable from CSWP cores without omitted seams; Q09 is split when the window exceeds 254, as required. Headings and whitespace seams are preserved verbatim. H1 and protected blocks remain barriers.

## Retrieval metrics

All tables below use the **33 gating queries**, excluding diagnostic/non-gating Q25/Q26. Recall is source recall under the unchanged metric. Precision counts relevant retrieval slots, so same-source repetition can inflate it. MRR is at 10. ABSENT metrics are unavailable (zero ABSENT labels). All-35 diagnostic aggregates and full native ranks are in `results.json`.

### dense

| Arm | K | Recall | Precision | nDCG | Exact evidence-span recall |
|---|---:|---:|---:|---:|---:|
| A | 1 | 0.89393939 | 0.93939394 | 0.93939394 | 0.51515152 |
| A | 3 | 0.95454545 | 0.76767677 | 0.95537806 | 0.77272727 |
| A | 5 | 0.98484848 | 0.57575758 | 0.96256675 | 0.80303030 |
| A | 10 | 1.00000000 | 0.34848485 | 0.96793763 | 0.93939394 |
| B | 1 | 0.89393939 | 0.93939394 | 0.91919192 | 0.19696970 |
| B | 3 | 0.92424242 | 0.85858586 | 0.91956737 | 0.31818182 |
| B | 5 | 0.93939394 | 0.86060606 | 0.92316171 | 0.45454545 |
| B | 10 | 0.98484848 | 0.83333333 | 0.94062964 | 0.63636364 |
| B-PACKED | 1 | 0.89393939 | 0.93939394 | 0.91919192 | 0.18181818 |
| B-PACKED | 3 | 0.92424242 | 0.88888889 | 0.91956737 | 0.37878788 |
| B-PACKED | 5 | 0.93939394 | 0.86666667 | 0.92316171 | 0.45454545 |
| B-PACKED | 10 | 0.98484848 | 0.83939394 | 0.94218123 | 0.66666667 |
| CSWP | 1 | 0.87878788 | 0.90909091 | 0.90909091 | 0.19696970 |
| CSWP | 3 | 0.93939394 | 0.87878788 | 0.92900589 | 0.62121212 |
| CSWP | 5 | 0.96969697 | 0.85454545 | 0.94072870 | 0.84848485 |
| CSWP | 10 | 0.98484848 | 0.61818182 | 0.94324104 | 0.90909091 |

MRR@10: A = 0.96969697; B = 0.95791246; B-PACKED = 0.95887446; CSWP = 0.94545455.

### bm25

| Arm | K | Recall | Precision | nDCG | Exact evidence-span recall |
|---|---:|---:|---:|---:|---:|
| A | 1 | 0.83333333 | 0.87878788 | 0.85858586 | 0.48484848 |
| A | 3 | 0.92424242 | 0.66666667 | 0.89189715 | 0.69696970 |
| A | 5 | 0.95454545 | 0.49090909 | 0.90494795 | 0.78787879 |
| A | 10 | 0.95454545 | 0.30303030 | 0.90494795 | 0.87878788 |
| B | 1 | 0.75757576 | 0.78787879 | 0.76767677 | 0.03030303 |
| B | 3 | 0.95454545 | 0.79797980 | 0.87762563 | 0.21212121 |
| B | 5 | 0.95454545 | 0.74545455 | 0.87762563 | 0.27272727 |
| B | 10 | 0.98484848 | 0.66363636 | 0.88718517 | 0.40909091 |
| B-PACKED | 1 | 0.78787879 | 0.81818182 | 0.79797980 | 0.04545455 |
| B-PACKED | 3 | 0.95454545 | 0.78787879 | 0.88880958 | 0.21212121 |
| B-PACKED | 5 | 0.95454545 | 0.72727273 | 0.88880958 | 0.30303030 |
| B-PACKED | 10 | 0.98484848 | 0.66363636 | 0.89836912 | 0.46969697 |
| CSWP | 1 | 0.83333333 | 0.87878788 | 0.85858586 | 0.24242424 |
| CSWP | 3 | 0.90909091 | 0.71717172 | 0.87828572 | 0.42424242 |
| CSWP | 5 | 0.92424242 | 0.65454545 | 0.88906875 | 0.54545455 |
| CSWP | 10 | 0.95454545 | 0.53939394 | 0.89916976 | 0.84848485 |

MRR@10: A = 0.91161616; B = 0.87247475; B-PACKED = 0.88762626; CSWP = 0.90836941.

### hybrid

| Arm | K | Recall | Precision | nDCG | Exact evidence-span recall |
|---|---:|---:|---:|---:|---:|
| A | 1 | 0.92424242 | 0.96969697 | 0.96969697 | 0.57575758 |
| A | 3 | 0.98484848 | 0.79797980 | 0.97094024 | 0.78787879 |
| A | 5 | 0.98484848 | 0.56969697 | 0.97094024 | 0.84848485 |
| A | 10 | 0.98484848 | 0.32727273 | 0.97094024 | 0.90909091 |
| B | 1 | 0.89393939 | 0.93939394 | 0.91919192 | 0.09090909 |
| B | 3 | 0.96969697 | 0.86868687 | 0.95448330 | 0.28787879 |
| B | 5 | 0.98484848 | 0.82424242 | 0.95807764 | 0.51515152 |
| B | 10 | 0.98484848 | 0.76363636 | 0.95807764 | 0.56060606 |
| B-PACKED | 1 | 0.89393939 | 0.93939394 | 0.91919192 | 0.09090909 |
| B-PACKED | 3 | 0.98484848 | 0.88888889 | 0.95537806 | 0.30303030 |
| B-PACKED | 5 | 0.98484848 | 0.83636364 | 0.95537806 | 0.50000000 |
| B-PACKED | 10 | 0.98484848 | 0.76969697 | 0.95537806 | 0.63636364 |
| CSWP | 1 | 0.89393939 | 0.93939394 | 0.93939394 | 0.28787879 |
| CSWP | 3 | 0.95454545 | 0.82828283 | 0.94172993 | 0.60606061 |
| CSWP | 5 | 0.98484848 | 0.79393939 | 0.95478073 | 0.78787879 |
| CSWP | 10 | 0.98484848 | 0.61212121 | 0.95478073 | 0.92424242 |

MRR@10: A = 0.97979798; B = 0.96969697; B-PACKED = 0.96969697; CSWP = 0.95707071.

## Per-query hybrid classification versus A

Gating IMPROVED: Q11, Q15, Q30, Q31.
Gating SAME: Q01, Q02, Q03, Q04, Q05, Q06, Q07, Q08, Q10, Q12, Q14, Q18, Q19, Q20, Q21, Q23, Q24, Q27, Q28, Q29, Q32, Q33, Q35.
Gating REGRESSED: Q09, Q13, Q16, Q17, Q22, Q34.

| Query | vs A | Evidence R@5 A / B / B-PACKED / CSWP |
|---|---|---|
| Q07 | SAME | 1.0 / 0.0 / 0.0 / 1.0 |
| Q09 | REGRESSED | 1.0 / 0.0 / 0.0 / 0.0 |
| Q14 | SAME | 1.0 / 0.0 / 0.0 / 1.0 |
| Q23 | SAME | 1.0 / 0.0 / 0.0 / 1.0 |
| Q01 | SAME | 1.0 / 0.0 / 0.0 / 1.0 |
| Q03 | SAME | 1.0 / 0.0 / 0.0 / 1.0 |
| Q08 | SAME | 1.0 / 0.5 / 0.5 / 1.0 |
| Q16 | REGRESSED | 1.0 / 0.0 / 0.0 / 0.0 |
| Q17 | REGRESSED | 1.0 / 0.5 / 0.0 / 0.0 |
| Q22 | REGRESSED | 1.0 / 0.5 / 0.5 / 0.5 |
| Q29 | SAME | 1.0 / 0.5 / 0.5 / 1.0 |
| Q02 | SAME | 1.0 / 0.0 / 0.0 / 1.0 |
| Q13 | REGRESSED | 1.0 / 0.0 / 0.0 / 0.0 |
| Q19 | SAME | 0.5 / 0.0 / 0.0 / 0.5 |
| Q20 | SAME | 1.0 / 0.5 / 0.5 / 1.0 |
| Q21 | SAME | 0.5 / 0.0 / 0.0 / 0.5 |

Q07/Q14/Q23 are recovered to A, not claimed as wins over A. Q02/Q19/Q20/Q21 match A and are not representation wins versus A; Q19/Q21 remain the known half-miss. Q13 remains a ranking failure and is worse than A. Q09 is representable and correctly split above 254, but hybrid top-5 still misses the span.

## Cost, diversity, crowding, latency

| Arm | Hybrid top-5 unique / repeated | Hybrid top-10 unique / repeated | Vectors / vs A | float32 bytes | BM25 pickle bytes |
|---|---:|---:|---:|---:|---:|
| A | 2.9697 / 2.0303 | 5.5758 / 4.4242 | 73 / 1.00x | 112,128 | 206,089 |
| B | 1.7879 / 3.2121 | 2.6364 / 7.3636 | 510 / 6.99x | 783,360 | 235,955 |
| B-PACKED | 1.7576 / 3.2424 | 2.6667 / 7.3333 | 459 / 6.29x | 705,024 | 230,937 |
| CSWP | 1.9394 / 3.0606 | 3.5152 / 6.4848 | 159 / 2.18x | 244,224 | 235,319 |

Crowding is no longer B's small-child occupancy: 20/159 units are ≤32 tokens (H1 roots) and 86/159 sit at the 254 ceiling. Remaining crowding is same-document large-window occupancy. Top-5 source diversity recovers toward A but does not reach it.

Latency is the mean of two run means over 35 queries, on one local machine, using identical arm ordering. It is descriptive, not a production benchmark.

| Arm | Dense mean ms | BM25 mean ms | RRF fusion mean ms | Document embedding ms |
|---|---:|---:|---:|---:|
| A | 0.065902 | 0.151030 | 0.025851 | 533.020 |
| B | 0.296072 | 0.481522 | 0.021542 | 1334.055 |
| B-PACKED | 0.254811 | 0.418855 | 0.020589 | 1430.133 |
| CSWP | 0.090571 | 0.184026 | 0.019425 | 874.042 |

CSWP dense latency is 1.37x A, BM25 1.22x A. Relative to B, both channels are substantially cheaper because the vector count fell from 510 to 159.

## Validation and decision

Two accepted retrieval repetitions have identical full results, packing fingerprints, per-arm hashes, and embedding hashes (`ad8203ddbbd95fc6f46c0566eef5ab5f83e451350cb27f31704d894df866cfd7`). Historical A/B rankings and metrics match 5A2 exactly; B-PACKED aggregate metrics match 5A2B exactly. Native-score historical-control differences: 0. Q07/Q09/Q14/Q23 coverability, lossless ledger, H1/protected barriers, tokenizer-limit, provenance, and determinism tests pass. Existing A/B/B-PACKED regression tests remain green. Retrieval-golden-v2 validator unchanged.

The frozen advancement gate requires material exact-evidence recovery toward or exceeding A, zero critical gating regressions versus A, zero silent truncation, zero provenance gaps, and deterministic IDs. Observed CSWP evidence R@5 is 0.78787879 versus A 0.84848485, with six conservative gating regressions versus A. **Do not advance CSWP-v1.**

Exact next recommendation: retain A as the quality reference, retain frozen B/B-PACKED/CSWP-v1 unchanged, and adjudicate the remaining split-window misses (Q09/Q16/Q17/Q22) and ranking-only failures (Q13/Q19/Q21) read-only. Do not retune overlap, implement Candidate C, or change the production index in this task.

Confidence: high for this frozen corpus and development set; no claim of generalization, absence rejection, or production readiness.
