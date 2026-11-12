# Phase 5A2B: same-parent structural packing ablation

**Evaluation complete; B-PACKED must not advance.** The frozen packing variant reduced vector count by 10% but did not recover evidence retrieval or resolve crowding. This is a valid negative result, not a structural failure. No tuning followed the results.

## Frozen scope and integrity

Required starting HEAD and commit parent: `a37b402227c8095cca2f3558c3cf60bbc52e2813`. Branch: `feature/phase5-rag-hardening`. Initial worktree clean; parent chain verified. The exact 5A2 checkpoint was pushed before implementation. The experiment commit remains local; no merge or PR.

The preregistered rule greedily packs consecutive paragraph/list children from the same document and structural parent, includes intervening whitespace verbatim, and checks full title/breadcrumb/content serialization against 254 content / 256 total MiniLM tokens. Headings, code, tables and other protected types remain singleton retrieval units. No C expansion, model, ranker, query, labels, or metric changes.

`experiment_manifest.json` preserves the original pre-retrieval contract and code hashes; `execution_manifest.json` binds the single recorded historical-control-check correction. The initial attempt stopped because native float scores differed from historical values. A-only diagnosis found 929 differences, maximum 4.8e-7, with identical ranks/metrics. `control_check_correction.json` records the correction and matching AST hashes of all packing/comparison functions. Historical A/B ranks, all metrics and diagnostics match exactly. Historical native-score drift is reported, not rounded away. Both accepted new runs have exactly equal full results and embedding hashes.

Provider calls: **0**. External network calls during deterministic evaluation: **0**. Tracing disabled and socket/DNS attempts fail closed. Git checkpoint verification/push occurred outside evaluation. Full-suite browser tests use local loopback.

## Representation

| Measure | B-PACKED |
|---|---:|
| Retrieval chunks | 459 |
| Tokens min / median / p95 / max | 7 / 31 / 209 / 254 |
| Packing groups with more than one B child | 28 |
| Average B children per output unit | 1.11111111 |
| Above 254 / provenance failures / duplicate IDs / boundary violations | 0 / 0 / 0 / 0 |

Fingerprint (both builds): `017bcb283683943f30fb18849344ea3b0eac7f3327a22ff306cef5f8d0fd645f`.
B fingerprint remains `a9402fea0c4fbbae6613346370c0759b0239d3d870a386526980e1c40815e852`. A remains 73 vectors, B remains 510. All 510 B children occur exactly once in order across 459 B-PACKED units. There are 431 singleton outputs and 28 merged groups. The 230 original heading children remain singleton units; this explains why the permitted packing rule can only modestly reduce count, without establishing heading removal as a validated fix.

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

MRR@10: A = 0.96969697; B = 0.95791246; B-PACKED = 0.95887446.

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

MRR@10: A = 0.91161616; B = 0.87247475; B-PACKED = 0.88762626.

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

MRR@10: A = 0.97979798; B = 0.96969697; B-PACKED = 0.96969697.

## Every gating query

Classification reuses the frozen hybrid dimensions: evidence Recall@5, source Recall@5, MRR@10 and nDCG@5. Any decrease takes precedence over increases. Channel-specific classifications and all metric deltas are in `per_query_comparison.json`; exact covered/missing source spans and top-five IDs are in `evidence_analysis.json`.

| Query | vs A | vs B | Evidence R@5 A / B / B-PACKED |
|---|---|---|---|
| Q01 | REGRESSED | SAME | 1.00 / 0.00 / 0.00 |
| Q02 | REGRESSED | SAME | 1.00 / 0.00 / 0.00 |
| Q03 | REGRESSED | SAME | 1.00 / 0.00 / 0.00 |
| Q04 | REGRESSED | REGRESSED | 1.00 / 1.00 / 0.50 |
| Q05 | SAME | SAME | 1.00 / 1.00 / 1.00 |
| Q06 | SAME | SAME | 1.00 / 1.00 / 1.00 |
| Q07 | REGRESSED | SAME | 1.00 / 0.00 / 0.00 |
| Q08 | REGRESSED | SAME | 1.00 / 0.50 / 0.50 |
| Q09 | REGRESSED | SAME | 1.00 / 0.00 / 0.00 |
| Q10 | SAME | SAME | 1.00 / 1.00 / 1.00 |
| Q11 | REGRESSED | REGRESSED | 1.00 / 1.00 / 1.00 |
| Q12 | SAME | IMPROVED | 0.50 / 0.50 / 0.50 |
| Q13 | REGRESSED | SAME | 1.00 / 0.00 / 0.00 |
| Q14 | REGRESSED | SAME | 1.00 / 0.00 / 0.00 |
| Q15 | IMPROVED | IMPROVED | 0.50 / 0.50 / 1.00 |
| Q16 | REGRESSED | SAME | 1.00 / 0.00 / 0.00 |
| Q17 | REGRESSED | REGRESSED | 1.00 / 0.50 / 0.00 |
| Q18 | SAME | SAME | 1.00 / 1.00 / 1.00 |
| Q19 | REGRESSED | SAME | 0.50 / 0.00 / 0.00 |
| Q20 | REGRESSED | SAME | 1.00 / 0.50 / 0.50 |
| Q21 | REGRESSED | SAME | 0.50 / 0.00 / 0.00 |
| Q22 | REGRESSED | SAME | 1.00 / 0.50 / 0.50 |
| Q23 | REGRESSED | SAME | 1.00 / 0.00 / 0.00 |
| Q24 | SAME | SAME | 1.00 / 1.00 / 1.00 |
| Q27 | SAME | SAME | 1.00 / 1.00 / 1.00 |
| Q28 | SAME | SAME | 1.00 / 1.00 / 1.00 |
| Q29 | REGRESSED | SAME | 1.00 / 0.50 / 0.50 |
| Q30 | IMPROVED | SAME | 0.00 / 0.50 / 0.50 |
| Q31 | IMPROVED | SAME | 0.50 / 1.00 / 1.00 |
| Q32 | SAME | SAME | 1.00 / 1.00 / 1.00 |
| Q33 | REGRESSED | SAME | 1.00 / 1.00 / 1.00 |
| Q34 | IMPROVED | SAME | 0.00 / 0.50 / 0.50 |
| Q35 | SAME | SAME | 0.50 / 0.50 / 0.50 |

Against A: 4 improved, 10 same, 19 regressed. Against B: 2 improved, 28 same, 3 regressed. Q15 improves from 0.5 to 1.0 evidence R@5. Q19 and Q21 remain 0.0 versus A at 0.5; Q22 remains 0.5 versus A at 1.0. Existing Q07/Q23 losses remain. New losses against B: Q04 evidence 1.0 to 0.5; Q17 evidence 0.5 to 0.0; Q11 nDCG@5 0.79670758 to 0.68852888.

## Diversity and cost

Crowding = returned retrieval slots minus distinct sources, averaged over the same 33 gating queries. Values below are hybrid; all three channels and K=1/3/5/10 are in `results.json`.

| Arm | Top-5 distinct / repeated | Top-10 distinct / repeated | Vectors / multiplier | Vector bytes | BM25 serialized bytes |
|---|---|---|---|---:|---:|
| A | 2.9697 / 2.0303 | 5.5758 / 4.4242 | 73 / 1.0000x | 112,128 | 206,089 |
| B | 1.7879 / 3.2121 | 2.6364 / 7.3636 | 510 / 6.9863x | 783,360 | 235,955 |
| B-PACKED | 1.7576 / 3.2424 | 2.6667 / 7.3333 | 459 / 6.2877x | 705,024 | 230,937 |

| Arm | Dense mean ms | BM25 mean ms | RRF fusion mean ms | Document embedding ms |
|---|---:|---:|---:|---:|
| A | 0.048819 | 0.110447 | 0.019887 | 645.340 |
| B | 0.277912 | 0.451261 | 0.020319 | 1850.890 |
| B-PACKED | 0.249977 | 0.408709 | 0.019677 | 1973.751 |

Latency is the mean of two run means over 35 queries, on one local machine, using identical arm ordering. It is descriptive, not a production benchmark. Dense and BM25 times exclude query embedding; RRF time is fusion only. Index sizes describe exact float32 vectors and pickle-protocol-4 BM25 serialization, not resident memory or Qdrant storage. Retrieval payload bytes are separately reported in `results.json`.

B-PACKED dense latency is 5.12x A, BM25 3.70x A. Relative to B, dense is 10.1% lower and BM25 9.4% lower. Document embedding time increases despite fewer vectors. No quality benefit justifies these costs versus A.

## Validation and decision

Full offline suite: **1,335 passed**, 3 deprecation warnings. New packing/artifact suite: **34 passed**. Initial sandbox suite: 1 loopback permission failure and 6 browser startup errors; passed with required local permissions. Ruff passes on new code with default rules and on the repository with CI fatal rules E9/F63/F7/F82. `git diff --check` passes. A/B source code, artifacts, corpus and golden labels are unchanged. Two build fingerprints, full result hashes, per-arm hashes and embedding hashes match exactly. No new implementation P0/P1 was demonstrated. The quality gates fail independently of structural correctness.

The frozen advancement gate requires evidence R@5 at least A, at least +0.10 over B, and zero gating regressions versus A. Observed B-PACKED evidence R@5 is 0.50000000, -0.01515152 versus B, -0.34848485 versus A, with 19 regressions versus A. **Do not advance B-PACKED.**

Exact next recommendation: retain A as the experiment control (including its known truncation limitation), retain B and this failed packing variant unchanged, and hand this evidence to the retrieval-design owner for read-only adjudication of the remaining heading occupancy and exact-span failures. Stop this variant; do not retune it, implement C, or change the production index in this task. Any next representation hypothesis needs separate authorization and preregistration.

Confidence: high for this frozen corpus and development set; no claim of generalization, absence rejection, or production readiness.
