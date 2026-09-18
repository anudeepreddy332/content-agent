# Phase 5A2H — frozen cross-encoder reranker experiment

**Evaluation complete; do not advance the reranker.** Exact evidence-span Recall@5 improves from CSWP control 0.78787879 to 0.86363636, exceeding A 0.84848485. This aggregate gain does not pass the advancement rule: Q20 loses critical evidence that both controls retrieved, while Q13 and Q22 remain below A. No result-driven tuning or substitute model was run.

## Scope and frozen identity

Required parent: `2a87e8e3e756872667c4468d4a52c34ea8917416`. The initial checkout was clean on `feature/phase5-rag-hardening`; HEAD and the local origin branch reference matched. No network fetch was performed. Exactly one local experiment commit is authorized; no push, merge or PR.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`, already available in the local cache. The contract binds all six model/tokenizer file hashes and library versions. Weights SHA-256: `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`. The config retains a historical L-12 name string, but the pinned weights and config contain six encoder layers and a `[1, 384]` classifier. That discrepancy is recorded, not silently substituted.

Scoring uses raw single float32 relevance logits with Identity activation. CPU, one intra/inter-op thread, eager attention, MKLDNN disabled, seed 0, deterministic algorithms, eval/inference mode and fixed batches of eight are frozen. Query and exact CSWP retrieval text are passed separately as a pair. Every pair is checked without truncation before any experimental inference. Model/tokenizer capacity is 512 total tokens; the 1,046 evaluated pairs peak at 293, with zero truncation.

The exact archived dense-top-20 and BM25-top-20 lists are replayed and deduplicated by canonical chunk ID. This isolates ordering and generates no new embeddings. Candidate pools preserve native ranks, scores and source spans. H1 roots remain eligible. Cross-encoder score descending, then chunk ID ascending, is the entire final ordering rule; there is no dense/BM25/RRF score fusion, MMR, diversity adjustment, rewriting or Candidate C. Full pool scores are retained, top-5 is the evaluated delivery set, and top-10 supports the requested diagnostic metrics.

CSWP fingerprint: `439ccdf81e2aff1bc0bf3448734cb71f774bc8d3e7619dd1e4b51b2685b79bb3`. Source bytes, representation, query text, golden-v2, dense/BM25 candidates, embedding identity, RRF and metric code are unchanged and hash checked. The artifact dependency chain also verifies the existing baseline/B inputs.

## Comparison on 33 gating queries

Q25/Q26 remain diagnostic and non-gating. Evidence recall below is exact source-span recall, requiring complete union coverage, not topic/source presence. Source recall, Precision@K, MRR and graded nDCG reuse the existing evaluator unchanged. All-35 metrics and every query are in each run result.

| Arm | K | Evidence recall | Source recall | Precision | Graded nDCG |
|---|---:|---:|---:|---:|---:|
| control | 1 | 0.28787879 | 0.89393939 | 0.93939394 | 0.93939394 |
| control | 3 | 0.60606061 | 0.95454545 | 0.82828283 | 0.94172993 |
| control | 5 | 0.78787879 | 0.98484848 | 0.79393939 | 0.95478073 |
| control | 10 | 0.92424242 | 0.98484848 | 0.61212121 | 0.95478073 |
| reranked | 1 | 0.36363636 | 0.95454545 | 1.00000000 | 0.97979798 |
| reranked | 3 | 0.66666667 | 0.98484848 | 0.88888889 | 0.97774595 |
| reranked | 5 | 0.86363636 | 0.98484848 | 0.81818182 | 0.97774595 |
| reranked | 10 | 0.92424242 | 0.98484848 | 0.63333333 | 0.97774595 |
| A | 1 | 0.57575758 | 0.92424242 | 0.96969697 | 0.96969697 |
| A | 3 | 0.78787879 | 0.98484848 | 0.79797980 | 0.97094024 |
| A | 5 | 0.84848485 | 0.98484848 | 0.56969697 | 0.97094024 |
| A | 10 | 0.90909091 | 0.98484848 | 0.32727273 | 0.97094024 |

MRR@10: control 0.95707071; reranked 1.00000000; A 0.97979798.

Pool exact-evidence recall is 0.96969697 on the 33 gating queries (0.97142857 on all 35); source recall is 1.0. Pools contain 24–38 candidates, mean 29.8857 across all 35 (29.7576 gating). Q13’s full gold span is available in the pool despite being absent from the control top-10.

## Query-level outcome

Evidence classification uses top-5 covered-span sets: any lost span is REGRESSED, otherwise any gained span is IMPROVED, otherwise SAME. Span swaps cannot hide behind equal recall. Ranking classification is separate: any decrease among source recall, precision and nDCG at 1/3/5/10 or MRR@10 is REGRESSED; mixed increases/decreases are listed explicitly. This is conservative component reporting, not a composite quality score.

- EVIDENCE IMPROVED: Q09, Q16, Q17.
- EVIDENCE REGRESSED: Q20.
- EVIDENCE SAME: Q01, Q02, Q03, Q04, Q05, Q06, Q07, Q08, Q10, Q11, Q12, Q13, Q14, Q15, Q18, Q19, Q21, Q22, Q23, Q24, Q27, Q28, Q29, Q30, Q31, Q32, Q33, Q34, Q35.
- RANKING-METRIC IMPROVED: Q02, Q08, Q13, Q15, Q22, Q24, Q27, Q28, Q32, Q33, Q34.
- RANKING-METRIC REGRESSED: Q11, Q12, Q18, Q21, Q23, Q31.
- RANKING-METRIC SAME: Q01, Q03, Q04, Q05, Q06, Q07, Q09, Q10, Q14, Q16, Q17, Q19, Q20, Q29, Q30, Q35.
- ranking_only_regressions: Q11, Q12, Q18, Q21, Q23, Q31.

| Query | Control evidence R@5 | Reranked | A | Diagnosis |
|---|---:|---:|---:|---|
| Q09 | 0.0 | 1.0 | 1.0 | Recovered; full span enters by rank 4 (control rank 6). Relevant H1 moves 2 → 7. |
| Q13 | 0.0 | 0.0 | 1.0 | Not recovered; full required span completes only at rank 17. Topic/source metrics improve while evidence remains 0. |
| Q16 | 0.0 | 1.0 | 1.0 | Recovered; full span enters by rank 5 (control rank 6). Relevant H1 moves 1 → 4 and remains in top-5. |
| Q17 | 0.0 | 1.0 | 1.0 | Both spans recovered by rank 4 (control rank 7). |
| Q19 | 0.5 | 0.5 | 0.5 | Tail span absent from union pool; reranking cannot recover it. |
| Q20 | 1.0 | 0.5 | 1.0 | New critical regression: embedding definition span moves from rank 5 to 6. All top-5 results remain from the same document. |
| Q21 | 0.5 | 0.5 | 0.5 | Unresolved tail span remains at rank 10; candidate pool contains it. Precision@3/@5 also regress. |
| Q22 | 0.5 | 0.5 | 1.0 | Unresolved gradient/Hessian span completes at rank 9 (control rank 10); missing-value evidence remains retrieved. |
| Q30 | 0.5 | 0.5 | 0.0 | Linear-regression span absent from union pool; reranking cannot recover it. |
| Q34 | 1.0 | 1.0 | 0.0 | Evidence SAME; ranking metrics IMPROVED. These labels are deliberately separate. |

Q20 is the only new gating evidence regression against CSWP. Q13/Q22 are continuing critical deficits against A, not newly introduced losses versus CSWP. Q19/Q30 are candidate-generation misses and are not counted as reranker-caused failures. `evidence_rank_analysis.json` records each span’s first complete coverage rank and all intersecting pool candidates.

## H1 and same-document occupancy

Across gating queries, H1 top-5 occurrences change from 16 to 2; 14 control occurrences leave top-5. Q09 demonstrates natural root demotion; Q16 demonstrates that the model can demote a root while retaining it in the final five. No H1 was filtered.

| Arm | Top-5 distinct sources | Repeated source slots | Max same-document slots |
|---|---:|---:|---:|
| control | 1.9394 | 3.0606 | 3.9394 |
| reranked | 1.7879 | 3.2121 | 4.0606 |
| A | 2.9697 | 2.0303 | 2.8182 |

Relevance reranking does not demonstrate a diversity improvement: distinct-source count declines. This observation does not authorize adding MMR to this arm.

## Measured cost and repeatability

Pooled over 70 query observations from two fresh processes, p50 is 598.483 ms, p95 787.353 ms, mean 620.658 ms. Run 1 p50/p95: 602.312/787.353 ms; run 2: 594.654/810.247 ms. Timings include pair tokenization, batching, inference and sorting; model loading is separate (55.206/45.562 ms). There is no warmup, retry or parameter sweep. This is one-machine offline cost, not a production SLO.

The model has 22,713,601 parameters (90,854,404 float32 parameter bytes); the weights file is 90,870,598 bytes. Observed process peak RSS is 1,230,979,072 and 1,232,322,560 bytes, including Python, evaluator and ML libraries as well as the model. Candidate count ≤40, input ≤512, fixed model and batches of eight provide the execution bounds.

Full result fingerprint in both runs: `9490f45a57392d8a9399e006d01a8215d98e7a25d89406efa807f2729a61def5`. Result files are byte-identical, including unrounded float32 scores, all ranks, query-level metrics and the scorer source hash. Timing/resource observations are deliberately outside the deterministic fingerprint.

## Decision and next action

**Do not advance.** The material aggregate gain and token/determinism checks pass, but zero critical evidence regressions does not: Q20 is newly worse than CSWP/A, and Q13/Q22 remain worse than A. These are quality blockers, not proof of an implementation defect. No P0 implementation failure was demonstrated.

Keep A and all frozen arms unchanged. Independently review the frozen Q20 regression and Q13/Q21/Q22 remaining exact-evidence ranking misses before authorizing another intervention; treat Q19/Q30 as a separate candidate-generation issue. No promotion, retuning, MMR, new model, push or merge.

ABSENT/OOS quality is unmeasurable, Q25/Q26 remain non-gating, and all queries are development/diagnostic. No generalization or production-readiness claim is made.

## Reference discipline

- *An Illustrated Guide to AI Agents*, PDF pages 45–48: relevance reranking of a retrieved shortlist; MMR is a separate relevance/redundancy intervention. This experiment uses the former only.
- *Building Applications with AI Agents*, PDF pages 44 and 284 (printed pages 22 and 262): assess cost/latency trade-offs and instrument control-versus-shadow differences. Its actor/critic reranking example is not treated as a passage-cross-encoder specification.
- *BASWE — 100 AI Engineering Interview Questions*, PDF pages 4–5, 15–16 and 36: broad cheap retrieval followed by pairwise cross-encoder scoring; retrieval-layer metrics; frozen single-variable comparisons and query-level regression reporting. Its example candidate counts/latencies are not imported as this experiment’s policy or measured result.

All references were read locally, without downloads or provider calls. The supplied task and frozen repository contracts govern execution; book examples do not authorize additional interventions.

## Reproduction and evidence

`scripts/phase5a2h_rerank.py freeze` refuses to replace the checked-in contract. For a separately authorized reproduction, run `run --output reports/phase5/phase5a2h/<new-directory>` using the pinned existing local snapshot; the runner refuses missing or changed model files, input/runtime drift, socket/DNS operations, oversize pairs and existing result overwrite. It does not download or substitute a model.

`candidate_pools.json` is the frozen pair/provenance ledger; `run-1/results.json` and `run-2/results.json` retain all scores and metrics; runtime files retain timing and resource observations. `evaluation_summary.json`, `evidence_rank_analysis.json`, and `validation_report.json` provide the decision, exact-span diagnosis and test evidence.

Provider calls: 0. External network during evaluation: 0. Production Qdrant reads/writes: 0. New embeddings: 0. LangSmith disabled. The complete regression suite separately permits its existing local browser/loopback harnesses while blocking external Python socket events.

## Validation result

138 focused Phase-5 tests and 1,375 full-suite tests passed (three existing deprecation warnings). Ruff fatal-tier, all checks on the two new Python files, and diff checks passed. Golden-v2 validates 35 queries and 54 evidence spans with zero source/quote mismatches. The full suite used 417 local harness socket/DNS events and zero audited external Python socket events; experimental runs forbid every socket/DNS operation.

`PHASE-5A2H-RERANKER-EVALUATION-READY` means that this completed negative experiment is ready for review. It does **not** mean the reranker is ready to advance.
