# Phase 5A2J — BGE second-reranker experiment

**Evaluation complete; do not advance the reranker.** BGE exact evidence-span Recall@5 on 33 gating queries improves from CSWP control 0.78787879 to 0.89393939, beating MS-MARCO 0.86363636 and historical A 0.84848485. The advancement contract still fails: Q13 and Q22 remain below A; Q19/Q30 remain candidate-generation misses.

## Frozen identity

Required parent: `d684cb3ade29ff29905cfbc00d56233bc85dd16a`. Dependency checkpoint pushed to `origin/feature/phase5-rag-hardening` at the same SHA before evaluation. Model: `BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70`. Candidate pools reuse Phase-5A2H fingerprint `df00e312…` unchanged.

## Three-arm comparison (33 gating queries)

| Arm | Evidence R@5 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|
| CSWP control | 0.78787879 | 0.95707071 | 0.95478073 |
| MS-MARCO reranker | 0.86363636 | 1.00000000 | 0.97774595 |
| BGE reranker | 0.89393939 | 0.96969697 | 0.95443376 |
| Historical A | 0.84848485 | 0.97979798 | 0.97094024 |

BGE restores Q20 (MS-MARCO regressed it). Q13 remains 0.0 exact evidence at top-5. Q22 remains 0.5 (gradient/Hessian span not in top-5). Q09/Q16 recover vs control; Q17 recovers vs control.

## Resources

- Parameters: 278,044,417 (~1.04 GB weights)
- Latency p50/p95 (35 queries): 2925 ms / 3713 ms per query pass
- Peak RSS: ~2.2 GB process peak
- MS-MARCO reference latency (5A2H): ~598 ms / ~787 ms p50/p95

Two offline runs produced byte-identical `results.json` fingerprints.
