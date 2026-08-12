# Exact-73 Retrieval-Channel Ablation — Hardened Evidence Report

Diagnostic only. No production cutover. No architecture change in this report.

## Provenance

| Field | Value |
| --- | --- |
| Logical base SHA | `794851dded770ce87d111e73735d000e23597eb1` |
| Evidence-hardening commit | see branch tip after this report lands |
| Branch | `experiment/exact73-retrieval-channel-ablation` |
| Fixture fingerprint | `4a1d5d1d67b56867c71497cb58ed4964d356a122a14d47ef822c227dba5924e4` |
| Chunks / sources | 73 / 20 |
| In-domain queries | 30 |
| Dense depth / BM25 depth / fused depth | 10 / 10 / 5 |
| RRF k | 60 |
| Local result artifact SHA-256 | `5c5a9cf1bf6ccafef3f028b01f216e434079a89751eb2cdffa4ef7ece78e4207` |
| Local artifact path (gitignored) | `outputs/exact73_channel_ablation/result.json` |

## Jina Feasibility Gate Correction

`JINA-LOCAL-FEASIBILITY-FAIL: ~6 GB peak RSS > frozen 4 GB gate`

Jina remains forensic-only. Not a deployable candidate. Ablation pins Jina inference to CPU for deterministic co-residence with MiniLM/GTE; standalone CPU/MPS rankings match the recorded `EXACT73-JINA-FAIL`.

## Seven-Arm Aggregate Metrics

| Arm | @3 hit | @3 recall | @3 nDCG | @3 cov | @3 pass | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25-only | 0.933 | 0.900 | 0.905 | 0.731 | 0.700 | 0.950 |
| MiniLM dense | 1.000 | 0.917 | 0.910 | 0.817 | 0.800 | 0.961 |
| GTE dense | 1.000 | 0.950 | 0.936 | 0.753 | 0.700 | 0.961 |
| Jina dense | 0.133 | 0.117 | 0.091 | 0.000 | 0.000 | 0.139 |
| MiniLM+RRF | 1.000 | 0.950 | 0.948 | 0.822 | 0.833 | 0.983 |
| GTE+RRF | 0.967 | 0.933 | 0.926 | 0.797 | 0.800 | 0.958 |
| Jina+RRF | 0.900 | 0.850 | 0.677 | 0.533 | 0.500 | 0.647 |

Fused arms reproduce prior exact-73 recorded metrics within ε=1e-6.

## Ranking Fingerprints (all seven arms)

| Arm | SHA-256 |
| --- | --- |
| BM25-only | `e4d4296837c2e95de6f6e5c195ad4beaf74294e88b031f84250c23bc3617f94e` |
| MiniLM dense | `52896afea168dadb48734842898a70eb086e914f32b0ab97480cbc20ba987b7b` |
| GTE dense | `bd1b7226cb5cbd61fdaf5444677221a5a35d0d80bbc72794d600a8c0487f48b0` |
| Jina dense | `1a83dd8938c5841de1349a363231e500546e64f0f45d75e56d37545cb87517b2` |
| MiniLM RRF | `b46423192380062dc0bc79cae26bf71783025b3e77dc5b2466448234fed684e7` |
| GTE RRF | `0a38e6d675e3c53123531907d290b598c4a9c7b3037f0851f7b63c18f4498784` |
| Jina RRF | `6e8fa179016b4bbb246bb609cd8bd288b3bcf60cf8d3ffe5807221099097c3e6` |

Repeatability (two consecutive full ablation runs):

`rankings_and_metrics_identical=true`

(Terminology is precise: all seven ranking fingerprints and all arm metrics matched. Full result JSON is not asserted as byte-identical across runs because runtime provenance paths such as temporary Jina package directories differ.)

## Chunk vs Source Alignment (BM25 ↔ dense)

Hypothesis under test (diagnostic only): GTE may retrieve the correct source but a different chunk than BM25, so chunk-identity RRF fails to reinforce.

| Model | depth | mean chunk overlap | mean source overlap | same relevant source, different chunk (queries) | same relevant chunk both channels (queries) | expected-source chunk reinforcement total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM | @3 | 1.800 | 1.200 | **0** | 28 | 50 |
| GTE | @3 | 1.767 | 1.133 | **3** | 27 | 50 |
| Jina | @3 | 0.067 | 0.267 | 2 | 1 | 1 |
| MiniLM | @5 | 2.700 | 1.867 | 3 | 29 | 70 |
| GTE | @5 | 2.867 | 1.833 | 1 | 29 | 74 |
| Jina | @5 | 0.200 | 0.967 | 6 | 3 | 3 |
| MiniLM | top-10 | 5.167 | 3.267 | 1 | 30 | 95 |
| GTE | top-10 | 5.333 | 3.133 | 1 | 29 | 97 |
| Jina | top-10 | 0.867 | 3.167 | 5 | 10 | 11 |

Interpretation: GTE has modestly more same-source/different-chunk cases at @3 (3 vs 0), but overall chunk-reinforcement totals are comparable to MiniLM (or higher at @5/top-10). Chunk-identity RRF misalignment is **not** a sufficient primary explanation of GTE fusion underperformance versus MiniLM. Jina shows near-zero chunk reinforcement.

## Dense-Induced Fusion Harm

A = `bm25_expected_displaced`  
B = `dense_induced_destructive_fusion` (A plus a non-expected dense identity present in fused top-k)

| Model | @1 A / B | @3 A / B | @5 A / B |
| --- | --- | --- | --- |
| MiniLM | 1 / **0** | 0 / **0** | 0 / **0** |
| GTE | 2 / **0** | 0 / **0** | 0 / **0** |
| Jina | 15 / **4** | 3 / **3** | 2 / **2** |

Earlier broad `destructive_fusion` counts overstated MiniLM/GTE harm. Under the tightened B criterion, only Jina shows clear dense-induced displacement.

## Early-Prefix Status (Question 6)

Status: **`UNSUPPORTED`**

Correlational annotation only:

- vs GTE: `CORRELATIONAL_PREFIX_PRESENCE_ONLY` (1 MiniLM-only concept-pass@3 query; prefix concepts present)
- vs Jina: `UNSUPPORTED`

Prefix concept presence does **not** establish beneficial MiniLM truncation causality.

## Classification Snapshot

Hardened diagnosis remains mixed:

- MiniLM+BM25+RRF still strongest fused configuration (Case D)
- GTE dense is competitive; fusion slightly drags @3 hit (Case C partial); chunk-alignment hypothesis is weak/partial
- Jina dense unsuitable; dense-induced fusion harm confirmed under tightened B criterion

Ready for architecture review. No threshold moves. No fusion redesign in this task.
