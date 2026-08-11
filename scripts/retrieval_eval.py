"""
Measures source-level retrieval quality and concept evidence coverage against a
fixed 35-query golden set, evaluated against whatever is
currently in the KB. To compare across corpus sizes, re-run this script after
each ingest stage and diff the JSON reports by hand — there is no built-in
loop or plotting; each invocation is a single point-in-time measurement.

Usage:
    uv run python scripts/retrieval_eval.py
    uv run python scripts/retrieval_eval.py --k 1 3 5 --output retrieval_report.json
"""

import sys
import json
import math
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.query_kb import query_kb, query_kb_context, query_kb_diagnostics
from tools.context_assembly import context_budget_stats


CONCEPT_PASS_THRESHOLD = 2 / 3


def reciprocal_rank(sources: list[str], expected_set: set[str], max_k: int) -> float:
    """Reciprocal rank of the first relevant result within the top `max_k`.

    A result at (1-indexed) rank i is relevant iff its source is in
    `expected_set`. Returns 1/rank of the first relevant hit, or 0.0 if none
    appears within `max_k`. Pure — no KB/network access.
    """
    for i, source in enumerate(sources[:max_k], start=1):
        if source in expected_set:
            return 1.0 / i
    return 0.0


def hit_at_k(sources: list[str], expected_set: set[str], k: int) -> float:
    """Return 1 when any expected source appears in the top-k, else 0."""
    return float(bool(expected_set.intersection(sources[:k])))


def source_recall_at_k(sources: list[str], expected_set: set[str], k: int) -> float:
    """Return the fraction of unique expected sources retrieved within the top-k."""
    if not expected_set:
        return 0.0
    return len(expected_set.intersection(sources[:k])) / len(expected_set)


def ndcg_at_k(sources: list[str], expected_set: set[str], k: int) -> float:
    """Normalized DCG@k with binary relevance keyed on source.

    DCG@k = Σ_{i=1..min(k, len(sources))} rel_i / log2(i+1), where only the
    first occurrence of an expected source receives relevance credit. This keeps
    source-level labels aligned with chunk-level retrieval while preserving the
    rank penalty for repeated sibling chunks.
    IDCG@k = Σ_{i=1..min(len(expected_set), k)} 1/log2(i+1) (the ideal ranking
    puts every relevant source first). Returns DCG/IDCG, or 0.0 when IDCG == 0.
    Pure — no KB/network access.
    """
    dcg = 0.0
    credited_sources: set[str] = set()
    for i, source in enumerate(sources[:k], start=1):
        if source in expected_set and source not in credited_sources:
            dcg += 1.0 / math.log2(i + 1)
            credited_sources.add(source)

    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return min(1.0, dcg / idcg) if idcg > 0 else 0.0


def concepts_in_text(text: str, required_concepts: list[str]) -> list[str]:
    """Return required concepts found by the historical deterministic substring rule."""
    lowered = text.lower()
    return [concept for concept in required_concepts if concept.lower() in lowered]


def concept_coverage_at_k(results: list[dict], required_concepts: list[str], k: int) -> tuple[list[str], float]:
    """Return found concepts and their fraction in the concatenated top-k evidence."""
    if not required_concepts:
        return [], 0.0
    found = concepts_in_text(" ".join(result.get("text", "") for result in results[:k]), required_concepts)
    return found, len(found) / len(required_concepts)


def source_diversity_at_k(sources: list[str], k: int) -> tuple[int, int]:
    """Return unique source count and duplicate source slots without compressing ranks."""
    slots = sources[:k]
    unique_sources = len(set(slots))
    return unique_sources, len(slots) - unique_sources


def summarize_context_budget(values: list[dict]) -> dict:
    """Return mean and maximum bounded-context diagnostics across queries."""
    fields = (
        "evidence_windows",
        "total_context_chars",
        "estimated_context_tokens",
        "unique_sources",
        "duplicate_chars_removed",
        "truncated_chars",
        "seed_budget_exceeded_windows",
        "seed_budget_overflow_chars",
    )
    return {
        f"mean_{field}": round(sum(value[field] for value in values) / len(values), 3)
        for field in fields
    } | {
        f"max_{field}": max(value[field] for value in values)
        for field in fields
    }

# Golden set - each query has ground truth: which source should appear in top-k
# and what concepts must be in the returned text.
# Contains 25 queries covering all 20 seed documents, with easy, medium, hard,
# adversarial, and multi-hop varieties.

GOLDEN_SET = [
    # Easy (direct topic match, exact terminology)
    {
        "query": "how gradient descent updates weights",
        "expected_sources": ["gradient-descent"],
        "required_concepts": ["learning rate", "loss", "gradient"],
        "difficulty": "easy",
    },
    {
        "query": "query key value in self-attention",
        "expected_sources": ["attention-transformers"],
        "required_concepts": ["query", "key", "value"],
        "difficulty": "easy",
    },
    {
        "query": "difference between lasso and ridge regression",
        "expected_sources": ["ridge-lasso-elasticnet"],
        "required_concepts": ["L1", "L2", "sparsity"],
        "difficulty": "easy",
    },
    {
        "query": "what is CatBoost categorical feature handling",
        "expected_sources": ["catboost"],
        "required_concepts": ["target statistics", "ordered boosting", "permutation"],
        "difficulty": "easy",
    },
    {
        "query": "ReAct agent pattern reasoning and acting",
        "expected_sources": ["react-agent-pattern"],
        "required_concepts": ["thought", "action", "observation"],
        "difficulty": "easy",
    },
    {
        "query": "k nearest neighbors algorithm basics",
        "expected_sources": ["k-nearest-neighbors"],
        "required_concepts": ["distance metric", "majority vote", "lazy learning"],
        "difficulty": "easy",
    },
    {
        "query": "XGBoost regularization parameters",
        "expected_sources": ["xgboost"],
        "required_concepts": ["gamma", "lambda", "leaf weight"],
        "difficulty": "easy",
    },
    {
        "query": "what is a feedforward neural network",
        "expected_sources": ["feedforward-neural-networks"],
        "required_concepts": ["hidden layer", "activation function", "backpropagation"],
        "difficulty": "easy",
    },
    {
        "query": "support vector machine margin maximization",
        "expected_sources": ["support-vector-machines"],
        "required_concepts": ["hyperplane", "support vectors", "kernel trick"],
        "difficulty": "easy",
    },
    {
        "query": "Naive Bayes conditional independence assumption",
        "expected_sources": ["naive-bayes"],
        "required_concepts": ["Bayes", "conditional independence", "posterior probability"],
        "difficulty": "easy",
    },

    # Medium (paraphrase, slight inference, combining two related docs)
    {
        "query": "vanishing gradient problem in deep networks",
        "expected_sources": ["backpropagation", "gradient-descent"],
        "required_concepts": ["gradient", "deep", "layer", "chain rule"],
        "difficulty": "medium",
    },
    {
        "query": "why random forests reduce overfitting compared to a single decision tree",
        "expected_sources": ["random-forest", "decision-trees"],
        "required_concepts": ["ensemble", "variance", "bootstrap", "feature subsampling"],
        "difficulty": "medium",
    },
    {
        "query": "how to deploy LLM agents with reliability and cost control",
        "expected_sources": ["agentic-ai-production"],
        "required_concepts": ["step budget", "cost bound", "observability", "guardrails"],
        "difficulty": "medium",
    },
    {
        "query": "logistic regression probability estimation formula",
        "expected_sources": ["linear-logistic-regression"],
        "required_concepts": ["sigmoid", "log-odds", "maximum likelihood"],
        "difficulty": "medium",
    },
    {
        "query": "LangGraph for cyclic agent workflows",
        "expected_sources": ["langgraph-state-machines"],
        "required_concepts": ["state machine", "conditional edges", "checkpointing", "human-in-the-loop"],
        "difficulty": "medium",
    },
    {
        "query": "prompt engineering best practices for production",
        "expected_sources": ["prompt-engineering-production"],
        "required_concepts": ["versioning", "monitoring", "A/B testing", "guardrails"],
        "difficulty": "medium",
    },
    {
        "query": "when to use multi-agent systems vs single agent",
        "expected_sources": ["multi-agent-systems"],
        "required_concepts": ["specialization", "coordination", "Dec-POMDP", "debate"],
        "difficulty": "medium",
    },
    {
        "query": "elastic net combining L1 and L2 penalties",
        "expected_sources": ["ridge-lasso-elasticnet"],
        "required_concepts": ["ElasticNet", "l1_ratio", "grouping effect"],
        "difficulty": "medium",
    },
    {
        "query": "how RAG reduces hallucination in LLMs",
        "expected_sources": ["retrieval-augmented-generation"],
        "required_concepts": ["non-parametric memory", "grounding", "retriever", "generator"],
        "difficulty": "medium",
    },
    {
        "query": "embedding vectors for semantic search",
        "expected_sources": ["embedding-models-vector-search"],
        "required_concepts": ["cosine similarity", "ANN index", "HNSW", "dense representation"],
        "difficulty": "medium",
    },

    # Hard (adversarial phrasing, indirect, or multi-hop across 2+ sources)
    {
        "query": "my neural net is deep but gradients become tiny — which algorithm computes them and how do I avoid this?",
        "expected_sources": ["backpropagation", "gradient-descent"],
        "required_concepts": ["chain rule", "vanishing gradient", "activation function", "ReLU"],
        "difficulty": "hard",
        "note": "Adversarial: conversational, no direct mention of backprop or gradient descent by name."
    },
    {
        "query": "tree ensemble that uses gradient and hessian and can handle missing values automatically",
        "expected_sources": ["xgboost"],
        "required_concepts": ["second-order approximation", "sparsity-aware", "default direction"],
        "difficulty": "hard",
        "note": "Adversarial: describes XGBoost without naming it."
    },
    {
        "query": "I want to classify text with many features but features are not independent — is there a fast probabilistic model that still works well?",
        "expected_sources": ["naive-bayes"],
        "required_concepts": ["conditional independence", "spam filtering", "multinomial", "despite correlation"],
        "difficulty": "hard",
        "note": "Adversarial: challenges the assumption but asks for Naive Bayes anyway."
    },
    {
        "query": "what happens to the decision boundary when you increase k in k-NN from 1 to a large number?",
        "expected_sources": ["k-nearest-neighbors"],
        "required_concepts": ["smoothing", "bias-variance tradeoff", "Voronoi", "majority vote"],
        "difficulty": "hard",
        "note": "Requires understanding of effect of hyperparameter, not just definition."
    },
    {
        "query": "compare the objective functions of ridge regression and support vector regression — how do they penalize coefficients differently?",
        "expected_sources": ["ridge-lasso-elasticnet", "support-vector-machines"],
        "required_concepts": ["L2 penalty", "ε-insensitive loss", "slack variables", "regularization"],
        "difficulty": "hard",
        "note": "Multi-hop: answer spans two distinct source documents (ridge and SVR)."
    },
    {
        "query": "I need to build a chatbot that can look up company documents, but also remember conversation state and loop to ask follow-up questions. Which two frameworks should I combine?",
        "expected_sources": ["retrieval-augmented-generation", "langgraph-state-machines"],
        "required_concepts": ["retrieval", "state machine", "checkpointing", "tool use"],
        "difficulty": "hard",
        "note": "Multi-hop: RAG for knowledge, LangGraph for conversational state and loops."
    },
    {
        "query": "how does CatBoost's ordered target encoding prevent leakage compared to regular target encoding?",
        "expected_sources": ["catboost"],
        "required_concepts": ["permutation", "lag", "smoothing", "overfitting"],
        "difficulty": "hard",
        "note": "Deep technical question requiring precise understanding of CatBoost internals."
    },
    {
        "query": "In the ReAct pattern, what is the risk if the model hallucinates an observation instead of executing a tool?",
        "expected_sources": ["react-agent-pattern"],
        "required_concepts": ["fabricated observations", "hallucination", "action parsing", "grounding"],
        "difficulty": "hard",
        "note": "Adversarial: focuses on failure mode, not the definition."
    },
    {
        "query": "design a production prompt for sentiment analysis that includes JSON output, version control, and injection protection — what are the key engineering practices?",
        "expected_sources": ["prompt-engineering-production"],
        "required_concepts": ["template sanitization", "schema validation", "CI/CD", "monitoring"],
        "difficulty": "hard",
        "note": "Applies prompt engineering to a concrete scenario."
    },
    {
        "query": "when would you choose a linear model over a random forest even if the random forest has higher accuracy?",
        "expected_sources": ["linear-logistic-regression", "random-forest"],
        "required_concepts": ["interpretability", "inference", "coefficients", "extrapolation"],
        "difficulty": "hard",
        "note": "Multi-hop: compares advantages of linear models vs random forests."
    },

    # Out-of-scope queries (genuinely not present in any seed document)
    {
        "query": "what is the BERT architecture and how does masking work?",
        "expected_sources": [],
        "required_concepts": [],
        "difficulty": "out-of-scope",
        "note": "BERT is not covered in any of the 20 seed documents. Retriever should return very low similarity scores or an empty result.",
        "min_distance_threshold": 0.5  # Cosine distance > 0.5 indicates no good match (assuming normalized embeddings)
    },
    {
        "query": "explain convolutional neural networks and pooling layers",
        "expected_sources": [],
        "required_concepts": [],
        "difficulty": "out-of-scope",
        "note": "CNNs are not in the seed docs. The documents cover feedforward NNs, attention, trees, regression, but not convolutional layers.",
        "min_distance_threshold": 0.5
    },
    {
        "query": "how does batch normalization stabilize training?",
        "expected_sources": [],
        "required_concepts": [],
        "difficulty": "out-of-scope",
        "note": "Batch norm is not mentioned in any provided document.",
        "min_distance_threshold": 0.5
    },
    {
        "query": "what is the difference between vanilla RNNs and LSTMs?",
        "expected_sources": [],
        "required_concepts": [],
        "difficulty": "out-of-scope",
        "note": "Recurrent networks and LSTMs are absent from the seed corpus.",
        "min_distance_threshold": 0.5
    },
    {
        "query": "explain the Adam optimizer and its hyperparameters",
        "expected_sources": [],
        "required_concepts": [],
        "difficulty": "out-of-scope",
        "note": "Adam is not covered; gradient descent is covered but not Adam specifically.",
        "min_distance_threshold": 0.5
    },
]

def run_retrieval_eval(
    k_values: list[int] = None,
    *,
    expanded_context: bool = False,
) -> dict:
    """Evaluate source ranking, concept evidence, diversity, and raw OOS signals.

    `recall@k` is retained as a historical alias for the old any-source metric;
    new comparisons must use `hit@k` and `source_recall@k` explicitly.
    """
    if k_values is None:
        k_values = [1, 3, 5]
    k_values = sorted(set(k_values))
    max_k = max(k_values)
    legacy_concept_k = 5 if 5 in k_values else max_k
    results = {
        "metric_semantics": {
            "recall@k": "legacy any-source hit@k; retained for historical comparison only",
            "out_of_scope_rejection_rate": "deprecated non-gating metric; fused distance is not calibrated",
            "retrieval_unit": "expanded evidence windows" if expanded_context else "raw retrieved child chunks",
        },
        "mrr": None,
        "concept_hit_rate": None,
        "out_of_scope_rejection_rate": None,
        "per_query": [],
        "total_queries": len(GOLDEN_SET),
    }
    for k in k_values:
        results.update({
            f"recall@{k}": None,
            f"hit@{k}": None,
            f"source_recall@{k}": None,
            f"ndcg@{k}": None,
            f"concept_coverage@{k}": None,
            f"concept_pass@{k}": None,
            f"unique_sources@{k}": None,
            f"duplicate_source_slots@{k}": None,
        })

    in_domain_queries = 0
    hit_sums = {k: 0.0 for k in k_values}
    source_recall_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}
    rr_sum = 0.0
    concept_eligible = 0
    concept_coverage_sums = {k: 0.0 for k in k_values}
    concept_pass_sums = {k: 0.0 for k in k_values}
    unique_source_sums = {k: 0.0 for k in k_values}
    duplicate_slot_sums = {k: 0.0 for k in k_values}
    oos_queries = 0
    context_budget_values = {"draft": [], "verifier": []}

    for item in GOLDEN_SET:
        if expanded_context:
            retrieved = query_kb_context(
                item["query"],
                n_children=10,
                n_windows=max_k,
            )
        else:
            retrieved = query_kb(item["query"], n_results=max_k)
        sources = [result.get("source", "unknown") for result in retrieved]
        is_in_domain = bool(item["expected_sources"])
        query_result = {
            "query": item["query"],
            "difficulty": item["difficulty"],
            "expected_sources": item["expected_sources"],
            "retrieved_sources": sources,
            "retrieved_distances": [result.get("distance") for result in retrieved],
            "retrieved_results": [],
            "recall_at_k": {}, "hit_at_k": {}, "source_recall_at_k": {},
            "reciprocal_rank": None, "ndcg_at_k": {}, "concepts_found": [],
            "concepts_found_at_k": {}, "concept_coverage_at_k": {}, "concept_pass_at_k": {},
            "source_diversity_at_k": {}, "oos_diagnostics": None,
            "context_budget": None,
        }
        for rank, result in enumerate(retrieved, start=1):
            retrieved_result = {
                "rank": rank,
                "source": result.get("source", "unknown"),
                "chunk_index": result.get("chunk_index"),
            }
            if expanded_context:
                retrieved_result.update({
                    "chunk_indices": result.get("chunk_indices", []),
                    "seed_chunk_indices": result.get("seed_chunk_indices", []),
                    "seed_ranks": result.get("seed_ranks", []),
                })
            query_result["retrieved_results"].append(retrieved_result)

        if expanded_context:
            query_result["context_budget"] = {
                "draft": context_budget_stats(retrieved, n_windows=3),
                "verifier": context_budget_stats(retrieved, n_windows=5),
            }
            context_budget_values["draft"].append(query_result["context_budget"]["draft"])
            context_budget_values["verifier"].append(query_result["context_budget"]["verifier"])

        if is_in_domain:
            in_domain_queries += 1
            expected_set = set(item["expected_sources"])
            for k in k_values:
                hit = hit_at_k(sources, expected_set, k)
                source_recall = source_recall_at_k(sources, expected_set, k)
                ndcg = ndcg_at_k(sources, expected_set, k)
                query_result["recall_at_k"][f"recall@{k}"] = bool(hit)
                query_result["hit_at_k"][f"hit@{k}"] = hit
                query_result["source_recall_at_k"][f"source_recall@{k}"] = round(source_recall, 3)
                query_result["ndcg_at_k"][f"ndcg@{k}"] = round(ndcg, 3)
                hit_sums[k] += hit
                source_recall_sums[k] += source_recall
                ndcg_sums[k] += ndcg
            rr = reciprocal_rank(sources, expected_set, max_k)
            rr_sum += rr
            query_result["reciprocal_rank"] = round(rr, 3)
        else:
            for k in k_values:
                query_result["recall_at_k"][f"recall@{k}"] = None
                query_result["hit_at_k"][f"hit@{k}"] = None
                query_result["source_recall_at_k"][f"source_recall@{k}"] = None

        required_concepts = item["required_concepts"]
        if required_concepts:
            concept_eligible += 1
            for k in k_values:
                found, coverage = concept_coverage_at_k(retrieved, required_concepts, k)
                concept_pass = float(coverage >= CONCEPT_PASS_THRESHOLD)
                query_result["concepts_found_at_k"][f"concepts@{k}"] = found
                query_result["concept_coverage_at_k"][f"concept_coverage@{k}"] = round(coverage, 3)
                query_result["concept_pass_at_k"][f"concept_pass@{k}"] = concept_pass
                concept_coverage_sums[k] += coverage
                concept_pass_sums[k] += concept_pass
            query_result["concepts_found"] = query_result["concepts_found_at_k"][f"concepts@{legacy_concept_k}"]

        for k in k_values:
            unique_sources, duplicate_slots = source_diversity_at_k(sources, k)
            query_result["source_diversity_at_k"][f"unique_sources@{k}"] = unique_sources
            query_result["source_diversity_at_k"][f"duplicate_source_slots@{k}"] = duplicate_slots
            unique_source_sums[k] += unique_sources
            duplicate_slot_sums[k] += duplicate_slots

        if not is_in_domain and "min_distance_threshold" in item:
            oos_queries += 1
            query_result["oos_diagnostics"] = query_kb_diagnostics(item["query"], n_results=max_k)
        results["per_query"].append(query_result)

    for k in k_values:
        if in_domain_queries:
            results[f"recall@{k}"] = round(hit_sums[k] / in_domain_queries, 3)
            results[f"hit@{k}"] = round(hit_sums[k] / in_domain_queries, 3)
            results[f"source_recall@{k}"] = round(source_recall_sums[k] / in_domain_queries, 3)
            results[f"ndcg@{k}"] = round(ndcg_sums[k] / in_domain_queries, 3)
        if concept_eligible:
            results[f"concept_coverage@{k}"] = round(concept_coverage_sums[k] / concept_eligible, 3)
            results[f"concept_pass@{k}"] = round(concept_pass_sums[k] / concept_eligible, 3)
        results[f"unique_sources@{k}"] = round(unique_source_sums[k] / len(GOLDEN_SET), 3)
        results[f"duplicate_source_slots@{k}"] = round(duplicate_slot_sums[k] / len(GOLDEN_SET), 3)

    results["mrr"] = round(rr_sum / in_domain_queries, 3) if in_domain_queries else None
    results["concept_hit_rate"] = results[f"concept_pass@{legacy_concept_k}"]
    results["in_domain_queries"] = in_domain_queries
    results["concept_eligible_queries"] = concept_eligible
    results["out_of_scope_queries"] = oos_queries
    if expanded_context:
        results["context_budget"] = {
            consumer: summarize_context_budget(values)
            for consumer, values in context_budget_values.items()
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k", nargs="+", type=int, default=[1,3,5]
    )
    parser.add_argument(
        "--output", type=str, default="outputs/retrieval_eval.json"
    )
    parser.add_argument(
        "--expanded-context",
        action="store_true",
        help="Evaluate source-aware evidence windows assembled from ten raw child candidates",
    )
    args = parser.parse_args()

    print("Running retrieval eval...")

    results = run_retrieval_eval(
        k_values=args.k,
        expanded_context=args.expanded_context,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\nRetrieval Eval Results")
    print(f"{'─' * 50}")

    for k in args.k:
        print(f"  hit@{k}: {results[f'hit@{k}']:.1%}")
        print(f"  source_recall@{k}: {results[f'source_recall@{k}']:.1%}")

    mrr = results.get("mrr")
    if mrr is not None:
        print(f"  MRR: {mrr:.1%}")
    else:
        print("  MRR: N/A (no in-domain queries)")

    for k in args.k:
        score = results.get(f"ndcg@{k}")
        if score is not None:
            print(f"  source nDCG@{k}: {score:.1%}")
        else:
            print(f"  nDCG@{k}: N/A (no in-domain queries)")

    for k in args.k:
        print(f"  concept coverage@{k}: {results[f'concept_coverage@{k}']:.1%}")
        print(f"  concept pass@{k}: {results[f'concept_pass@{k}']:.1%}")
        print(f"  mean unique sources@{k}: {results[f'unique_sources@{k}']}")
        print(f"  mean duplicate source slots@{k}: {results[f'duplicate_source_slots@{k}']}")
    print("  out-of-scope rejection: deprecated/non-gating; raw diagnostics recorded per OOS query")
    print(f"\nFull report: {out_path}")

    # Flag in-domain failures
    failures = [
        q for q in results["per_query"]
        if q["expected_sources"] and not q["hit_at_k"].get("hit@3", False)
    ]
    if failures:
        print(f"\n[WARN] {len(failures)} in-domain queries missed at recall@3:")
        for f in failures[:10]:  # limit output
            print(f"  [{f['difficulty']}] {f['query']}")
            print(f"    expected: {f['expected_sources']}")
            print(f"    got:      {f['retrieved_sources'][:3]}")

    print("\nOOS diagnostics are reported without a rejection gate; calibration remains separate work.")


if __name__ == "__main__":
    main()
