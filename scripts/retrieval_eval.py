"""
Measures retrieval accuracy (recall@k, concept-hit rate, out-of-scope rejection
rate) against a fixed 35-query golden set, evaluated against whatever is
currently in the KB. To compare across corpus sizes, re-run this script after
each ingest stage and diff the JSON reports by hand — there is no built-in
loop or plotting; each invocation is a single point-in-time measurement.

Usage:
    uv run python scripts/retrieval_eval.py
    uv run python scripts/retrieval_eval.py --k 1 3 5 --output retrieval_report.json
"""

import sys, json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.query_kb import query_kb

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

def run_retrieval_eval(k_values: list[int] = None) -> dict:
    """
    Run retrieval evaluation on a golden set.

    Args:
        k_values: List of k values for recall@k

    Returns:
        dict with metrics and per-query details.
    """

    if k_values is None:
        k_values = [1,3,5]

    # Initialize results
    results = {f"recall@{k}": 0 for k in k_values}
    results["concept_hit_rate"] = 0
    results["out_of_scope_rejection_rate"] = 0
    results["per_query"] = []
    results["total_queries"] = len(GOLDEN_SET)

    # Separate counters
    in_domain_queries = 0
    in_domain_recall_hits = {k: 0 for k in k_values}
    concept_eligible = 0
    concept_hits = 0
    oos_eligible = 0
    oos_rejections = 0

    max_k = max(k_values)

    for item in GOLDEN_SET:
        # Retrieve docs
        retrieved = query_kb(item["query"], n_results=max_k)
        sources = [r["source"] for r in retrieved]
        distances = [r.get("distance") for r in retrieved[:max_k]]
        combined_text = " ".join(r["text"] for r in retrieved).lower()

        query_result = {
            "query": item["query"],
            "difficulty": item["difficulty"],
            "expected_sources": item["expected_sources"],
            "retrieved_sources": sources,
            "retrieved_distances": distances,
            "recall_at_k": {},
            "concepts_found": [],
            "out_of_scope_violation": False,
        }

        # Recall for in-domain queries
        is_in_domain = bool(item["expected_sources"])

        if is_in_domain:
            in_domain_queries += 1
            expected_set = set(item["expected_sources"])

            for k in k_values:
                top_k_sources = set(sources[:k])
                hit = bool(expected_set.intersection(top_k_sources))
                query_result["recall_at_k"][f"recall@{k}"] = hit

                if hit:
                    in_domain_recall_hits[k] += 1

        else:
            # Out-of-scope: recall not applicable
            for k in k_values:
                query_result["recall_at_k"][f"recall@{k}"] = None


        # Concept check (only if required_concepts non-empty)
        if item["required_concepts"]:
            concept_eligible += 1
            concepts_found = [c for c in item["required_concepts"] if c.lower() in combined_text]
            query_result["concepts_found"] = concepts_found

            # Require at least 67% of concepts present
            if len(concepts_found) >= len(item["required_concepts"]) * 0.67:
                concept_hits += 1

        # Out-of-scope rejection check
        if "min_distance_threshold" in item and not is_in_domain:
            oos_eligible += 1
            # Expect smallest distance (most similar) to be > threshold
            if distances and distances[0] > item["min_distance_threshold"]:
                oos_rejections += 1
            else:
                query_result["out_of_scope_violation"] = True

        results["per_query"].append(query_result)

    # Normalize metrics
    for k in k_values:
        if in_domain_queries > 0:
            results[f"recall@{k}"] = round(in_domain_recall_hits[k] / in_domain_queries, 3)
        else:
            results[f"recall@{k}"] = None

    results["concept_hit_rate"] = round(concept_hits / concept_eligible, 3) if concept_eligible > 0 else None
    results["out_of_scope_rejection_rate"] = round(oos_rejections / oos_eligible, 3) if oos_eligible > 0 else None

    # Add metadata
    results["in_domain_queries"] = in_domain_queries
    results["concept_eligible_queries"] = concept_eligible
    results["out_of_scope_queries"] = oos_eligible

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k", nargs="+", type=int, default=[1,3,5]
    )
    parser.add_argument(
        "--output", type=str, default="outputs/retrieval_eval.json"
    )
    args = parser.parse_args()

    print("Running retrieval eval...")

    results = run_retrieval_eval(k_values=args.k)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\nRetrieval Eval Results")
    print(f"{'─' * 50}")

    for k in args.k:
        score = results.get(f"recall@{k}")
        if score is not None:
            bar = "█" * int(score * 20)
            print(f"  recall@{k}: {score:.1%}  {bar}")
        else:
            print(f"  recall@{k}: N/A (no in-domain queries)")

    print(f"  concept hit rate: {results['concept_hit_rate'] if results['concept_hit_rate'] is not None else 'N/A'}")
    print(f"  out-of-scope rejection rate: {results['out_of_scope_rejection_rate'] if results['out_of_scope_rejection_rate'] is not None else 'N/A'}")
    print(f"\nFull report: {out_path}")

    # Flag in-domain failures
    failures = [
        q for q in results["per_query"]
        if q["expected_sources"] and not q["recall_at_k"].get("recall@3", False)
    ]
    if failures:
        print(f"\n[WARN] {len(failures)} in-domain queries missed at recall@3:")
        for f in failures[:10]:  # limit output
            print(f"  [{f['difficulty']}] {f['query']}")
            print(f"    expected: {f['expected_sources']}")
            print(f"    got:      {f['retrieved_sources'][:3]}")

    # Flag out-of-scope violations (false positives)
    oos_violations = [
        q for q in results["per_query"]
        if q.get("out_of_scope_violation", False)
    ]
    if oos_violations:
        print(f"\n[ERROR] {len(oos_violations)} out-of-scope queries had distance <= threshold (false positives):")
        for v in oos_violations[:5]:
            print(f"  {v['query']}")
            print(f"    distances: {v['retrieved_distances'][:3]}")
            print(f"    got: {v['retrieved_sources'][:2]}")


if __name__ == "__main__":
    main()