# M2 claim review (human adjudication)

For each claim, open the source_url (or check the frozen cache) and fill the last 2 columns.

supported = is the claim's substance actually in the provided source(s)? yes / partial / no

in_sources = does ANY retrieved source for this topic contain the info? yes / no


Pool size: 220 unverified+weak claims. Sample: 30.


| # | topic | status | conf | claim | source_url | supported | in_sources |
|---|---|---|---|---|---|-----------|------------|
| 1 | Embedding Models & | unverified | 0.0 | At millions of vectors, this is too slow. | — |           |            |
| 2 | Multi-Agent System | weak | 0.7 | Multi-agent systems shine when tasks are naturally parallelizable, require diverse expertise, or need iterative refinement across different domains. | https://www.techaheadcorp.com/blog/ways-multi-agent-ai-fails-in-production | partial   | yes        |
| 3 | Multi-Agent System | unverified | 0.0 | In LangGraph, you define this as a StateGraph. | — | No        | No         |
| 4 | Embedding Models & | unverified | 0.0 | Larger vectors (768d) are more expressive but slower. | — | No        | No         |
| 5 | ReAct Agent Patter | unverified | 0.0 | Tool execution latency adds up. | — | No        | No         |
| 6 | Embedding Models & | unverified | 0.0 | Popular libraries include FAISS (Facebook AI Similarity Search), Annoy (Spotify), and HNSWlib (hierarchical navigable small world graphs). | — | no        | no         |
| 7 | Support Vector Mac | unverified | 0.0 | SVMs are not suitable for very large datasets (n > 100k) due to O(n²) training time and memory requirements. | — | partial   | yes        |
| 8 | CatBoost | weak | 0.7 | This makes the decision function a piecewise-constant function over a grid of hyper-rectangles. | https://www.geeksforgeeks.org/machine-learning/catboost-algorithms | no        | no         |
| 9 | Embedding Models & | unverified | 0.0 | For example, 'apple' (fruit) and 'Apple' (company) may be close if the model conflates contexts. | — | no        | no         |
| 10 | Multi-Agent System | unverified | 0.0 | A single agent has a limited context window and a single set of instructions. | — | no        | no         |
| 11 | ReAct Agent Patter | unverified | 0.0 | The model can hallucinate observations if the tool returns an error — you must surface the actual error message, not let the model guess. | — | no        | no         |
| 12 | ReAct Agent Patter | unverified | 0.0 | For latency-sensitive apps, batch tool calls or use a faster model. | — | no        | no         |
| 13 | Multi-Agent System | unverified | 0.0 | Strict message contracts (Pydantic models) prevent inter-agent communication failures. | — | no        | no         |
| 14 | CatBoost | unverified | 0.0 | Prediction shift occurs when the target distribution used to compute gradients at a given boosting iteration is the same distribution used to train the model. | — | no        | no         |
| 15 | ReAct Agent Patter | unverified | 0.0 | Mitigation for brittle parsing: use robust parsing with fallback prompts. | — | no        | no         |
| 16 | Embedding Models & | weak | 0.7 | Embeddings can be biased by training data — a model trained on Wikipedia may not handle domain-specific jargon. | https://weaviate.io/blog/vector-search-explained | no        | no         |
| 17 | Multi-Agent System | unverified | 0.0 | This works well for tasks like research: a planner agent generates search queries, a retrieval agent runs them, a writer agent compiles the answer. | — | no        | no         |
| 18 | Multi-Agent System | unverified | 0.0 | A common pattern is the supervisor-agent topology. | — | no        | no         |
| 19 | Embedding Models & | unverified | 0.0 | Exact nearest neighbor search is O(n*d) — linear in the number of vectors and dimensionality. | — | no        | no         |
| 20 | CatBoost | unverified | 0.0 | CatBoost offers a boosting_type parameter: 'Ordered' (default for small datasets) and 'Plain' (standard boosting, faster). | — | no        | no         |
| 21 | Embedding Models & | unverified | 0.0 | Vector search can return false positives if the embedding space is not well-separated. | — | no        | no         |
| 22 | CatBoost | weak | 0.7 | CatBoost's default hyperparameters are tuned for medium-sized datasets. | https://www.geeksforgeeks.org/machine-learning/catboost-algorithms | no        | no         |
| 23 | Multi-Agent System | unverified | 0.0 | One agent (the supervisor) receives the user request, breaks it into subtasks, and dispatches each to a specialized worker agent. | — | no        | no         |
| 24 | ReAct Agent Patter | unverified | 0.0 | The loop can be modeled as a Markov decision process where the state is the full context, actions are tool calls, and the reward is task completion. | — | no        | bo         |
| 25 | CatBoost | weak | 0.8 | The GPU implementation has limitations: not all loss functions are supported, and memory usage scales with the number of categories. | https://www.geeksforgeeks.org/machine-learning/catboost-algorithms | no        | no         |
| 26 | Embedding Models & | unverified | 0.0 | Dimensionality reduction via PCA can help but may lose signal. | — | no        | no         |
| 27 | CatBoost | unverified | 0.0 | The algorithm uses a random permutation of the training data and for each iteration, it computes gradients for each example using a model that was trained on th | — | no        | no         |
| 28 | Embedding Models & | unverified | 0.0 | Models with 768 or 1024 dimensions can still work, but the index may need more data or tuning. | — | no        | no         |
| 29 | Embedding Models & | unverified | 0.0 | A common failure mode is the 'curse of dimensionality.' As vector dimensions increase, distances become less discriminative. | — | no        | no         |
| 30 | Support Vector Mac | unverified | 0.0 | SVMs require careful hyperparameter tuning (C, γ, kernel choice) and are less flexible than neural networks for raw pixel or sequence data. | — | partial   | yes        |