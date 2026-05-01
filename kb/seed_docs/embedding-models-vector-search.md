# Embedding Models & Vector Search

## Definition
Embedding models are functions that map discrete inputs (text tokens, images, audio) into dense, fixed-dimensional real-valued vectors, such that semantically similar items are placed close in the vector space. Vector search is the task of efficiently retrieving the top-k vectors most similar to a query vector from a large collection, using a distance metric (cosine, Euclidean) and approximate nearest neighbor (ANN) indexing.

## Intuition
Embeddings compress semantic meaning into a numeric point; items with related meaning cluster together. Vector search enables fast lookup of relevant items from millions of candidates by navigating a pre-built index, avoiding exhaustive pairwise comparison.

## Mathematical Formulation
An embedding model f: X → R^d maps input x to v = f(x). Similarity between query q and document d is measured via cosine similarity: cos(q,d) = (q·d) / (||q|| ||d||). The top-k retrieval solves k = argmax_{i∈D} cos(f(q), f(d_i)). Approximate algorithms (HNSW, IVF-PQ) build graph or partition-based indices to achieve sub-linear O(log N) query time by trading minimal accuracy.

## Estimation
Embedding models are trained using contrastive or triplet losses that maximize similarity between positive pairs and minimize similarity for negatives. Training data comes from parallel corpora, question-answer pairs, or co-occurrence signals. The vector index is built offline: all document vectors are ingested and an ANN index is constructed (e.g., HNSW graph, IVF centroids). At query time, the same encoder produces the query vector, and the ANN index returns top-k results.

## Assumptions
- Embedding space is Euclidean, and the consistency metric captures semantic similarity.
- Query distribution matches training data; domain shift can degrade relevance.
- Cosine similarity (or chosen metric) is a valid relevance proxy for the task.
- Approximate search accuracy loss is acceptable and configured.
- Index can be rebuilt when the corpus changes significantly.

## Key Properties
- **Semantic matching**: Captures synonymy, paraphrasing, and topical relatedness beyond keyword overlap.
- **Fixed-length representation**: Enables geometric operations (similarity, clustering, arithmetic).
- **Scalability**: Billions of vectors can be searched in milliseconds with ANN indexes.
- **Metric boundedness**: Cosine similarity ∈ [-1,1], magnitude-invariant; Euclidean distance is unbounded.
- **Offline indexing, online query**: Index preparation is amortized; retrieval is fast.

## Failure Modes
- **Out-of-domain encoding**: Model unfamiliar with niche vocabulary produces poor representations.
- **Curse of dimensionality**: Distances lose discriminability if dimension d is very high relative to data volume.
- **False nearest neighbors**: ANN may miss true top items, especially for skewed vector distributions.
- **Embedding collision**: Unrelated concepts mapped close, causing irrelevant results.
- **Index staleness**: Document changes not reflected until re-indexing.

## When to Use / Not Use
**Use**: Semantic search, retrieval-augmented generation, recommendation, deduplication, clustering, multimodal retrieval.

**Not use**: When exact keyword match is sufficient and interpretable (database LIKE query), or when corpus is tiny and exhaustive search is trivial, or when low-latency insertion/deletion without re-indexing is critical.

## Variants / Extensions
- **Sparse embeddings**: Bag-of-words or learned sparse vectors (SPLADE) for interpretable hybrid retrieval.
- **Multi-vector models**: Per-token embeddings for late interaction (ColBERT).
- **Multimodal embeddings**: Joint space for text and images (CLIP, ImageBind).
- **ANN index types**: HNSW (graph-based), IVF-PQ (quantization), LSH (hashing), Annoy (tree-based).
- **Matryoshka embeddings**: Nested dimensions for flexible accuracy-efficiency tradeoffs.

## Minimal Example (Python)
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
docs = ["Paris is the capital of France", "London is in the UK"]
vecs = model.encode(docs)  # (2,384)
index = faiss.IndexFlatIP(384)  # inner product = cosine for normalized vecs
index.add(vecs.astype(np.float32))
q_vec = model.encode(["What is the capital of France?"])
D, I = index.search(q_vec.astype(np.float32), k=1)  # top-1
```