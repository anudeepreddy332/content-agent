# Retrieval Augmented Generation (RAG)

## Definition
Retrieval Augmented Generation (RAG) is a framework that combines a pre-trained parametric memory (a sequence-to-sequence generator) with a non-parametric memory (a dense vector index of documents) via a neural retriever. At inference time, relevant documents are fetched from the index and provided as additional context for generation, enabling factual grounding and access to external knowledge without enlarging model parameters.

## Intuition
Standard language models store knowledge implicitly in their weights, which is expensive to update and prone to hallucination. RAG offloads factual recall to an external, easily updatable retrieval corpus, allowing the generator to focus on synthesizing an answer from retrieved evidence.

## Mathematical Formulation
Given an input query x, a retriever with parameters η assigns a document z from corpus D a probability p_η(z|x) ∝ exp( d(z)^T q(x) ), where q and d are dense encoders (e.g., BERT-based). A generator with parameters θ produces target y. Two common formulations:

**RAG-Sequence** marginalizes over the top-k documents Z:
p_Seq(y|x) = Σ_{z∈Z} p_η(z|x) ∏_t p_θ(y_t | y_{<t}, z, x)

**RAG-Token** marginalizes per generated token:
p_Token(y|x) = ∏_t Σ_{z∈Z} p_η(z|x) p_θ(y_t | y_{<t}, z, x)

In practice, often a single set of k retrieved documents Z is concatenated with x and passed to a generator that models p_θ(y|x, Z) without marginalization, using architectures like FiD where all documents are encoded jointly.

## Estimation
Training maximizes the log-likelihood of target y. The retriever is typically initialized from a pre-trained dense retriever (DPR) and can be kept frozen or fine-tuned. Direct backpropagation through discrete retrieval is circumvented by treating the top-k documents as fixed latent variables from the current retriever, computing the generator’s loss for those documents, and updating the retriever via re-indexing and asynchronous parameter updates, or by using an EM-like approach. The index is built offline with document encoders, and refreshed periodically. Contrastive training of the retriever uses in-batch negatives and hard negatives.

## Assumptions
- The external corpus contains documents relevant to the queries.
- The dense retriever can effectively index and rank documents by semantic similarity.
- The generator can handle variable-length concatenated evidence without confusion.
- Retrieval latency is within the application’s tolerance.
- The document index can be rebuilt or updated as knowledge evolves.

## Key Properties
- **Decoupled knowledge**: Model parameters and factual knowledge are separate; knowledge updates only require re-indexing.
- **Reduced hallucination**: Outputs are grounded in retrieved text, with provenance for verification.
- **Scalability**: The corpus can be arbitrarily large, limited only by retrieval efficiency.
- **Modularity**: Retriever and generator can be swapped or independently improved.
- **Optimization challenge**: Discrete retrieval is a non-differentiable bottleneck; training requires workarounds.

## Failure Modes
- **Retrieval failure**: The correct document is not in the top-k, causing the model to ignore or hallucinate.
- **Noisy retrieval**: Irrelevant documents distract the generator, degrading answer quality.
- **Factual distortion**: Mixing information from multiple documents can create inconsistent statements.
- **Staleness**: If the index is not updated, retrieved information becomes outdated.
- **Latency**: Retrieval step adds inference time, critical for real-time applications.

## When to Use / Not Use
**Use**: Open-domain question answering, fact verification, enterprise knowledge integration, long-tail knowledge tasks, any scenario requiring up-to-date or verifiable information grounded in documents.

**Not Use**: Tasks where the model’s implicit knowledge is sufficient and retrieval overhead is unacceptable; online chat requiring minimal latency without a retrieval backend; domains where the corpus is unavailable, unreliable, or too small to matter.

## Variants / Extensions
- **FiD (Fusion-in-Decoder)**: Concatenates all retrieved documents in the encoder, generates from fused representation.
- **RETRO**: Interleaves chunked retrieval with cross-attention in intermediate layers.
- **Atlas**: Jointly trains retriever and generator via a unified loss with full backpropagation through retrieval using approximate gradients.
- **Self-RAG**: Incorporates retrieval on-demand with reflection tokens to judge relevance.
- **REALM**: Pre-trains a masked language model with retrieval for open-domain representations.

## Minimal Example (Python)
```python
from transformers import RagTokenizer, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")
input_ids = tokenizer("What is the capital of France?", return_tensors="pt").input_ids
generated = model.generate(input_ids)
print(tokenizer.decode(generated[0], skip_special_tokens=True))
```
## Claim-Dense Reference Facts

- LlamaIndex and LangChain both implement RAG pipelines; LlamaIndex defaults to chunk_size=1024, chunk_overlap=20 tokens.
- Standard RAG retrieves k=3–5 documents; increasing to k=10 improves recall but increases context length and latency.
- Reranking with cross-encoders (e.g., ms-marco-MiniLM-L-6-v2) improves top-1 precision by 10–20% over bi-encoder retrieval alone.
- HyDE (Hypothetical Document Embeddings) generates a fake answer to the query and embeds it, improving retrieval for abstract queries.
- Parent-child chunking stores small chunks for retrieval but passes their larger parent chunks to the generator for context.
- RAG-Fusion generates multiple query variants, retrieves for each, and merges results with Reciprocal Rank Fusion.
- Contextual compression reduces retrieved text to only the relevant portions before passing to the generator.
- RAGAS is the standard evaluation framework for RAG pipelines, measuring faithfulness, answer relevancy, and context recall.
- Dense Passage Retrieval (DPR) uses separate question and passage encoders; both are BERT-base sized (110M parameters).
- Self-RAG uses four special tokens: [Retrieve], [ISREL], [ISSUP], [ISUSE] to decide when to retrieve and evaluate relevance.