# Attention Mechanism & Transformers

## Definition
An attention mechanism maps a query and a set of key-value pairs to an output, where the output is a weighted sum of values. Weights (attention scores) are computed via a compatibility function between the query and each key. A Transformer is a neural architecture built entirely on attention, without recurrence or convolution, using multi-head self-attention, feedforward layers, and positional encodings (Vaswani et al., 2017).

## Intuition
Attention dynamically aggregates information from all positions based on learned relevance. Transformers exploit this to capture long-range dependencies in parallel across the full sequence, avoiding the sequential bottleneck of recurrent models.

## Mathematical Formulation
**Scaled Dot-Product Attention**  
Given queries Q, keys K (dim d_k), values V (dim d_v):  
Attention(Q, K, V) = softmax( Q K^T / √d_k ) V.

**Multi-Head Attention**  
Linearly project Q, K, V h times, apply attention in parallel, concatenate, and project:  
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)  
MultiHead(Q,K,V) = Concat(head_1,… ,head_h) W^O.

**Transformer Layer**  
Input x (batch × seq_len × d_model):  
1. Multi-head self-attention (Q = K = V = x), with residual connection and layer normalization.  
2. Position-wise feedforward network (two linear layers with activation, typically ReLU/GELU), with residual connection and layer normalization.  
Positional encodings (fixed sinusoidal or learned) are added to input embeddings to encode order.

## Estimation
Transformers are trained end-to-end with backpropagation. Loss functions are task-specific: cross-entropy for language modeling or sequence-to-sequence tasks. Optimization often uses Adam with warm-up and gradient clipping. Autoregressive decoders apply causal masking to prevent attending to future tokens. Large-scale pre-training (BERT, GPT) uses self-supervised objectives, then fine-tuning on downstream tasks.

## Assumptions
- Dependencies can be modeled via pairwise attention scores, not requiring sequential processing.
- Sufficient sequence length for learning long-range interactions.
- Positional information must be explicitly supplied.
- Residual connections are necessary for training deep architectures.
- Quadratic attention cost is acceptable for typical sequence lengths.

## Key Properties
- **Global receptive field**: Each output can directly attend to any input, enabling O(1) path length for long-range dependencies.
- **Parallel computation**: All timesteps processed simultaneously during training.
- **Permutation equivariance** without positional encoding; position must be injected.
- **O(n²·d) complexity** for self-attention on n tokens; memory-intensive.
- **Interpretable attention weights** provide some insight into model focus.

## Failure Modes
- **Quadratic memory/time** prohibits very long sequences (>10k tokens) without specialized variants.
- **Training instability** in large models may require careful normalization and learning rate schedules.
- **Position insensitivity** with fixed encodings may fail to capture fine-grained relative order for unseen lengths.
- **Over-smoothing**: Token representations may become indistinguishable in deep stacks without residual connections.
- **Inductive bias poverty**: Lacks built-in locality or sequential bias; requires large data to learn these priors.

## When to Use / Not Use
**Use**: NLP tasks (translation, classification, generation), vision (Vision Transformers), speech, and any domain with global dependencies where parallel processing is beneficial. Pre-training + fine-tuning paradigm yields state-of-the-art results.

**Not Use**: Very long sequences without efficient attention variants; real-time low-latency constraints on long inputs; extremely small data regimes where strong inductive biases (e.g., CNNs) are crucial.

## Variants / Extensions
- **Encoder-only** (BERT), **decoder-only** (GPT), **encoder-decoder** (T5).
- **Efficient attention**: Sparse (Longformer), linear (Performer), low-rank approximations.
- **Vision Transformer (ViT)**: Patch-based image processing.
- **Transformer-XL**: Extended context via segment-level recurrence.
- **Swin Transformer**: Hierarchical shifted windows for vision.

## Minimal Example (Python)
```python
import torch.nn as nn

class SelfAttnBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)
```