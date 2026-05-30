# Backpropagation

## Definition
Backpropagation is an algorithm that computes the gradient of a scalar loss function with respect to all parameters of a differentiable computational graph by applying the chain rule recursively from output to input. It enables efficient gradient-based learning in neural networks.

## Intuition
A deep network is a composition of many functions. Backpropagation computes the gradient of the loss with respect to each internal parameter by propagating error signals backward through the network, reusing partial results to avoid redundant computation and achieving linear time complexity in the number of parameters.

## Mathematical Formulation
Consider a feedforward graph with layers l=1,...,L. Forward: a^(0)=x, a^(l)=f_l(a^(l-1); θ^(l)). Loss L = ℓ(a^(L), y). Define the error signal δ^(L) = ∂L/∂a^(L) · f_L'(z^(L)) (for elementwise activation). For each layer l = L down to 1:

δ^(l) = ((W^(l+1))^T δ^(l+1)) ⊙ g_l'(z^(l))

where z^(l) = W^(l) a^(l-1) + b^(l), and g_l is activation with derivative g_l'. Gradients w.r.t. parameters:

∂L/∂W^(l) = δ^(l) (a^(l-1))^T
∂L/∂b^(l) = δ^(l)

These accumulate over mini-batches. More generally, for an arbitrary scalar output y and tensor input x, backpropagation computes vector-Jacobian products: ∂y/∂x = Σ_i (∂y/∂x_i) e_i, efficiently using the transpose of the Jacobian of each operation.

## Estimation
Backpropagation is the gradient computation step within parameter estimation. During training:
1. Forward pass: compute and store activations and intermediate tensors.
2. Compute scalar loss.
3. Backward pass: invoke backpropagation to obtain gradients of loss w.r.t. every parameter tensor.
4. Update parameters using an optimizer (e.g., SGD).
Automatic differentiation frameworks (PyTorch, JAX, TensorFlow) track the computation graph and perform backpropagation automatically.

## Assumptions
- All operations in the graph are differentiable almost everywhere and their local derivatives are known.
- The graph is a directed acyclic graph (DAG); recurrent structures are unrolled.
- Sufficient memory exists to store intermediate activations for the backward pass (or recomputation is used).
- No irreversible operations that destroy gradient flow (e.g., discrete sampling) unless surrogate gradients are employed.

## Key Properties
- **Linear time complexity**: Total cost of one forward and one backward pass is at most a small constant factor (typically ~2–3x) of the forward pass alone.
- **Modular**: Each operator only needs to provide a forward function and a backward (VJP) function.
- **Exact gradients**: Computes analytical gradients to numerical precision, avoiding error of finite differences.
- **Memory cost**: Scales with the number of stored intermediate values, which can be large for deep or high-resolution networks.

## Failure Modes
- **NaN/Inf gradients**: Arise from operations like division by zero, log of zero, or exploding intermediate values.
- **Numerical instability**: Large dynamic ranges can cause underflow/overflow in gradient mathematics.
- **Memory exhaustion**: Storing all activations for a large graph can exceed RAM.
- **Broken gradient flow**: Non-differentiable operations (hard thresholding) yield zero or incorrect gradients unless smoothed.
- **Incorrect custom gradients**: Manual implementation of backward rules can introduce errors.

## When to Use / Not Use
**Use**: Always when training any differentiable parametric model with gradient descent; it is the standard method for computing gradients in deep learning.

**Not use**: When the model is not differentiable, when gradient-free optimization is preferred (e.g., evolutionary strategies), or when exact symbolic differentiation of a small fixed function via closed form is simpler and memory is not an issue.

## Variants / Extensions
- **Backpropagation through time (BPTT)**: Unrolls recurrent networks over sequence steps and applies backprop.
- **Truncated BPTT**: Limits the unrolling length a timent for efficiency.
- **Gradient checkpointing**: Trades computation for memory by recomputing select activations during backward.
- **Second-order backprop (Hessian-vector products)**: Computes products with the Hessian using a second backward pass.
- **Synthetic gradients**: Use learned models to approximate backpropagation signals for decoupling layers.

## Minimal Example (Python)
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2 * x**2
y.backward()          # computes dy/dx
print(x.grad)         # 3*2^2 + 4*2 = 20
```

## Claim-Dense Reference Facts

- PyTorch autograd records operations on tensors with requires_grad=True in a dynamic computation graph.
- The `.backward()` call in PyTorch accumulates gradients in `.grad` attributes; call `.zero_grad()` to reset.
- Gradient checkpointing reduces memory from O(L) to O(√L) by recomputing activations during backward pass.
- In PyTorch, `torch.utils.checkpoint.checkpoint()` wraps a function to enable gradient checkpointing.
- Vanishing gradients occur when |∂L/∂W^(l)| < 1 repeatedly across layers, exponentially shrinking updates.
- ReLU mitigates vanishing gradients by having gradient 1 for positive inputs vs sigmoid's maximum of 0.25.
- Exploding gradients are mitigated by gradient clipping, typically at max_norm=1.0 in transformers.
- BPTT for RNNs unrolls the network through time; truncated BPTT limits unrolling to k=20–100 steps.
- The memory cost of storing all activations for backward is proportional to batch_size × sequence_length × d_model.
- Xavier initialization sets weights from U(-√(6/(n_in+n_out)), √(6/(n_in+n_out))) to preserve gradient variance.