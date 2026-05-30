# Gradient Descent

## Definition
Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. It moves parameters in the direction of the negative gradient of the objective function with respect to the parameters.

## Intuition
The gradient of a function points in the direction of steepest ascent. By repeatedly taking small steps in the opposite direction, the algorithm travels downhill on the error surface, converging to a point where the gradient is zero (a critical point).

## Mathematical Formulation
Given a parameter vector θ ∈ R^d and a differentiable loss function J(θ), the update rule for step t is:

θ_{t+1} = θ_t − η · ∇J(θ_t)

where η > 0 is the learning rate (step size). The gradient ∇J(θ_t) is the vector of partial derivatives [∂J/∂θ_1, ..., ∂J/∂θ_d] evaluated at θ_t.

For a dataset of m examples, the full-batch gradient uses:

∇J(θ) = (1/m) Σ_{i=1}^m ∇L(f(x_i; θ), y_i)

In practice, the gradient is often approximated using a mini-batch of size B ≪ m.

## Estimation
Gradient descent estimates model parameters by minimizing a loss function. The procedure:
1. Initialize θ_0 (e.g., small random values).
2. For t = 0, 1, ..., until convergence:
   a. Compute gradient estimate g_t = (1/|B|) Σ_{i∈B} ∇_θ L(f(x_i; θ_t), y_i).
   b. Update θ_{t+1} = θ_t − η g_t.
3. Stop when gradient magnitude or loss improvement falls below a threshold, or after a fixed number of epochs.

Convergence is monitored via training/validation loss. The estimate θ_final is the minimizer found.

## Assumptions
- The loss function J(θ) is differentiable (almost everywhere, or subgradients exist).
- Learning rate η is suitably chosen; too large causes divergence, too small causes slow convergence.
- Data examples are i.i.d. samples from the target distribution (for generalization).
- The parameter space is unconstrained or handled via projection (or penalty) if constrained.

## Key Properties
- **Monotonic improvement**: For sufficiently small η and convex J, J(θ_{t+1}) ≤ J(θ_t). For non-convex J, this holds locally with careful step sizing.
- **Linear convergence rate** for strongly convex, smooth functions with constant step size; sublinear for convex, non-strongly convex.
- **Sensitivity to scaling**: Performance depends heavily on the conditioning of the problem; ill-conditioned problems lead to slow zigzagging.
- **Global convergence to stationary point** under Lipschitz gradient and diminishing step sizes (for stochastic variants).

## Failure Modes
- **Divergence**: Learning rate too high; loss increases to infinity.
- **Slow progress**: Learning rate too small or poorly conditioned curvature.
- **Oscillations**: Near steep ravines, constant step size leads to bouncing, slowing convergence.
- **Saddle points and local minima**: May stall in flat regions or local minima for non-convex functions, though high-dimensional saddle points are more prevalent.
- **Vanishing/exploding gradients**: Severely scaled gradients in deep networks disrupt updates.

## When to Use / Not Use
**Use**: As the default optimizer for differentiable loss functions in machine learning (linear/logistic regression, neural networks). Scalable to large datasets via mini-batches.

**Not use**: When loss is non-differentiable (use subgradient methods, evolutionary algorithms), when problem is small and exact solution is needed (closed-form OLS), when convergence speed is critical and second-order methods are feasible (Newton's method), or when the function has many local minima and global search is required.

## Variants / Extensions
- **Stochastic Gradient Descent (SGD)**: Uses a single example or mini-batch; noisy but faster.
- **Momentum**: Adds a velocity term to damp oscillations and accelerate convergence.
- **Nesterov Accelerated Gradient (NAG)**: Looks ahead along momentum direction.
- **AdaGrad, RMSProp, Adam**: Adaptive learning rate methods that scale updates per-parameter.
- **Batch Gradient Descent**: Full dataset per step; deterministic but computationally heavy.

## Minimal Example (Python)
```python
# Minimize f(x) = (x-3)^2
def f(x): return (x-3)**2
def gradf(x): return 2*(x-3)

x = 0.0          # initial guess
lr = 0.1
for t in range(100):
    x = x - lr * gradf(x)
print(x)  # ~3.0
```
## Claim-Dense Reference Facts

- The standard SGD update rule is θ = θ - η·∇J(θ) where η is the learning rate scalar.
- Adam optimizer uses β₁=0.9, β₂=0.999, ε=1e-8 as default hyperparameters in PyTorch.
- Momentum SGD adds a velocity term: v = γv + η·∇J(θ), θ = θ - v, where γ=0.9 is typical.
- Learning rate warmup linearly increases η from 0 to target over the first 5–10% of training steps.
- Cosine annealing decays learning rate as η_t = η_min + 0.5(η_max - η_min)(1 + cos(πt/T)).
- Gradient clipping by norm is applied when ||g|| > threshold, rescaling g = g·(threshold/||g||).
- PyTorch's `torch.optim.SGD` accepts lr, momentum, weight_decay, and nesterov parameters.
- For logistic regression on convex loss, gradient descent with constant step size converges at O(1/k) rate.
- Mini-batch size of 32–256 is standard; larger batches require learning rate scaling (linear scaling rule).
- Cyclical learning rates (CLR) oscillate between η_min and η_max over a fixed cycle of 2–10 epochs.