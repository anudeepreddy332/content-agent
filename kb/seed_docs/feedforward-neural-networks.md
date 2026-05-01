# Feedforward Neural Networks

## Definition
A feedforward neural network (FFNN) is an artificial neural network where connections between nodes do not form cycles. Information moves in one direction—from input, through hidden layers, to output—without feedback loops. When composed of fully connected layers with nonlinear activation functions, it is often called a multilayer perceptron (MLP).

## Intuition
The network learns a mapping from input vectors to output vectors by composing simple, parameterized transformations. Each layer extracts progressively abstract representations, enabling the model to capture complex, nonlinear relationships in data.

## Mathematical Formulation
An L-layer feedforward network defines a function f(x; θ) = f_L ∘ f_{L-1} ∘ ... ∘ f_1(x), where each layer l computes:

h^(l) = g_l( W^(l) h^(l-1) + b^(l) )

- h^(0) = x (input vector)
- W^(l) is a weight matrix, b^(l) a bias vector
- g_l is a nonlinear activation function (e.g., ReLU, sigmoid, tanh)
- h^(l) is the activation output of layer l
- For regression, output layer typically has identity activation; for classification, softmax or sigmoid.

All parameters are collected in θ = {W^(l), b^(l)}_{l=1}^L.

## Estimation
Parameters are learned by minimizing a loss function L(y_true, f(x; θ)) on a training set, using gradient-based optimization. Common procedure:
1. Forward pass: compute predictions.
2. Compute loss.
3. Backward pass: compute gradients of loss w.r.t. parameters via backpropagation (chain rule).
4. Update parameters using an optimizer (SGD, Adam, etc.) with learning rate η: θ ← θ − η ∇_θ L.
Regularization (L1/L2, dropout) may be added. Training uses mini-batches.

## Assumptions
- Training examples are independent and identically distributed (i.i.d.).
- The target function can be approximated by a composition of affine transformations and elementwise nonlinearities.
- Sufficient network capacity (width, depth) exists for the task.
- Input features are appropriately scaled (normalized).

## Key Properties
- **Universal approximation**: A single hidden layer with a non-polynomial activation can approximate any continuous function on a compact set, given enough units.
- **Differentiability**: The entire network is a differentiable function, enabling gradient-based training.
- **Representation hierarchy**: Deeper layers can learn hierarchical and distributed representations.
- **Scalability**: Training can be parallelized over data and model parameters.

## Failure Modes
- **Overfitting**: Too many parameters relative to data; poor generalization.
- **Vanishing/exploding gradients**: In deep networks, gradients may become extremely small or large, impeding training (mitigated by ReLU, batch normalization, skip connections).
- **Local minima / saddle points**: Optimization may converge to suboptimal solutions, though often benign in high dimensions.
- **Dead neurons**: With ReLU, large negative bias can permanently deactivate units.
- **Inability to handle variable-length inputs or sequential dependence** without architectural changes.

## When to Use / Not Use
**Use**: Tabular data, classification, regression, approximating static mappings, when relationships are nonlinear but not inherently sequential or spatially structured. Good for baseline models and problems with fixed-size input.

**Not use**: When data exhibits temporal dynamics (use RNNs, LSTMs), spatial locality (use CNNs), graph structure (use GNNs), or when interpretability is critical. Also not ideal for extremely low-data regimes where simpler models suffice.

## Variants / Extensions
- **Deep feedforward networks**: Increased depth for hierarchical features.
- **Convolutional neural networks (CNNs)**: Replace matrix multiplication with convolution for spatial invariance.
- **Recurrent neural networks (RNNs)**: Add feedback connections for sequences.
- **Residual networks (ResNets)**: Introduce skip connections to ease gradient flow in deep architectures.
- **Transformer feedforward blocks**: Position-wise FFNNs used within transformer layers.

## Minimal Example (Python)
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)  # regression output
)
# Loss: MSELoss, Optimizer: Adam
```