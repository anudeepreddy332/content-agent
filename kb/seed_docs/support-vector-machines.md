# Support Vector Machines

## Definition
A Support Vector Machine (SVM) is a supervised learning algorithm that constructs a hyperplane in a high-dimensional space to separate classes. The objective is to maximize the margin between the closest points of different classes. Flexibility for non-linear boundaries is provided by the kernel trick, which implicitly computes dot products in a transformed feature space. The method also extends to regression (support vector regression) and novelty detection.

## Intuition
The separating hyperplane with the largest margin is preferred because it is expected to generalize better, as it tolerates small perturbations without misclassifying. Only the training points lying exactly on the margin or violating it (support vectors) determine the solution, making the model sparse and the decision boundary robust to points far away.

## Mathematical Formulation
For a binary classification problem with labels y_i ∈ {−1,+1}, the primal hard-margin objective is: minimize (1/2)‖w‖² subject to y_i(w·φ(x_i) + b) ≥ 1. To handle non-separable data, slack variables ξ_i ≥ 0 are introduced: minimize (1/2)‖w‖² + C Σ ξ_i subject to y_i(w·φ(x_i) + b) ≥ 1 − ξ_i, where C > 0 controls the trade-off between margin width and misclassification.

The dual formulation is: maximize Σ α_i − (1/2) Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j), subject to 0 ≤ α_i ≤ C and Σ α_i y_i = 0. The kernel function K(x_i, x_j) = ⟨φ(x_i), φ(x_j)⟩ avoids explicit computation of φ. The decision function becomes: f(x) = sign(Σ α_i y_i K(x_i, x) + b).

For regression (SVR), an ε-insensitive loss is used: minimize (1/2)‖w‖² + C Σ (ξ_i + ξ_i^*), with constraints y_i − w·φ(x_i) − b ≤ ε + ξ_i, etc.

## Estimation
The dual is a convex quadratic program. Sequential Minimal Optimization (SMO) decomposes it into small sub-problems, enabling efficient training. The bias b is computed from the KKT conditions. Common kernels: linear K(u,v)=u·v; polynomial K(u,v)=(γ u·v + r)^d; RBF K(u,v)=exp(−γ‖u−v‖²). Hyperparameters C and kernel parameters are typically selected via grid search with cross-validation.

## Assumptions
Data is assumed to be (approximately) linearly separable in the kernel-induced feature space. The chosen kernel must be a valid Mercer kernel. Feature scaling (normalization/standardization) is essential for distance-based kernels like RBF. No prior distribution on the data is required. The support vectors define the boundary; data points far from the boundary have no influence.

## Key Properties
The solution is sparse: only a subset of training points become support vectors. The optimization is convex, guaranteeing a unique global optimum for the dual (for convex K). The margin maximization principle relates to structural risk minimization and provides theoretical bounds on generalization error. SVMs directly handle high-dimensional data and, with a suitable kernel, can yield complex non-linear boundaries without an explicit feature map.

## Failure Modes
Sensitive to the choice of C and kernel parameters: a small RBF γ leads to overfitting, a large γ to underfitting. No direct probabilistic output (requires Platt scaling). Training time scales poorly for very large datasets (quadratic to cubic in n). SVMs struggle when classes heavily overlap with no clear margin and when noise dominates. Multi-class problems require decompositions (one-vs-one, one-vs-rest) that may not capture full structure. Features must be scaled; otherwise, the margin is distorted.

## When to Use / Not Use
Use when the number of attributes is large relative to the number of instances (e.g., text classification, bioinformatics), a clear separation is feasible, and model sparsity (few support vectors) is beneficial. Do not use when interpretability of coefficients is required (kernel SVM has black-box weights), when the data set is very large and linear models suffice, or when probability estimates are needed without calibration.

## Variants / Extensions
C-SVC (penalty C) and ν-SVC (parameter ν bounds fraction of support vectors and margin errors). One-class SVM for outlier detection. ε-SVR for regression. Weighted SVM for class imbalance. Approximations like linear SVM (LinearSVC) and kernel approximation (Nyström, random Fourier features) scale to massive datasets. Multi-class extensions use one-vs-one or one-vs-rest.

## Minimal Example (Python)
```python
from sklearn.svm import SVC, SVR
X = [[0,0], [1,1], [1,0], [0,1]]
y = [0, 1, 0, 1]
clf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X, y)
print(clf.predict([[0.5, 0.5]]))
reg = SVR(kernel='linear').fit(X, [1, 3, 2, 4])
print(reg.predict([[0.5, 0.5]]))
```