# XGBoost

## Definition
XGBoost (eXtreme Gradient Boosting) is a scalable machine learning framework that builds an ensemble of regression trees via gradient boosting. It uses second-order Taylor approximation of the loss function, explicit regularization, and system optimizations to achieve high performance and speed.

## Intuition
Trees are added sequentially; each new tree is trained to reduce the residual error of the current ensemble. Instead of fitting raw residuals, XGBoost fits the gradient and curvature of the loss, enabling custom loss functions and faster convergence. Complexity penalties on number of leaves and leaf weights prevent overfitting, while shrinkage (learning rate) makes boosting more robust.

## Mathematical Formulation
The t-th tree f_t is chosen to minimize an objective function at step t:

Obj = sum_i L(y_i, y_hat_i^(t-1) + f_t(x_i)) + Omega(f_t)

L is approximated via second-order Taylor expansion around y_hat^(t-1):

L(y_i, y_hat^(t-1) + f_t(x_i)) ≈ L(y_i, y_hat^(t-1)) + g_i f_t(x_i) + 0.5 h_i [f_t(x_i)]^2

where g_i = ∂L/∂y_hat and h_i = ∂²L/∂y_hat² (gradient and hessian). Regularization Omega(f_t) = gamma T + 0.5 lambda ||w||², with T leaves and leaf weights w.

For a fixed tree structure, optimal leaf weight for leaf j:

w_j* = - (sum_{i in leaf j} g_i) / (sum_{i in leaf j} h_i + lambda)

The resulting optimal objective reduction (split gain) is:

Gain = 0.5 [ (G_L)^2/(H_L + lambda) + (G_R)^2/(H_R + lambda) - (G)^2/(H + lambda) ] - gamma

A node is split if gain > 0 (gamma acts as complexity threshold). Tree is grown greedily using this gain.

## Estimation
Additive training uses approximate splitting via weighted quantile sketch on sorted gradient statistics. Hyperparameters: learning rate (eta) shrinks new tree weights; max_depth controls tree size; subsample, colsample_bytree introduce stochasticity; lambda (L2), alpha (L1) and gamma regulate complexity. Early stopping on a validation set is common.

## Assumptions
Additive expansion of trees; twice-differentiable convex loss for Taylor expansion; base learners are shallow decision trees; no distributional or scaling assumptions on features. Missing values are handled by learning a default direction during training.

## Key Properties
Extremely fast tree building with parallelized split finding and out-of-core computation. Handles missing values natively. Built-in L1/L2 regularization mitigates overfitting. Feature importance from gain, coverage, or frequency. Works with many loss functions (regression, classification, ranking). Robust to unscaled data.

## Failure Modes
Sensitive to hyperparameters; overfits with too many rounds, deep trees, or low regularization. Overly high gamma can halt splitting prematurely. May not extrapolate beyond training range. Not interpretable as a single tree; ensemble nature obscures decision logic. Potential for target leakage in complex feature engineering pipelines.

## When to Use / Not Use
Use for structured/tabular data where predictive accuracy is paramount, for competitions, and when feature importance insights are desired. Not use when model interpretability is critical (white-box decisions), when the dataset is tiny, or when testing with extreme input values outside observed ranges.

## Variants / Extensions
LightGBM (gradient-based one-side sampling, leaf-wise growth), CatBoost (ordered boosting, symmetric trees), XGBoost for ranking, monotonic constraints, GPU-accelerated training, and DART (dropouts in boosting).

## Minimal Example (Python)
```python
import xgboost as xgb
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=4, random_state=0)
model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
model.fit(X, y)
print(model.predict(X[:2]))
```