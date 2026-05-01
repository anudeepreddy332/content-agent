# Ridge, Lasso & ElasticNet

## Definition
Ridge, Lasso, and ElasticNet are regularized linear regression methods that add a penalty to the ordinary least squares objective. Ridge uses an L2 penalty, Lasso an L1 penalty, and ElasticNet a convex combination of both, controlling coefficient magnitude and, for Lasso/ElasticNet, inducing sparsity.

## Intuition
Unpenalized OLS can overfit when predictors are many or correlated. Shrinking coefficients toward zero reduces variance at the cost of increased bias. The L1 penalty can push coefficients exactly to zero (feature selection); L2 distributes shrinkage across all coefficients. ElasticNet blends L1 and L2 to select variables even when groups of predictors are strongly correlated, where Lasso would select only one arbitrarily.

## Mathematical Formulation
Minimize over β:
- Ridge: Σ(yᵢ − xᵢ·β)² + α Σⱼ βⱼ²
- Lasso: Σ(yᵢ − xᵢ·β)² + α Σⱼ |βⱼ|
- ElasticNet: Σ(yᵢ − xᵢ·β)² + α [ ρ Σⱼ |βⱼ| + 0.5(1−ρ) Σⱼ βⱼ² ]
Here α ≥ 0 controls overall penalty strength, ρ ∈ [0,1] balances L1 vs L2 ratio.

## Estimation
Ridge has closed-form β̂ = (XᵀX + αI)⁻¹Xᵀy. Lasso and ElasticNet generally lack closed-form solutions and are estimated via coordinate descent, LARS, or proximal gradient methods. All typically include an intercept term not penalized, and predictors are standardized before fitting to make the penalty scale-invariant.

## Assumptions
Same structural assumptions as OLS (linearity, independent errors, homoscedasticity) for consistent estimation, but regularization mitigates collinearity issues. Normality of errors is not required for prediction. The penalty parameter α is typically chosen via cross-validation.

## Key Properties
Ridge shrinks coefficients but never sets them to zero, is rotationally invariant, and handles p > n. Lasso performs both shrinkage and variable selection; it can produce sparse coefficient vectors but selects at most n variables when p > n. ElasticNet overcomes Lasso’s variable-selection instability when predictors are highly correlated, encouraging a grouping effect where correlated predictors have similar coefficients. All three embody the bias–variance trade-off, with larger α reducing variance and increasing bias.

## Failure Modes
Ridge cannot perform feature selection; all predictors remain, harming interpretability. Lasso may arbitrarily pick one from a set of highly correlated predictors, and its selection is unstable with small data. Lasso fails when true model is not sparse or when p ≫ n and n is small (saturated). ElasticNet requires tuning two hyperparameters, increasing computational cost. All methods are sensitive to α choice; inadequate cross-validation leads to under- or over-regularization. They are also sensitive to predictor scaling.

## When to Use / Not Use
Use when OLS overfits, collinearity exists, or p is close to or exceeds n. Prefer Lasso for interpretable sparse models. Prefer Ridge when all predictors have small effects. Prefer ElasticNet when predictors are many and correlated. Do not use if inference on original full coefficients without bias is primary; ridge/Lasso coefficients are biased. Not suited for non-linear relationships without basis expansion.

## Variants / Extensions
Group Lasso (penalty on groups of coefficients), adaptive Lasso (weighted L1 for oracle properties), fused Lasso, relaxed Lasso, and ElasticNet with cross-validation for both α and ρ. Extensions to generalized linear models (GLMNet) apply the same penalties to logistic and Poisson regression.

## Minimal Example (Python)
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
X = [[1,2],[2,3],[4,1],[5,5]]
y = [3.5, 5.2, 5.1, 10.2]
ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.1).fit(X, y)
enet = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
print(ridge.coef_, lasso.coef_, enet.coef_)
```