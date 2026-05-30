# Linear and Logistic Regression

## Definition
Linear regression models a continuous response y as a linear function of predictors X plus Gaussian noise. Logistic regression models a binary target Y by estimating P(Y=1) as the logistic sigmoid applied to a linear combination of X.

## Intuition
Linear regression fits a line (or hyperplane) by minimizing the sum of squared vertical distances. Logistic regression maximizes the Bernoulli likelihood, mapping linear scores to probabilities via the sigmoid curve, which naturally bounds outputs in (0,1).

## Mathematical Formulation
Linear: y = Xβ + ε, with ε ~ N(0, σ²). Prediction: ŷ = Xβ.  
Logistic: log(P(Y=1)/(1−P(Y=1))) = Xβ, equivalently P(Y=1) = 1/(1 + exp(−Xβ)).

## Estimation
Linear: ordinary least squares (OLS) minimizes Σ (yᵢ − xᵢ·β)². Closed-form β̂ = (XᵀX)⁻¹Xᵀy (if invertible).  
Logistic: no closed-form. Maximum likelihood is found iteratively (Newton-Raphson, IRLS) maximizing ℓ = ∏ pᵢ^{yᵢ}(1−pᵢ)^{1−yᵢ}, with pᵢ = 1/(1+exp(−xᵢ·β)).

## Assumptions
Linear: linearity, independent errors, homoscedasticity; normality needed only for exact inference. Gauss‑Markov conditions (zero mean, uncorrelated, homoscedastic errors) suffice for BLUE.  
Logistic: independent binary outcomes, log-odds linear in X, no perfect or quasi-separation, adequate sample size per predictor; no error variance assumption.

## Key Properties
Linear: under Gauss-Markov, OLS is BLUE; predictions are unbounded; R² measures explained variance.  
Logistic: MLE is consistent and asymptotically normal; outputs are calibrated probabilities; coefficients are log-odds ratios; the decision boundary p=0.5 is the hyperplane Xβ=0.

## Failure Modes
Linear: heteroscedasticity inflates SEs, nonlinearity causes bias, outliers are influential, multicollinearity destabilizes coefficients, extrapolation gives misleading predictions.  
Logistic: complete separation yields infinite estimates, quasi-separation inflates errors, multicollinearity destabilizes, severe class imbalance biases the intercept, non-linear log-odds harms fit, miscalibration possible.

## When to Use / Not Use
Use linear regression for continuous unbounded outcomes when the mean is linear in predictors and coefficient interpretation is important. Avoid for proportions, counts, or binary data.  
Use logistic regression for binary responses requiring probability estimates and a roughly linear log-odds. Avoid for multi-class (use multinomial) or highly non-linear boundaries without proper transformations.

## Variants / Extensions
Linear: ridge (L2), lasso (L1), elastic net, generalized linear models (GLM), polynomial/spline expansions.  
Logistic: multinomial (softmax) for >2 classes, ordinal logistic, regularized (L1/L2) logistic, Firth’s penalized likelihood for small samples, Bayesian logistic.

## Minimal Example (Python)
```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

X = np.array([[1], [2], [3], [4]])
y_lin = np.array([2.2, 3.1, 4.8, 6.1])
y_log = np.array([0, 0, 1, 1])

lin = LinearRegression().fit(X, y_lin)
print(lin.coef_, lin.intercept_)

log = LogisticRegression().fit(X, y_log)
print(log.coef_, log.intercept_)
```
## Claim-Dense Reference Facts

- Scikit-learn `LinearRegression` uses scipy.linalg.lstsq (SVD‑based) by default; no regularization, intercept fitted unless `fit_intercept=False`.
- Scikit-learn `LogisticRegression` defaults: penalty='l2', C=1.0, solver='lbfgs', max_iter=100, multi_class='auto'.
- The logistic sigmoid function is σ(z) = 1 / (1 + e⁻ᶻ); its derivative is σ(z)(1−σ(z)).
- Odds = p/(1−p); log‑odds = ln(odds); each unit increase in x multiplies odds by exp(β).
- Logistic regression decision boundary: p=0.5 ↔ log‑odds=0 ↔ Xβ=0.
- `LogisticRegression` solvers: 'lbfgs' (L2 only or no penalty), 'liblinear' (L1/L2, small datasets), 'saga' (L1/L2, elasticnet, large datasets).
- ‘lbfgs’ solver handles multinomial loss; ‘liblinear’ only supports one‑vs‑rest.
- For `penalty='elasticnet'`, set `l1_ratio` (0=ridge, 1=lasso) and `solver='saga'`.
- C is inverse regularization strength: smaller C = stronger penalty; typical search range [0.001, 0.01, 0.1, 1, 10, 100].
- LogisticRegression `class_weight='balanced'` adjusts weights inversely proportional to class frequencies.
- Logistic regression predicts probabilities via `predict_proba()`; `predict()` returns the most likely class.
- LinearRegression `score()` returns R² (coefficient of determination), bounded ≤ 1, can be negative for poorly fitting models.
- Logistic regression typical evaluation: accuracy, precision, recall, F1, ROC‑AUC; `roc_auc_score(y_true, y_proba[:,1])`.
- Assumptions for linear regression coefficient tests (t‑tests): errors normally distributed; without normality, use bootstrap confidence intervals.
- Variance Inflation Factor (VIF) > 5–10 indicates multicollinearity among predictors.
- For perfect separation in logistic regression, Firth’s penalized likelihood or adding a small L2 penalty (increasing C) can stabilize.
- Scikit-learn `LinearRegression` does NOT natively handle missing values; pre‑impute with `SimpleImputer` or `IterativeImputer`.
- Polynomial features can be generated with `sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)`.
- Feature scaling (e.g., `StandardScaler`) is not required for linear regression but essential for regularized logistic regression.
- The normal equation β̂ = (XᵀX)⁻¹Xᵀy is O(p³) in time; for p > 10k features, use SGD or mini‑batch methods.