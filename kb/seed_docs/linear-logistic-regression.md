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