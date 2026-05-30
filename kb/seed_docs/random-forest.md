# Random Forest

## Definition
Random Forest is an ensemble method that constructs multiple decision trees and combines them by majority vote (classification) or averaging (regression). Randomness is injected through bootstrap resampling of the training data and random feature subsets at each split, decorrelating the trees and reducing variance.

## Intuition
A single decision tree has low bias but high variance. Averaging many nearly unbiased but noisy trees reduces variance without increasing bias, provided the trees are not all identical. The double randomization (bootstrap samples and random feature subsets) ensures diversity among trees, so their errors cancel out.

## Mathematical Formulation
For i = 1 to B trees:
1. Draw bootstrap sample D_i of size N from training data D.
2. Grow tree T_i on D_i: at each node, select m ≤ p features at random, choose the best split among those m.
Final prediction:
- Classification: ŷ = majority vote {T_i(x)}.
- Regression: ŷ = (1/B) Σ T_i(x).

No explicit objective function is optimized globally; the ensemble aggregates identically distributed but not independent trees. The generalization error for classification converges to an upper bound involving the strength (accuracy) of individual trees and their correlation (dependence).

## Estimation
Hyperparameters: number of trees B, number of features sampled at each split m, and parameters controlling individual tree growth (e.g., max depth, min samples leaf). Typical defaults: m = sqrt(p) for classification, m = p/3 for regression. B is set large enough for error to stabilize. No pruning is applied; individual trees are grown deep. Out-of-bag (OOB) samples — observations not in a tree's bootstrap — provide an unbiased estimate of generalization error. Variable importance is estimated from OOB error increase when permuting a feature or from total impurity reduction.

## Assumptions
Independence of errors across trees is approximated via randomization. Underlying true function is sufficiently smooth to be approximated by piecewise constant functions. Features carry signal for the target; no strong assumption underlies functional form, error distribution, or scaling. Observations are i.i.d., but bootstrap induces dependence. Enough trees are grown to stabilize the average.

## Key Properties
High predictive accuracy, often with minimal tuning. Inherently handles mixed-type predictors and missing data. Provides OOB error and variable importance measures. Robust to outliers and noise. Parallelizable: each tree is grown independently. Less interpretable than a single tree but more stable. Tends not to overfit with increasing B, but individual deep trees may still overfit bootstrap samples, though aggregation mitigates this.

## Failure Modes
Performs poorly on high-cardinality noisy features that can overfit individual trees if m is not small enough. Can be outperformed by boosting when bias is high (e.g., underfitting weak learners). High computational cost for very large B or deep trees on big data. Extrapolation in regression is limited by the range of training responses within leaves. Sensitive to data with many irrelevant features if feature sampling is too large.

## When to Use / Not Use
Use as a strong general-purpose classifier/regressor, especially with tabular data, many features, and little domain knowledge to design parametric models. Good for feature selection via importance scores. Not use when interpretability of the model is paramount, when inference on coefficients is required, or when the data is highly structured in a way better suited to deep learning (e.g., images, sequences) with ample samples.

## Variants / Extensions
Extremely Randomized Trees (ExtraTrees): randomizes split thresholds rather than searching for the optimum, often faster and sometimes lower variance. Quantile Random Forest: predicts conditional quantiles. Causal Forest for heterogeneous treatment effects. Isolation Forest for anomaly detection based on path lengths. Rotation Forest using PCA on feature subsets. Random survival forests for time-to-event data.

## Minimal Example (Python)
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
X = [[0,0], [1,1], [1,0], [0,1], [0.5,0.5]]
y_clf = [0, 1, 0, 1, 0]
y_reg = [0, 1.2, 0.1, 0.9, 0.5]

rf_clf = RandomForestClassifier(n_estimators=10, max_depth=3).fit(X, y_clf)
print(rf_clf.predict([[0.3, 0.3]]))

rf_reg = RandomForestRegressor(n_estimators=10).fit(X, y_reg)
print(rf_reg.predict([[0.3, 0.3]]))
```
## Claim-Dense Reference Facts

- scikit-learn's RandomForestClassifier defaults: n_estimators=100, max_features='sqrt', bootstrap=True.
- Bootstrap sampling (bagging) trains each tree on a random sample with replacement; ~63.2% of samples are in-bag.
- Out-of-bag (OOB) error uses the ~36.8% of samples not selected for each tree as a free validation estimate.
- max_features='sqrt' uses √m features per split for classification; 'm/3' is recommended for regression.
- Feature importance in Random Forest is mean decrease in impurity (MDI) across all trees, normalised to sum to 1.
- Permutation importance is a more reliable alternative: measures accuracy drop when a feature's values are shuffled.
- Bias-variance tradeoff: individual deep trees have low bias but high variance; averaging reduces variance.
- n_jobs=-1 enables parallel tree training using all CPU cores; scales near-linearly up to ~32 cores.
- Random Forest handles missing values poorly by default; use iterative imputation or median fill before training.
- Extremely Randomised Trees (ExtraTreesClassifier) further randomizes split thresholds, reducing variance at cost of bias.