# Naive Bayes

## Definition
Naive Bayes is a family of probabilistic classifiers based on Bayes’ theorem, applying the “naive” assumption that features are conditionally independent given the class label. Different variants handle continuous, categorical, or count features by specifying appropriate per-feature likelihoods.

## Intuition
By combining the prior probability of each class with the likelihood of the features under that class, classification reduces to a product of per-feature contributions. The independence assumption dramatically reduces the number of parameters to estimate, making the model tractable even with many features and limited data, while often preserving a competitive decision boundary.

## Mathematical Formulation
Given class y and feature vector x = (x₁,…,xₚ), Bayes’ rule gives P(y|x) ∝ P(y) · P(x|y). Under conditional independence, P(x|y) = ∏ᵢ P(xᵢ|y). The decision rule is ŷ = argmax_y P(y) ∏ᵢ P(xᵢ|y). The likelihood form P(xᵢ|y) depends on the data type:
- Gaussian NB: P(xᵢ|y) = (1/√(2πσ²_{y,i})) exp(−(xᵢ − μ_{y,i})²/(2σ²_{y,i})) for continuous xᵢ.
- Multinomial NB: P(x|y) = ( (∑ᵢ xᵢ)! / ∏ᵢ xᵢ! ) ∏ᵢ p_{y,i}^{xᵢ}, modelling count vectors, often used with term frequencies.
- Bernoulli NB: each xᵢ ∈ {0,1}, with P(xᵢ|y) = p_{y,i}^{xᵢ} (1 − p_{y,i})^{1−xᵢ}.

## Estimation
Parameters are estimated via maximum likelihood (or Bayesian MAP with a smoothing prior). The class prior P(y) is the fraction of training samples in class y. For Gaussian NB, μ_{y,i} and σ²_{y,i} are the empirical within-class mean and variance. For multinomial/Bernoulli NB, the probabilities p_{y,i} are sample proportions smoothed with an additive parameter α (Laplace smoothing if α=1): p_{y,i} = (∑_{x∈y} xᵢ + α) / (∑_{i}(∑_{x∈y} xᵢ + α)). Smoothing prevents zero estimates for unobserved feature values.

## Assumptions
Features are conditionally independent given the class (the naive assumption). The chosen parametric form (e.g., Gaussian) is appropriate for each feature within a class. Observations are independent and identically distributed. No missing features; handling of missingness requires additional marginalization.

## Key Properties
Training and prediction are extremely fast (linear in number of features, instances, and classes). The naive independence assumption biases probability estimates but often preserves the ranking of classes, yielding a correct maximum a posteriori decision even with feature correlations. Handles high-dimensional data and small samples robustly. Interpretable: per-feature class-conditional parameters give direct insight. Smoothing guarantees non-zero likelihoods. No hyperparameter tuning beyond smoothing α in basic form.

## Failure Modes
Severe violation of conditional independence (e.g., highly redundant features) can inflate confidence and degrade probability calibration while possibly keeping classification accurate. Continuous features that are multimodal or heavy-tailed violate the Gaussian assumption, harming performance. Strongly overlapping classes with similar feature means yield poor separation. Zero-frequency problem without smoothing yields undefined predictions. Cannot model feature interactions directly.

## When to Use / Not Use
Use as a strong baseline for text classification (spam filtering, topic categorization), real-time applications requiring fast online updates, and when probabilistic outputs are needed. Not suited when feature interactions are crucial (e.g., image pixels) and a linear/non-linear discriminative model would be better, or when precise probability estimation is required and independence is grossly violated.

## Variants / Extensions
Gaussian NB for continuous data, Multinomial NB for term frequencies, Bernoulli NB for binary occurrences, Complement NB for imbalanced text, Categorical NB for categorical features. Tree-Augmented NB relaxes independence by adding a tree structure among features. AODE (Averaged One-Dependence Estimators) averages over all possible one-dependence structures. Semi-supervised NB uses unlabeled data for parameter estimation.

## Minimal Example (Python)
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([0, 0, 1, 1])
gnb = GaussianNB().fit(X, y)
print(gnb.predict([[0, 0]]))

X_counts = np.array([[0,1,1], [1,0,1], [1,2,0], [2,1,0]])
y = np.array([0, 0, 1, 1])
mnb = MultinomialNB(alpha=1).fit(X_counts, y)
print(mnb.predict([[0, 2, 1]]))
```