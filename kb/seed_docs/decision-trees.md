# Decision Trees

## Definition
A decision tree is a non-parametric supervised learning model that recursively partitions the feature space into axis-aligned regions, assigning a constant prediction to each region. Classification trees predict the majority class; regression trees predict the mean target. The structure is a binary tree where each internal node tests a feature against a threshold, and each leaf holds a prediction.

## Intuition
The model repeatedly asks yes/no questions about the features. At each step, it splits the data to make the resulting subsets as homogeneous as possible with respect to the target. The final leaf nodes represent pure or nearly pure groups; predictions are formed by summarizing the training instances that fall into each leaf.

## Mathematical Formulation
Consider classification with K classes. Let p_k be the proportion of class k in a node. Common impurity measures:
- Gini impurity: G = 1 − Σ p_k²
- Entropy: H = − Σ p_k log₂(p_k)

For regression, impurity is mean squared error: MSE = (1/N) Σ (yᵢ − ȳ)², with ȳ the node mean.

A split divides node R into two child nodes R_left and R_right via a feature j and threshold s. The split is chosen to maximize the impurity reduction:
ΔI = I(parent) − (N_left/N) I(left) − (N_right/N) I(right)

The tree is grown greedily until a stopping criterion (min samples per leaf, max depth, or pure nodes). Post-hoc pruning (e.g., cost-complexity pruning) removes branches to minimize:
R_α(T) = Σ_leaf N_leaf · I(leaf) + α |leaves|

where α controls the trade-off. The final prediction for x is f(x) = mean or mode of training y in leaf containing x.

## Estimation
The standard CART algorithm uses a top-down greedy search over all features and possible thresholds. At each node, the best split is chosen. Continuous features are sorted and midpoint thresholds evaluated; categorical features are split by grouping categories. The process repeats until stopping criteria are met. Pruning is performed by weakest-link cutting, often using cross-validation to select α. Hyperparameters include max depth, min samples split, min samples leaf, and pruning constant.

## Assumptions
Piecewise constant target function; axis-aligned decision boundaries; training samples are i.i.d.; feature interactions that can be approximated by sequential univariate splits; noise does not dominate the signal. No distributional or scaling assumptions on features.

## Key Properties
Interpretability: tree structure directly shows decision rules. Handles both numerical and categorical variables without scaling or encoding. Captures non-linear relationships and interactions. Variable importance can be derived from total impurity reduction. High variance: small data changes can yield completely different trees. Tends to overfit, but controlled by pruning or ensemble methods.

## Failure Modes
Overfitting to noise without pruning/regularization. Instability: small perturbations cause large structural changes. Bias toward features with more levels or higher cardinality. Oblique decision boundaries are poorly approximated by axis-aligned splits, requiring many nodes. Regression trees cannot extrapolate beyond training range. Imbalanced data can bias splits unless adjustments are made.

## When to Use / Not Use
Use when interpretability is required, features are mixed-type, non-linearities exist, and interactions are suspected. Serve as base learners for ensemble methods. Do not use when smooth additive relationships dominate and a linear model would be more efficient; when extrapolation beyond training feature ranges is needed; or when very high-dimensional data makes trees deep and unstable without ensembles.

## Variants / Extensions
C4.5 (uses gain ratio, multi-way splits), CHAID (chi-square based multi-way splits), oblique trees (linear combinations at nodes), model trees (linear models at leaves), ensemble methods like Random Forests and Gradient Boosted Trees that reduce variance and bias, Bayesian trees, and evolutionary trees.

## Minimal Example (Python)
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
X = [[0,0], [1,1], [1,0], [0,1]]
y_clf = [0, 1, 0, 1]
y_reg = [0.1, 1.2, 0.3, 1.0]

clf = DecisionTreeClassifier(max_depth=2).fit(X, y_clf)
print(clf.predict([[0.5, 0.5]]))

reg = DecisionTreeRegressor(max_depth=2).fit(X, y_reg)
print(reg.predict([[0.5, 0.5]]))
```
## Claim-Dense Reference Facts

- scikit-learn's DecisionTreeClassifier uses CART algorithm; splits minimise weighted Gini impurity by default.
- Gini impurity: G = 1 - Σ p_i², where p_i is the proportion of class i at the node.
- Information gain uses entropy: H = -Σ p_i log₂(p_i); split selects feature maximising gain = H(parent) - weighted H(children).
- max_depth=None by default in scikit-learn, causing full tree growth until pure leaves (high overfitting risk).
- min_samples_split (default 2) and min_samples_leaf (default 1) are the primary pre-pruning controls.
- Post-pruning via cost-complexity pruning: ccp_alpha parameter; higher values prune more aggressively.
- Decision tree training is O(n · m · log n) where n = samples, m = features.
- Feature importance is computed as total impurity reduction weighted by node sample count, normalised to sum to 1.
- Regression trees minimise MSE; leaf prediction is the mean of all training samples reaching that leaf.
- Oblique splits (linear combinations of features) are not supported by scikit-learn CART; axis-aligned only.