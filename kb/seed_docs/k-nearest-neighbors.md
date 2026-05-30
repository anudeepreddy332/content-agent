# K-Nearest Neighbors

## Definition
K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning algorithm. For a query point, it identifies the k closest training instances according to a chosen distance metric. Classification uses majority vote among these neighbors; regression uses the average of their target values. No explicit training phase occurs; all computation is deferred until prediction time (lazy learning).

## Intuition
Predict the label of a new point by looking at the labels of its nearest neighbors. Nearness in feature space implies similarity. The hyperparameter k controls the locality of the decision: a small k captures fine details but is sensitive to noise, while a large k averages over a wider region, producing a smoother boundary. The underlying assumption is that the target function is locally smooth.

## Mathematical Formulation
Given training set D = {(x_i, y_i)}, a distance metric d (e.g., Euclidean), and a query x, define N_k(x) as the indices of the k training points with smallest d(x, x_i).

Classification: predicted class ŷ = argmax_y Σ_{i∈N_k(x)} I(y_i = y).  
Regression: ŷ = (1/k) Σ_{i∈N_k(x)} y_i.  
Weighted variants use weights w_i = 1 / d(x, x_i) or a kernel, then ŷ = Σ w_i y_i / Σ w_i.

## Estimation
No model parameters are estimated in a traditional sense. The algorithm’s performance depends on the choice of k, distance metric, and any feature weighting. These hyperparameters are typically tuned via cross-validation on the training data. Choosing a smaller k reduces bias but increases variance, while a larger k does the opposite. Feature scaling (e.g., standardization) is essential to prevent certain features from dominating the distance calculation.

## Assumptions
Locally constant or slowly varying target function. Independence and representativeness of training samples. Equal relevance of all features to the distance metric unless explicitly weighted. The distance metric must reflect meaningful similarity in the data space. No assumptions about the global distribution of the data.

## Key Properties
Non-parametric: no functional form is imposed. Lazy learning: training is merely storing the dataset. The decision boundary is a Voronoi tessellation (for k=1) or a smoothed version. As training size n → ∞ with k → ∞ and k/n → 0, the risk converges to the Bayes risk (consistent, under certain conditions). Query time complexity is O(nd) for brute-force; memory footprint is O(nd). Suffers from the curse of dimensionality: data sparsity makes local neighborhoods nearly empty in high dimensions.

## Failure Modes
Curse of dimensionality: in high-dimensional spaces, all points become nearly equidistant, so the notion of “nearest” degrades. Large n makes prediction slow without spatial indexing. Irrelevant or noisy features distort distances, harming prediction. Imbalanced classes can cause the majority class to dominate smaller neighborhoods. Outliers heavily influence small k. Sparse regions yield unreliable extrapolation. Sensitive to the choice of distance metric and feature scaling.

## When to Use / Not Use
Use for small-to-medium-sized, low-dimensional datasets where a simple, interpretable baseline is needed, or when local explanations are valuable. Do not use for high-dimensional data (e.g., text, images, hundreds of features), ultra-large training sets with real-time latency requirements, or when feature importance is non-uniform and unknown without additional weighting. Not suited for extrapolation far from training data.

## Variants / Extensions
Weighted KNN uses distance-based weights to reduce the influence of far neighbors. Radius-based neighbors uses all points within a fixed radius. Data reduction methods (condensed nearest neighbor, edited nearest neighbor) remove noisy or redundant instances. Approximate nearest neighbor algorithms (k-d trees, ball trees, locality-sensitive hashing) accelerate search for large datasets. KNN regression with local linear or polynomial fitting. Multi-label KNN, Bayesian KNN.

## Minimal Example (Python)
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
X = [[0], [1], [2], [3]]
y_clf = [0, 0, 1, 1]
y_reg = [0, 0.1, 2, 2.1]

clf = KNeighborsClassifier(n_neighbors=3).fit(X, y_clf)
print(clf.predict([[1.5]]))

reg = KNeighborsRegressor(n_neighbors=2).fit(X, y_reg)
print(reg.predict([[1.5]]))
```
## Claim-Dense Reference Facts

- scikit-learn's KNeighborsClassifier defaults to n_neighbors=5, metric='minkowski' with p=2 (Euclidean).
- KD-tree query time is O(log N) for d < 20 dimensions; for d > 20, brute-force becomes competitive.
- Ball tree handles high-dimensional data better than KD-tree; scikit-learn auto-selects via algorithm='auto'.
- Weighted KNN uses w = 1/distance; uniform voting uses w = 1 for all k neighbors.
- Optimal k is typically found via cross-validation; small k overfits, large k underfits (high bias).
- For binary classification, k is chosen odd to avoid ties; for k=1, training accuracy is always 100%.
- Curse of dimensionality: in d=100 dimensions, the ratio of max to min distance between points approaches 1.
- Manhattan distance (L1) is preferred over Euclidean (L2) for high-dimensional sparse data like text.
- Normalisation (MinMaxScaler or StandardScaler) is required before KNN; unnormalized features with large ranges dominate distance.
- scikit-learn's BallTree and KDTree support leaf_size parameter (default 30) trading memory for query speed.