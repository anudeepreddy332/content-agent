# CatBoost

## Definition
CatBoost is a gradient boosting library that builds an ensemble of oblivious decision trees. It is distinguished by native handling of categorical features without manual preprocessing, using ordered target statistics with permutations to avoid target leakage, and employs an ordered boosting mechanism to reduce overfitting.

## Intuition
Traditional GBDT requires conversion of categorical features to numerical ones (one-hot, label encoding), which can cause combinatorial explosion or introduce arbitrary ordering. CatBoost encodes categories by computing target statistics: replacing a category with the mean target (or a smoothed variant) conditioned on that category. To prevent data leakage from using the same target to encode and predict, CatBoost uses permutations and lag-exclusion, encoding each instance using only past data within a random ordering. This yields a powerful, yet leakage-free, representation of categorical variables.

## Mathematical Formulation
Standard gradient boosting objective: add tree f_t to minimize sum L(y_i, F_{t-1}(x_i) + f_t(x_i)) + Omega(f_t). CatBoost uses oblivious trees, where the same splitting criterion and the same feature-threshold pair are applied to all nodes at a given depth. A tree of depth d has 2^d leaves.

For categorical feature k with value v, the ordered target statistic is:
TS_{v} = (sum_{i in D_k, before j} y_i + a * p) / (count_{i in D_k, before j} + a)
where j is the index in a random permutation, D_k is the set of objects with that category, “before j” restricts to preceding objects in the permutation, a is a smoothing strength, and p is the global average target. This ensures that the target for object j is encoded only using data preceding it.

The oblivious tree structure enables evaluation via flat index from binary features, making inference extremely fast.

## Estimation
A sequence of random permutations is drawn. For each permutation, target statistics are computed with per-instance leave-out. Trees are built greedily using these statistics and gradients. Hyperparameters: iterations, learning_rate, depth (tree depth; number of leaves = 2^depth), l2_leaf_reg, random_strength (amount of randomness in scoring splits), bagging_temperature, and one_hot_max_size for categories. Early stopping by validation set is recommended.

## Assumptions
The boosted model is an additive expansion of trees. The approximation with gradient descent is suitable; loss is differentiable. Categorical features contain meaningful signal exploitable via target encoding. No scaling or distributional assumptions on numeric features; missing values are handled by treating them as a special category or value.

## Key Properties
Top-tier predictive performance with minimal hyperparameter tuning (defaults work well). Native categorical feature handling reduces feature engineering. Ordered TS encoding effectively combats overfitting due to target leakage. Oblivious trees make inference fast and model size small. Built-in support for text features (since version 0.19). Provides feature importance and supports GPU training.

## Failure Modes
When target encoding for a category has too few observations, estimates are noisy; smoothing mitigates but does not eliminate. Oblivious trees may struggle to capture complex interactions that require different splits across nodes at same depth, limiting expressiveness relative to asymmetric trees. Very large number of categorical features can make permutation-based encoding memory intensive. With many iterations, overfitting still occurs if not stopped early.

## When to Use / Not Use
Use CatBoost when the data contains many categorical features, when quick strong baselines are needed, or when avoiding manual one-hot encoding. Excellent for competitions and production due to default robustness. Not ideal if the tree depth needs to model highly asymmetric interactions (consider XGBoost/LightGBM with asymmetric trees), or when domain-specific encoding already exists and is superior. Inference speed benefit of oblivious trees diminishes if extremely deep trees are required.

## Variants / Extensions
CatBoost for ranking (YetiRank, PairLogit), multitask regression, survival analysis. Integration with scikit-learn API. Text feature embeddings via transformers. Other GBDT libraries: LightGBM (leaf‑wise growth, gradient-based one-side sampling), XGBoost (level‑wise, regularized).

## Minimal Example (Python)
```python
from catboost import CatBoostClassifier
X = [['cat', 1], ['dog', 2], ['cat', 3], ['dog', 1]]
y = [0, 1, 0, 1]
model = CatBoostClassifier(iterations=10, depth=2, verbose=0,
                           cat_features=[0])
model.fit(X, y)
print(model.predict([['cat', 2]]))
```