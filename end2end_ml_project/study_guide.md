# Machine Learning Study Guide: End-to-End Workflow

This guide summarizes the key concepts, rationale, and code snippets for the data processing phase of a Machine Learning project.

---

## Phase 1: Data Acquisition & Splitting

### 1. Data Loading

**Concept:** Automate data ingestion to ensure reproducibility across machines.

```text
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))
```

### 2. Random Splitting

**Concept:** Set aside a "Test Set" (final exam) that the model never sees during training.

```text
from sklearn.model_selection import train_test_split
# random_state=42 ensures the split is the same every time you run the code
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

### 3. Stratified Splitting

**Concept:** Ensures the Test Set has the same demographic proportions (e.g., income brackets) as the real world. Critical for small or skewed datasets.

```text
# 1. Create a category bin for stratification
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

# 2. Split based on that category
train_set, test_set = train_test_split(
    housing, stratify=housing["income_cat"], test_size=0.2, random_state=42
)
```

## Phase 2: Exploratory Data Analysis (EDA)

### 4. Inspection

**Concept:** Get the "shape" and "quality" of the data immediately.

```text
housing.info()      # Check for nulls and data types
housing.describe()  # Summary statistics (mean, std, min, max)
```

### 5. Geospatial Visualization

**Concept:** Plot 4 dimensions on a 2D screen (X=longitude, Y=latitude, Size=population, Color=price).

```text
housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing["population"] / 100,
    label="population",
    c="median_house_value",
    cmap="jet",
    colorbar=True
)
```

### 6. Correlation Analysis

**Concept:** Pearson's r (-1 to +1) checks linear relationships.

- +1: Perfect positive correlation
- -1: Perfect negative correlation
- 0: No linear correlation (non-linear relationships may still exist)

```text
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))
```

## Phase 3: Data Cleaning

### 7. Imputation (Handling Missing Data)

**Concept:** Fill empty cells (NaN) with a robust statistic such as the median. Better than deleting rows or zero-filling.

```text
from sklearn.impute import SimpleImputer

# 1. Define the imputer
imputer = SimpleImputer(strategy="median")

# 2. Fit (learn medians from training data)
imputer.fit(housing_num)

# 3. Transform (fill the NaNs)
X = imputer.transform(housing_num)
```

### 8. Handling Text Attributes

**Concept:** Translate text ("Inland", "Near Ocean") into numbers so models can use them.

- OrdinalEncoder: for genuinely ordered categories.
- OneHotEncoder: for nominal categories (creates binary switches).

```text
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing[["ocean_proximity"]])
# Returns a sparse matrix (efficient memory usage)
```

## Phase 4: Feature Scaling & Transformation

### 9. Min-Max Scaling (Normalization)

**Concept:** Squashes data into a fixed range (e.g., 0 to 1 or -1 to 1).

- Pros: Useful for neural networks.
- Cons: Sensitive to outliers.

```text
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
housing_scaled = scaler.fit_transform(housing_num)
```

### 10. Standard Scaling (Standardization)

**Concept:** Centers data at mean=0 with std=1.

- Pros: Robust to outliers compared to min-max; standard for linear models and SVMs.

```text
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
housing_scaled = scaler.fit_transform(housing_num)
```

### 11. Log Transformation

**Concept:** Fixes heavy-tailed/skewed distributions (e.g., population or income) so they look more normal.

```text
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# log1p calculates log(1 + x) to be safe with zero values
log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
population_log = log_transformer.transform(housing[["population"]])
```

### 12. RBF Similarity (Multimodal Transformation)

**Concept:** The "spotlight" approach: measure similarity to a specific value (e.g., age 35) to capture modes that linear features miss.

```text
from sklearn.metrics.pairwise import rbf_kernel

# Measures similarity to age 35 (gamma controls how "wide" the spotlight is)
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
```

### 13. Target Transformation

**Concept:** Scale the target (label) during training and automatically un-scale predictions so humans can read them.

```text
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model = TransformedTargetRegressor(
    regressor=LinearRegression(),
    transformer=StandardScaler()
)
# Automatically scales y before training and un-scales after predicting
model.fit(X_train, y_train)
```

### 14. Pipelines (Chaining Transformations and Estimators)

**Concept:** A Pipeline chains multiple preprocessing steps and a final estimator into a single object. This makes code cleaner, reduces the chance of data leakage, and integrates seamlessly with grid search and cross-validation.

- Benefits:
  - Keeps preprocessing and modeling steps together.
  - Ensures transforms are applied consistently to train/validation/test sets.
  - Works with scikit-learn search utilities (GridSearchCV, etc.).
  - Access intermediate steps via `pipeline.named_steps`.

```text
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Fit and predict like a single estimator
pipeline.fit(housing[["median_income"]], housing_labels)
predictions = pipeline.predict(housing[["median_income"]].iloc[:5])
print(predictions)
```

Note: If the pipeline's last step is an estimator (like LinearRegression), `pipeline.fit_transform` is not available — use `pipeline.named_steps` or remove the final estimator from the pipeline to get transformed features.

### 15. ColumnTransformer + Pipelines (Separate flows for different column types)

**Concept:** Different column types (numerical vs categorical) usually require different preprocessing. Use small pipelines for each type and combine them with `ColumnTransformer` so the entire preprocessing can be treated as one transformable object.

- Benefits:
  - Apply tailored transforms to specific columns.
  - Keep code modular and testable.
  - Use as the preprocessing step in a larger Pipeline (preprocessing + estimator).

```text
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])

full_pipeline = ColumnTransformer(transformers=[
    ('num', num_pipeline, housing.select_dtypes(include=[np.number]).columns),
    ('cat', cat_pipeline, ['ocean_proximity'])
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)
```

Tip: To retrieve human-readable feature names after a `ColumnTransformer`, use the `get_feature_names_out()` method on the transformer or pipeline where available (scikit-learn >=1.0).

### ClusterSimilarity (Custom Transformer) and Advanced Feature Creators

**Concept:**
- Sometimes builtin transformers aren't enough. You can create custom transformers by subclassing `BaseEstimator` and `TransformerMixin` so they behave like scikit-learn estimators. This keeps them compatible with Pipelines and ColumnTransformer.
- `ClusterSimilarity` in the repo is a custom transformer that:
  - Fits a KMeans model on geographic features (latitude/longitude).
  - Transforms each row into similarity scores to each cluster center using an RBF kernel.
  - Implements `get_feature_names_out` so downstream code can label the new features.

Why this is useful:
- Converts raw geo-coordinates into a richer, nonlinear representation (similarity to representative locations).
- The resulting features can help linear models capture complex spatial effects without manual feature engineering.

Example (conceptual):

```text
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        # Learn cluster centers using KMeans
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        # Return RBF similarity to each cluster center
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
```

Notes:
- Keep the transformer stateless except for learned attributes (e.g., `kmeans_`) so it serializes cleanly with joblib.
- Choose `gamma` to control the softness of similarity; smaller gamma => wider similarity.

### "All together": Ratio features, small pipelines, and the final ColumnTransformer

**Concept:** The repo builds several small pipelines and custom feature transformers, then composes them with a `ColumnTransformer` to get a single preprocessing object. Key pieces:

1. Ratio features (FunctionTransformer)
   - `column_ratio` computes pairwise ratios (e.g., bedrooms / rooms).
   - `FunctionTransformer` wraps the function and can provide `feature_names_out` via a helper.

2. Short pipelines built with `make_pipeline` or `Pipeline`
   - Numeric pipelines: impute (median) -> transform (log or ratio) -> scale.
   - Categorical pipelines: impute (most_frequent) -> OneHotEncode( handle_unknown='ignore').
   - Log pipeline: applies np.log then scales to normalize heavy-tailed features.

3. Combine everything with `ColumnTransformer` and `remainder` pipeline
   - Each named transformer receives specific columns (or selectors).
   - `remainder=default_num_pipeline` ensures any remaining numeric column(s) are still imputed/scaled.
   - After `fit_transform`, you get a dense (or sparse) matrix ready for modeling.

Concise example (from the repo):

```text
# Simple ratio helper used by FunctionTransformer
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

# Provide friendly feature names for the transformer
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

# A reusable small pipeline for ratio features
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

# Categorical pipeline
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

# Log pipeline for heavy-tailed numeric features
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())

# Custom cluster similarity transformer instantiation
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

# Default pipeline for any remaining numeric columns
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

# Final ColumnTransformer assembly
preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                           "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
],
    remainder=default_num_pipeline)  # e.g., housing_median_age

# Fit and transform
housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared.shape)
# Optional: inspect first rows, round for readability
print(housing_prepared[:5].round(2))
# Retrieve feature names (scikit-learn >= 1.0)
print(preprocessing.get_feature_names_out())
```

Why this pattern works well
- Each small pipeline focuses on a single concern (imputation, scaling, encoding).
- The `ColumnTransformer` keeps transformations column-aware and efficient.
- The final `preprocessing` object can be embedded as the first step of a larger Pipeline together with an estimator (e.g., `Pipeline([('preproc', preprocessing), ('clf', model)])`).

Tips and gotchas
- When using FunctionTransformer to create new features, supply `feature_names_out` so `get_feature_names_out()` works end-to-end.
- `OneHotEncoder(handle_unknown='ignore')` is recommended when your training categories might not cover all categories seen in production.
- If `ColumnTransformer` returns a sparse matrix, some estimators may not accept it — convert to dense only if memory allows.
- Always fit preprocessing on training data only and reuse for validation/test to avoid leakage.

---

## Model training & evaluation helpers

Below are concise explanations and usage examples for the training and evaluation helper functions defined in `prepare_data.py` (functions starting after the preprocessing assembly).

```text
train_linear_regression()
- Definition: Trains a linear regression pipeline that includes the preprocessing step.
  It fits the model on the full training set, prints first predictions, and reports RMSE/MSE.
- Use when: you want a fast, interpretable baseline model to compare against more complex models.
- Example:
  lin_reg = make_pipeline(preprocessing, LinearRegression())
  lin_reg.fit(housing, housing_labels)
  preds = lin_reg.predict(housing[:5])
```

```text
train_decision_tree()
- Definition: Trains a DecisionTreeRegressor inside a pipeline with preprocessing.
  Reports predictions and computes RMSE on the training set (trees often overfit without pruning).
- Use when: you need a nonlinear model that captures interactions but check for overfitting.
- Example:
  tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
  tree_reg.fit(housing, housing_labels)
  preds = tree_reg.predict(housing[:5])
```

```text
cross_val_score_with_decision_tree()
- Definition: Runs K-fold cross-validation (cv=10) for the decision tree pipeline,
  converts negative MSE scores to RMSE, and prints the distribution, mean, and stddev.
- Use when: you want a reliable estimate of generalization performance and variance.
- Example:
  scores = cross_val_score(tree_reg, housing, housing_labels, scoring='neg_mean_squared_error', cv=10)
  rmse_scores = np.sqrt(-scores)
```

```text
cross_val_score_with_linear_regression()
- Definition: Performs 10-fold CV for the linear regression pipeline, returning RMSE scores
  and summary statistics to compare with other models' CV performance.
- Use when: evaluating a linear baseline under cross-validation for robust comparison.
- Example:
  scores = cross_val_score(lin_reg, housing, housing_labels, scoring='neg_mean_squared_error', cv=10)
  rmse_scores = np.sqrt(-scores)
```

```text
cross_val_score_with_random_forest()
- Definition: Runs cross-validation for a RandomForestRegressor pipeline and reports RMSE stats.
- Use when: you want to evaluate a stronger ensemble model that typically yields lower error but costs more.
- Example:
  forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
  scores = cross_val_score(forest_reg, housing, housing_labels, scoring='neg_mean_squared_error', cv=10)
```

```text
grid_search_cv_with_decision_tree()
- Definition: Runs GridSearchCV over a Pipeline that includes preprocessing and a RandomForestRegressor.
  It tunes preprocessing hyperparameters (e.g., number of geo clusters) and model hyperparameters
  (e.g., max_features) using an explicit param_grid and CV folds; prints CV results.
- Use when: you want exhaustive search over a small, well-chosen grid of hyperparameters.
- Example param grid snippet:
  param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10], 'random_forest__max_features': [4,6,8]},
  ]
  grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
  grid_search.fit(housing, housing_labels)
```

```text
random_search_cv_with_random_forest()
- Definition: Uses RandomizedSearchCV to sample hyperparameter combinations from distributions
  (n_iter controls samples). Faster than grid search for large parameter spaces; useful to
  discover promising regions before a finer grid search.
- Use when: you have many hyperparameters or large candidate sets and need a budgeted search.
- Example param distributions snippet:
  param_distributions = {
    'preprocessing__geo__n_clusters': [5,8,10,15],
    'random_forest__max_features': [4,6,8,10],
    'random_forest__n_estimators': [50,100,150]
  }
  rand_search = RandomizedSearchCV(full_pipeline, param_distributions, n_iter=10, cv=3)
```

---

**Notes**

- In real pipelines, always fit preprocessing transforms on the training set only and apply them to validation/test sets.
- Use cross-validation and holdout sets to avoid data leakage.
- Small, clear visualizations and stats help detect problems early.

---

**References and Further Reading**

- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron)
- Scikit-Learn documentation: https://scikit-learn.org
