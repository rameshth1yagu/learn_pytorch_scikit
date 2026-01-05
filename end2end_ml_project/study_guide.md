# Machine Learning Study Guide: End-to-End Workflow

This guide summarizes the key concepts, rationale, and code snippets for the data processing phase of a Machine Learning project.

---

## Phase 1: Data Acquisition & Splitting

### 1. Data Loading

**Concept:** Automate data ingestion to ensure reproducibility across machines.

```python
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

```python
from sklearn.model_selection import train_test_split
# random_state=42 ensures the split is the same every time you run the code
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

### 3. Stratified Splitting

**Concept:** Ensures the Test Set has the same demographic proportions (e.g., income brackets) as the real world. Critical for small or skewed datasets.

```python
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

```python
housing.info()      # Check for nulls and data types
housing.describe()  # Summary statistics (mean, std, min, max)
```

### 5. Geospatial Visualization

**Concept:** Plot 4 dimensions on a 2D screen (X=longitude, Y=latitude, Size=population, Color=price).

```python
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

```python
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))
```

## Phase 3: Data Cleaning

### 7. Imputation (Handling Missing Data)

**Concept:** Fill empty cells (NaN) with a robust statistic such as the median. Better than deleting rows or zero-filling.

```python
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

```python
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

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
housing_scaled = scaler.fit_transform(housing_num)
```

### 10. Standard Scaling (Standardization)

**Concept:** Centers data at mean=0 with std=1.

- Pros: Robust to outliers compared to min-max; standard for linear models and SVMs.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
housing_scaled = scaler.fit_transform(housing_num)
```

### 11. Log Transformation

**Concept:** Fixes heavy-tailed/skewed distributions (e.g., population or income) so they look more normal.

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# log1p calculates log(1 + x) to be safe with zero values
log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
population_log = log_transformer.transform(housing[["population"]])
```

### 12. RBF Similarity (Multimodal Transformation)

**Concept:** The "spotlight" approach: measure similarity to a specific value (e.g., age 35) to capture modes that linear features miss.

```python
from sklearn.metrics.pairwise import rbf_kernel

# Measures similarity to age 35 (gamma controls how "wide" the spotlight is)
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
```

### 13. Target Transformation

**Concept:** Scale the target (label) during training and automatically un-scale predictions so humans can read them.

```python
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

---

**Notes**

- In real pipelines, always fit preprocessing transforms on the training set only and apply them to validation/test sets.
- Use cross-validation and holdout sets to avoid data leakage.
- Small, clear visualizations and stats help detect problems early.

---

**References and Further Reading**

- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron)
- Scikit-Learn documentation: https://scikit-learn.org
