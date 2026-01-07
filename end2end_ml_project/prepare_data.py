from scipy.signal import correlate
import get_data
from sklearn.impute import SimpleImputer#, KNNImputer, IterativeImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import set_config
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# --- SETUP PHASE ---
# 1. Load the data using our stratification helper to ensure a representative sample.
train_set, _ = get_data.train_test_split_with_stratification()

# 2. Separate Predictors (Features) from Labels (Targets).
# We drop "median_house_value" because we don't want the model to see the answer key.
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()


def clean_data_options():
    """
    CONCEPT: Handling Missing Data (The 3 Options)
    Real-world data is dirty. 'total_bedrooms' has missing values (NaN).
    We have 3 standard strategies to fix this:
    """
    # Option 1: Nuclear Option. Delete any row with a missing value.
    # Risk: You lose valuable data from other columns just because one cell is empty.
    housing.dropna(subset=["total_bedrooms"], inplace=True)

    # Option 2: amputation. Delete the entire column.
    # Risk: You lose a potentially useful feature ("Bedrooms") entirely.
    housing.drop("total_bedrooms", axis=1, inplace=True)

    # Option 3: Imputation (Best Practice). Fill the holes with a reasonable guess.
    # We use the Median because it is robust against outliers (unlike Mean).
    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)

def sklearn_imputation(data, enable_hist: bool = False):
    """
    CONCEPT: Automated Imputation
    Using Scikit-Learn's SimpleImputer is better than manual Pandas filling because:
    1. It remembers the median from the Training Set.
    2. It can apply that EXACT median to the Test Set later (Consistency).
    """
    # Strategy="median": Calculate the middle value.
    imputer = SimpleImputer(strategy="median")

    # Imputation only works on Numbers, so we must exclude text columns.
    data_numeric = data.select_dtypes(include=[np.number])

    # Estimator Phase (Fit): Learn the medians from the data.
    imputer.fit(data_numeric)

    # Transformer Phase (Transform): Fill in the missing values.
    # Returns a plain NumPy array (loses column names).
    X = imputer.transform(data_numeric)

    # Inspection: Verify what the imputer actually learned.
    print(f"Imputer's hyperparameter - imputer.strategy: {imputer.strategy}")
    print(f"Imputer's learned parameters (The Medians): {imputer.statistics_}")

    # Validation: Compare with manual Pandas calculation to ensure correctness.
    print(f"Median by pandas: {data_numeric.median().values}")

    # Restoration: Convert the raw NumPy array back into a readable DataFrame.
    x_data_frame = pd.DataFrame(X, columns=data_numeric.columns, index=data_numeric.index)
    x_data_frame.info()

    if enable_hist:
        x_data_frame.hist(bins=50, figsize=(12, 8))

    return x_data_frame, X

def text_attributes(data):
    """
    CONCEPT: Encoding Categorical Data
    Machine Learning models only understand numbers. We must translate text ("Ocean")
    into math.
    """
    print("\nText Attributes")
    housing_cat = data[["ocean_proximity"]]
    print(housing_cat.head(10))

    # Method 1: Ordinal Encoding (0, 1, 2, 3)
    # Problem: The model assumes 0 and 1 are "similar" and 0 and 4 are "different".
    # This is bad for "Ocean Proximity" because the categories aren't ordered.
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print(f"OrdinalEncoder's categories_: {ordinal_encoder.categories_}")
    print(housing_cat_encoded[:10])

    # Method 2: One-Hot Encoding (Binary Switches) [Recommended]
    # Creates separate columns for each category (e.g., "Is_Island?", "Is_Inland?").
    # This prevents the model from assuming false relationships between categories.
    one_hot_encoder = OneHotEncoder()
    housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat)

    # Note: Returns a Sparse Matrix (saves memory by not storing zeros).
    print(f"OneHotEncoder's categories_: {one_hot_encoder.categories_}")
    print(housing_cat_1hot.toarray()[:10]) # .toarray() forces it to display dense matrix

    # Pandas get_dummies: Quick & Dirty version of OneHotEncoder.
    # Good for analysis, but bad for pipelines because it doesn't "remember" columns.
    print(f"pd.get_dummies: {pd.get_dummies(housing['ocean_proximity'])}")

def null_entries(data):
    """
    Utility: Locates specific rows that contain nulls so we can inspect them.
    """
    null_rows_idx = data.isnull().any(axis=1)
    print(f"Null rows sample:\n{data[null_rows_idx].head()}")

# --- EXECUTION: CLEANING ---
housing_tr, X = sklearn_imputation(housing)
text_attributes(housing)
null_entries(housing)

# --- FEATURE SCALING SECTIONS ---

def min_max_scaling(data):
    """
    CONCEPT: Normalization (Min-Max Scaling)
    Squashes all values strictly between -1 and 1 (or 0 and 1).
    Pros: Great for Neural Networks.
    Cons: Very sensitive to outliers. A single billionaire crushes everyone else to 0.
    """
    print(f"\n Min Max Scaling")
    min_max_scaler = MinMaxScaler(feature_range=(-1,1))
    data_min_max_scaled = min_max_scaler.fit_transform(data)

    print(f"Sample scaled data: {data_min_max_scaled[:1]}")
    data_frame = pd.DataFrame(data_min_max_scaled, columns=data.columns)
    # Notice how the X-axis on the histogram is now -1 to 1
    data_frame.hist(bins=50, figsize=(12, 8))

def standard_scaling(data):
    """
    CONCEPT: Standardization (Z-Score Scaling)
    Centers data around 0 with a standard deviation of 1.
    Pros: Handles outliers much better than MinMax. Best for Linear Regression/SVM.
    Cons: Does not guarantee a fixed min/max range.
    """
    standard_scaler = StandardScaler()
    data_standard_scaled = standard_scaler.fit_transform(data)
    data_frame = pd.DataFrame(data_standard_scaled, columns=data.columns)
    data_frame.hist(bins=50, figsize=(12, 8))

def standard_scaling_with_mean(data):
    """
    Variation: Scaling for Sparse Matrices.
    If 'with_mean=False', it doesn't center around 0.
    This is required for Sparse Matrices (lots of zeros) to preserve the zeros
    and keep memory usage low.
    """
    standard_scaler = StandardScaler(with_mean=False)
    data_standard_scaled = standard_scaler.fit_transform(data)
    data_frame = pd.DataFrame(data_standard_scaled, columns=data.columns)
    data_frame.hist(bins=50, figsize=(12, 8))

def log_transform(data):
    """
    CONCEPT: Handling Skewed Data (Log Transformation)
    Many natural phenomena (population, income) follow a "Power Law" (Long Tail).
    Models hate this. Logarithms squash the tail and make the distribution
    look like a nice, normal Bell Curve.
    """
    # np.log1p calculates log(1+x) to avoid "Divide by Zero" errors if data is 0.
    log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

    data_log_transformed = log_transformer.transform(data)
    data_frame = pd.DataFrame(data_log_transformed, columns=data.columns)

    # Visualization: Compare "Before" vs "After"
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

    # Original: Skewed to the left (Long Tail)
    data["population"].hist(ax=axes[0], bins=50)
    axes[0].set_title("Original Data (Skewed)")

    # Transformed: Symmetrical (Bell Curve) - The model loves this!
    data_frame["population"].hist(bins=50, ax=axes[1])
    axes[1].set_title("Log Transformed Data (Normal)")

# --- ADVANCED FEATURE ENGINEERING ---

def rbf_kernel_char():
    """
    CONCEPT: Similarity Features (RBF Kernel)
    Visualizes how the RBF function works. It acts like a "Spotlight".
    It highlights houses that are exactly 35 years old and ignores others.
    Gamma controls the width of the spotlight.
    """
    # Create dummy data range (0 to 52 years) to plot the smooth curve
    ages = np.linspace(housing["housing_median_age"].min(),
                       housing["housing_median_age"].max(),
                       500).reshape(-1, 1)

    gamma1 = 0.1   # Narrow spotlight (High specificity)
    gamma2 = 0.03  # Wide spotlight (General similarity)

    # Calculate similarity scores for every age against the landmark "35"
    rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
    rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the original Histogram of Ages
    ax1.set_xlabel("Housing median age")
    ax1.set_ylabel("Number of districts")
    ax1.hist(housing["housing_median_age"], bins=50, alpha=0.7)

    # Plot the RBF Curves on top (using a secondary Y-axis)
    ax2 = ax1.twinx()
    color = "blue"
    ax2.plot(ages, rbf1, color=color, label="gamma = 0.10 (Narrow)")
    ax2.plot(ages, rbf2, color=color, label="gamma = 0.03 (Wide)", linestyle="--")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel("Age similarity", color=color)

    plt.legend(loc="upper left")
    plt.title("RBF Similarity: Highlighting the 35-Year-Old Peak")

def transforming_multimodal_distribution_using_rbf_kernel(data):
    """
    CONCEPT: Solving Multimodal Distributions
    The age data has "bumps" (Modes) at different ages. A linear model can't see this.
    We add a new column 'age_35' which measures "Closeness to 35".
    This helps the model learn: "If house is ~35 years old, price is lower."
    """
    print("\nTransforming Multimodal Distribution using RBF Kernel")
    # Add the new feature to the dataset
    housing["age_35"] = rbf_kernel(data[["housing_median_age"]], [[35]], gamma=0.1)

    # Visualization
    rbf_kernel_char()


def label_scaling_example():
    """
    Demonstration: scale the target (labels), train a simple model on the scaled labels,
    then invert the scaling to get predictions back in the original label units.

    Purpose / Concept:
    - Sometimes you want to scale the target variable (labels) before training a model,
      for example when the target has a very different scale than the features or when
      a model trains more stably with zero-mean targets (e.g., for some optimization setups).

    Contract:
    - Inputs (closed-over globals):
      - `housing` (DataFrame): predictors/features including the `median_income` column.
      - `housing_labels` (Series): original target values (median_house_value).
    - Outputs:
      - Returns NumPy array `predictions` containing the model predictions transformed back
        into the original label scale (shape: (n_samples, 1) for this demo slice).
    - Error modes / notes:
      - This demo fits the scaler on the training labels directly (as shown). In a real
        pipeline, fit the scaler only on the training split and reuse it on validation/test.
      - The sklearn LinearRegression `predict` returns a 2D array when trained on a 2D target
        (shape: [n_samples, n_targets]). We preserve that shape so `inverse_transform` works.
    """

    # Create a scaler for the target (labels). We use StandardScaler to center and scale
    # the labels (zero mean, unit variance). This returns a 2D array when applied to a
    # one-column DataFrame.
    target_scaler = StandardScaler()

    # Fit the scaler on the labels and transform them. We convert `housing_labels` to a
    # DataFrame (shape: [n_samples, 1]) so inverse_transform later will accept the same shape.
    scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

    # Train a simple linear regression model to predict the scaled labels from median_income.
    # Note: `housing[["median_income"]]` is a 2D DataFrame (n_samples, 1) as required by sklearn.
    model = LinearRegression()
    model.fit(housing[["median_income"]], scaled_labels)

    # Take a small slice of the feature to act as "new" data we want predictions for.
    some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data (shape: [5, 1])

    # Predict on the new data. Because we trained on a 2D target, `predict` returns a 2D array
    # (shape: [5, 1]) which we can pass directly to inverse_transform.
    scaled_predictions = model.predict(some_new_data)

    # Convert the scaled predictions back to the original label units so the results are
    # interpretable (e.g., actual median house values in dollars).
    predictions = target_scaler.inverse_transform(scaled_predictions)

    # Return the de-scaled predictions so callers can inspect them (keeps the function useful).
    return predictions

def transformed_target_regressor():
    """
    CONCEPT: TransformedTargetRegressor
    Scikit-Learn provides a built-in wrapper to handle target scaling automatically.
    It applies the specified transformation to the target during training and inversely
    transforms predictions back to the original scale.
    """
    model = TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=StandardScaler()
    )
    model.fit(housing[["median_income"]], housing_labels)
    predictions = model.predict(housing[["median_income"]].iloc[:5])
    print(f"\nTransformedTargetRegressor Predictions (first 5):\n{predictions}")

def pipeline_example():
    """
    CONCEPT: Pipelines
    Demonstrates how to create a Scikit-Learn Pipeline that chains multiple
    transformations and a final estimator into a single object.
    """
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Fit the pipeline on the training data
    pipeline.fit(housing[["median_income"]], housing_labels)

    # Make predictions using the pipeline
    predictions = pipeline.predict(housing[["median_income"]].iloc[:5])
    print(f"\nPipeline Predictions (first 5):\n{predictions}")
    # pipeline's last estimator is LinearRegression, so we can access its not access fit_transform
    #housing_num_prepared = pipeline.fit_transform(housing[["median_income"]])
    #print(f"\nPipeline Transformed Data (first 5):\n{housing_num_prepared[:5].round(2)}")
    # pipeline's last estimator is LinearRegression, so we can access its coefficients
    print(f"Pipeline Linear Regression Coefficients: {pipeline.named_steps['regressor'].coef_}")
    # set_config(display='diagram')
    # pipeline

def combined_pipeline_for_numerical_and_categorical():
    """
    CONCEPT: ColumnTransformer with Pipelines
    Demonstrates how to create separate pipelines for numerical and categorical
    data, then combine them using ColumnTransformer.
    """
    # Numerical pipeline
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder())
    ])

    # Combine pipelines using ColumnTransformer
    full_pipeline = ColumnTransformer(transformers=[
        ('num', num_pipeline, housing.select_dtypes(include=[np.number]).columns),
        ('cat', cat_pipeline, ['ocean_proximity'])
    ])

    # Fit and transform the entire dataset
    housing_prepared = full_pipeline.fit_transform(housing)

    print(f"\nCombined Pipeline Transformed Data Shape: {housing_prepared.shape}")

# --- EXECUTION: TRANSFORMATION ---
# Select only number columns for scaling operations
housing_num = housing.select_dtypes(include=[np.number])

min_max_scaling(housing_num)
standard_scaling(housing_num)
# Standard scaling without centering (useful for sparse data)
standard_scaling_with_mean(housing_num)

# Log transform to fix skewed population data
log_transform(housing_num)

# Advanced RBF transformation to handle the specific "Age 35" bump
transforming_multimodal_distribution_using_rbf_kernel(housing)

# Demonstrate label scaling with a simple linear regression
predictions = label_scaling_example()
print(f"\nLabel Scaling Example Predictions (first 5):\n{predictions}")

# Demonstrate TransformedTargetRegressor
transformed_target_regressor()

# Demonstrate Pipelines
pipeline_example()

# Demonstrate Combined Pipeline for Numerical and Categorical Data
combined_pipeline_for_numerical_and_categorical()

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

### All together
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                           "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age

housing_prepared = preprocessing.fit_transform(housing)
print(f"\nFinal Prepared Housing Data Shape: {housing_prepared.shape}")
print(f"Final Prepared Housing Data (first 5):\n{housing_prepared[:5].round(2)}")
print(f"Feature Names:  {preprocessing.get_feature_names_out()}")

### --- EXECUTION: TRAINING A MODEL ---
def train_linear_regression():
    lin_reg = make_pipeline(preprocessing, LinearRegression())
    lin_reg.fit(housing, housing_labels)
    housing_predictions = lin_reg.predict(housing)
    print(f"\nLinear Regression Predictions (first 5):\n{housing_predictions[:5].round(-2)}")
    print(f"\nLabel True Values (first 5):\n{housing_labels.iloc[:5].values}")
    lin_mse = root_mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(f"\nLinear Regression RMSE: {lin_rmse:.2f}")
    print(f"\nLinear Regression MSE: {lin_mse:.2f}")

def train_decision_tree():
    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
    tree_reg.fit(housing, housing_labels)
    housing_predictions = tree_reg.predict(housing)
    print(f"\nDecision Tree Predictions (first 5):\n{housing_predictions[:5].round(-2)}")
    print(f"\nLabel True Values (first 5):\n{housing_labels.iloc[:5].values}")
    tree_mse = root_mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(f"\nDecision Tree RMSE: {tree_rmse:.2f}")
    print(f"\nDecision Tree MSE: {tree_mse:.2f}")

def cross_val_score_with_decision_tree():
    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
    scores = cross_val_score(tree_reg, housing, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    print(f"\nDecision Tree Cross-Validation RMSE Scores: {tree_rmse_scores}")
    print(f"Mean: {tree_rmse_scores.mean():.2f}, Stddev: {tree_rmse_scores.std():.2f}")
    print(f"Description: {pd.Series(tree_rmse_scores).describe()}")

def cross_val_score_with_linear_regression():
    lin_reg = make_pipeline(preprocessing, LinearRegression())
    scores = cross_val_score(lin_reg, housing, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-scores)
    print(f"\nLinear Regression Cross-Validation RMSE Scores: {lin_rmse_scores}")
    print(f"Mean: {lin_rmse_scores.mean():.2f}, Stddev: {lin_rmse_scores.std():.2f}")
    print(f"Description: {pd.Series(lin_rmse_scores).describe()}")

def cross_val_score_with_random_forest():
    forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
    scores = cross_val_score(forest_reg, housing, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-scores)
    print(f"\nRandom Forest Cross-Validation RMSE Scores: {forest_rmse_scores}")
    print(f"Mean: {forest_rmse_scores.mean():.2f}, Stddev: {forest_rmse_scores.std():.2f}")
    print(f"Description: {pd.Series(forest_rmse_scores).describe()}")

def grid_search_cv_with_decision_tree():
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ])
    param_grid = [
        {'preprocessing__geo__n_clusters': [5, 8, 10],
         'random_forest__max_features': [4, 6, 8]},
        {'preprocessing__geo__n_clusters': [10, 15],
         'random_forest__max_features': [6, 8, 10]},
    ]
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                               scoring='neg_mean_squared_error')
    grid_search.fit(housing, housing_labels)
    cv_res = grid_search.cv_results_
    for score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
        print(np.sqrt(-score), params)

def random_search_cv_with_random_forest():
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ])
    param_distributions = {
        'preprocessing__geo__n_clusters': [5, 8, 10, 15],
        'random_forest__max_features': [4, 6, 8, 10],
        'random_forest__n_estimators': [50, 100, 150],
        'random_forest__max_depth': [None, 10, 20, 30],
    }
    random_search = RandomizedSearchCV(full_pipeline, param_distributions,
                                       n_iter=10, cv=3,
                                       scoring='neg_mean_squared_error',
                                       random_state=42)
    random_search.fit(housing, housing_labels)
    cv_res = random_search.cv_results_
    for score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
        print(np.sqrt(-score), params)

train_linear_regression()
train_decision_tree()
cross_val_score_with_decision_tree()
cross_val_score_with_linear_regression()
#cross_val_score_with_random_forest()
#grid_search_cv_with_decision_tree()
#random_search_cv_with_random_forest()
#plt.show()