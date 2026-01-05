from scipy.signal import correlate
import get_data
from sklearn.impute import SimpleImputer#, KNNImputer, IterativeImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel

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

plt.show()