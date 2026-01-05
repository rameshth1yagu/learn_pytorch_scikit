from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from scipy.stats import binom

def load_housing_data():
    """
    CONCEPT: Automating Data Ingestion
    This function ensures the data exists locally. If not, it downloads it.
    This makes the script reproducible on any machine without manual setup.
    """
    tarball_path = Path("datasets/housing.tgz")
    # Check if we already have the compressed file
    if not tarball_path.is_file():
        # Create directory if it doesn't exist
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        # Download the file
        urllib.request.urlretrieve(url, tarball_path)
        # Extract the TGZ file
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")

    # Return the CSV as a Pandas DataFrame
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def standard_operating_procedure(housing_data: any):
    """
    CONCEPT: Initial Data Inspection
    Before doing any ML, we must understand the "Shape" and "Quality" of the data.
    """
    # 1. Peek at the data: Check if columns look correct (e.g., headers are not rows)
    print(housing_data.head())

    # 2. Tech Specs: Check for missing values (Nulls) and data types (float vs object)
    # Important: If 'total_bedrooms' has fewer non-nulls than others, we have missing data.
    housing_data.info()

    # 3. Categorical Analysis: 'ocean_proximity' is text, not a number.
    # We need to see how many categories exist and if they are balanced.
    print(housing_data["ocean_proximity"].value_counts())

    # 4. Statistical Summary:
    # Key things to look for:
    # - Mean vs Median (Is the data skewed?)
    # - Min/Max (Are there outliers? e.g., a house costing $500,000,000?)
    print(housing_data.describe())

def hist(housing_data: any, show):
    """
    CONCEPT: Distribution Visualization
    Histograms tell us the "Shape" of the data (Bell curve, Long tail, etc.)
    """
    # Set global font sizes for better readability on high-res screens
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    if show:
        # Create a histogram for EVERY numerical column at once.
        # bins=50: High resolution to see fine details (like the price cap at $500k)
        housing_data.hist(bins=50, figsize=(12,8))
        plt.show()

def shuffle_and_split_data(data, test_ratio):
    """
    CONCEPT: Manual Random Splitting (The Hard Way)
    Demonstrates the logic behind splitting: Shuffle indices -> Slice array.
    WARNING: This is not stable. If you run it again, you get a different test set.
    """
    # Create a random permutation of indices (e.g., [5, 0, 19, ...])
    shuffled_indices = np.random.permutation(len(data))
    print(shuffled_indices)

    # Calculate the cut point
    test_set_size = int(len(data) * test_ratio)

    # Slice the shuffled list
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    # Return the rows matching those indices
    return data.iloc[train_indices], data.iloc[test_indices]

def sklearn_shuffle_and_split_data(data, test_ratio):
    """
    CONCEPT: Reproducible Random Splitting (The Standard Way)
    Uses 'random_state=42' to ensure we get the SAME random split every time.
    """
    return train_test_split(data, test_size=test_ratio, random_state=42)

def data_volume(source, train, test):
    """
    Utility: Verifies that rows weren't lost during the split.
    Total should equal Train + Test.
    """
    print(f"Source: {len(source)}")
    print(f"Train: {len(train)}")
    print(f"Test: {len(test)}")

def create_feature_stratum(data):
    """
    CONCEPT: Discretization / Binning
    We convert continuous Income (0.5 to 15.0) into discrete Categories (1, 2, 3, 4, 5).
    WHY? We want to ensure our Test Set represents all income levels fairly.
    If we don't bin it, we might accidentally get a Test Set with ZERO rich people.
    """
    data["income_cat"] = pd.cut(data["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    # Visualization code commented out:
    # This would show a bell-curve histogram of the new categories.

def stratified_shuffle_split():
    """
    CONCEPT: Advanced Stratification (Cross-Validation Prep)
    StratifiedShuffleSplit creates MULTIPLE splits (n_splits=5), not just one.
    This is useful for Cross-Validation later.
    """
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    strat_splits = []

    # .split returns the indices (Row IDs), not the data itself.
    for train_index, test_index in splitter.split(housing_full, housing_full["income_cat"]):
        strat_train_set = housing_full.loc[train_index]
        strat_test_set = housing_full.loc[test_index]
        strat_splits.append([strat_train_set, strat_test_set])

def train_test_split_with_stratification():
    """
    CONCEPT: Stratified Sampling
    This forces the Test Set to have the exact same percentage of
    Rich/Middle/Poor people as the original full dataset.
    """
    # stratify=housing_full["income_cat"]: The column to mimic
    train_set, test_set = train_test_split(housing_full,
                                           stratify=housing_full["income_cat"],
                                           test_size=0.2,
                                           random_state=42)

    print(f"train_test_split_with_stratification: value count: {test_set['income_cat'].value_counts()}")
    # Calculate proportions to prove it worked (should match population %)
    print(f"Distribution: {test_set['income_cat'].value_counts() / len(test_set)}")
    return train_set, test_set

def delete_income_cat(data, test):
    """
    CONCEPT: Cleanup
    'income_cat' was a temporary helper column used only for splitting.
    We remove it because the ML model should learn from the raw 'median_income',
    not our artificial category.
    """
    for set_ in (data, test):
        set_.drop("income_cat", axis=1, inplace=True)

def find_null(data):
    """
    CONCEPT: Data Auditing
    Counts how many cells are empty.
    'total_bedrooms' typically has nulls that we will need to Impute later.
    """
    print(f"Data Null values: {data.isnull().sum()}")
    print(f"total_bedrooms - null values: {data['total_bedrooms'].isnull().sum()}")

# --- EXECUTION FLOW ---

# 1. Load Data
housing_full = load_housing_data()

# 2. Create the Stratum (Income Category) so we can split fairly
create_feature_stratum(housing_full)

# 3. Perform the Split (Choose ONE method)
# Method A: Random (Bad for skewed data)
#train_set, test_set = shuffle_and_split_data(housing_full, 0.2)
#train_set, test_set = sklearn_shuffle_and_split_data(housing_full, 0.2)

# Method B: Stratified (Best Practice)
train_set, test_set = train_test_split_with_stratification()

# 4. Verify Volume
data_volume(housing_full, train_set, test_set)

# 5. Inspect the Training Set (Never inspect Test Set!)
standard_operating_procedure(train_set)

# 6. Check for missing data
find_null(train_set)

# 7. Visualize
hist(train_set, False)

# 8. Clean up the helper column
delete_income_cat(train_set, test_set)


# --- EXTRAS: PROVING WHY STRATIFICATION MATTERS ---

def sampling_bias():
    """
    CONCEPT: Probability Theory (Binomial Distribution)
    Calculates the mathematical risk of getting a "Bad Sample" (Skewed Gender Ratio)
    if we used purely random sampling on a small population (1000 people).
    Result (approx 10.7%) proves random sampling is risky.
    """
    sample_size = 1000
    ratio_female = 0.516
    # Probability of getting < 49% females
    proba_too_small = binom(sample_size, ratio_female).cdf(490 - 1)
    # Probability of getting > 54% females
    proba_too_large = 1 - binom(sample_size, ratio_female).cdf(540)
    print(f"Sampling Bias Risk: {proba_too_small + proba_too_large}")

def income_cat_proportions(data):
    # Helper to calculate % of each income category
    return data["income_cat"].value_counts() / len(data)

def compare_random_and_stratified_sampling():
    """
    CONCEPT: Bias Comparison
    Creates a table proving that Stratified Sampling (Error ~0%)
    is superior to Random Sampling (Error ~1-2%).
    """
    # Create a bad random split for comparison
    train_set, test_set = train_test_split(housing_full, test_size=0.2, random_state=42)
    # Re-run stratified split (we need the temp object before deletion)
    strat_train_set, strat_test_set = train_test_split_with_stratification()

    # Build comparison table
    compare_props = pd.DataFrame({
        "Overall %": income_cat_proportions(housing_full),
        "Stratified %": income_cat_proportions(strat_test_set),
        "Random %": income_cat_proportions(test_set),
    }).sort_index()

    compare_props.index.name = "Income Category"
    # Calculate % Error
    compare_props["Strat. Error %"] = (compare_props["Stratified %"] / compare_props["Overall %"] - 1)
    compare_props["Rand. Error %"] = (compare_props["Random %"] / compare_props["Overall %"] - 1)

    # Display formatted table
    print((compare_props * 100).round(2))

# Run the proof
sampling_bias()
compare_random_and_stratified_sampling()