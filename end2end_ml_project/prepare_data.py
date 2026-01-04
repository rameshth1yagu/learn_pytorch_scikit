from scipy.signal import correlate

import get_data
from sklearn.impute import SimpleImputer#, KNNImputer, IterativeImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel

train_set, _ = get_data.train_test_split_with_stratification()
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

def clean_data_options():
    # Option 1: Get rid of the corresponding districts.
    housing.dropna(subset=["total_bedrooms"], inplace=True)  # option 1
    # Option 2: Get rid of entire attribute
    housing.drop("total_bedrooms", axis=1, inplace=True)  # option 2
    # Option 3: Imputation
    # Fill the empty / null with zero, mean, median etc
    median = housing["total_bedrooms"].median()  # option 3
    housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)

def sklearn_imputation(data, enable_hist: bool = False):
    imputer = SimpleImputer(strategy="median")
    data_numeric = data.select_dtypes(include=[np.number])
    # Estimator
    imputer.fit(data_numeric)
    # Transformer
    X = imputer.transform(data_numeric)

    # Estimator Inspection
    print(f"Imputer's hyperparameter - imputer.strategy: {imputer.strategy}")
    print(f"Imputer's learned parameters - imputer.statistics_: {imputer.statistics_}")
    print(f"Median by pandas: {data_numeric.median().values}")
    x_data_frame = pd.DataFrame(X, columns=data_numeric.columns, index=data_numeric.index)
    x_data_frame.info()
    if enable_hist:
        x_data_frame.hist(bins=50, figsize=(12, 8))
    return x_data_frame, X

def text_attributes(data):
    print("\nText Attributes")
    housing_cat = data[["ocean_proximity"]]
    print(housing_cat.head(10))
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print(f"OrdinalEncoder's categories_: {ordinal_encoder.categories_}")
    print(housing_cat_encoded[:10])
    one_hot_encoder = OneHotEncoder()
    housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat)
    print(f"OneHotEncoder's categories_: {one_hot_encoder.categories_}")
    print(housing_cat_1hot.toarray()[:10])
    print(f"housing_cat_1hot: {housing_cat_1hot}")
    print(f"housing_cat_1hot: {housing_cat_1hot.toarray()}")
    # get_dummies is almost same like OneHotEncoder, but used for analysis.
    # For actual ML pipelines, OneHotEncoder is best approach
    print(f"pd.get_dummies: {pd.get_dummies(housing["ocean_proximity"])}")

def null_entries(data):
    null_rows_idx = data.isnull().any(axis=1)
    print(f"Null rows: {data[null_rows_idx]}")
    data.loc[null_rows_idx].head()

housing_tr, X = sklearn_imputation(housing)
text_attributes(housing)
null_entries(housing)

# Feature scaling
def min_max_scaling(data):
    print(f"\n Min Max Scaling")
    min_max_scaler = MinMaxScaler(feature_range=(-1,1))
    data_min_max_scaled = min_max_scaler.fit_transform(data)
    # print(f"min_max_scaler's data_min_: {min_max_scaler.data_min_}")
    # print(f"min_max_scaler's data_max_: {min_max_scaler.data_max_}")
    # print(f"min_max_scaler's scale_: {min_max_scaler.scale_}")
    # print(f"min_max_scaler's min_: {min_max_scaler.min_}")
    print(data_min_max_scaled[:1])
    data_frame = pd.DataFrame(data_min_max_scaled, columns=data.columns)
    data_frame.hist(bins=50, figsize=(12, 8))

def standard_scaling(data):
    standard_scaler = StandardScaler()
    data_standard_scaled = standard_scaler.fit_transform(data)
    data_frame = pd.DataFrame(data_standard_scaled, columns=data.columns)
    data_frame.hist(bins=50, figsize=(12, 8))

def standard_scaling_with_mean(data):
    standard_scaler = StandardScaler(with_mean=False)
    data_standard_scaled = standard_scaler.fit_transform(data)
    data_frame = pd.DataFrame(data_standard_scaled, columns=data.columns)
    data_frame.hist(bins=50, figsize=(12, 8))

def log_transform(data):
    log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
    data_log_transformed = log_transformer.transform(data)
    data_frame = pd.DataFrame(data_log_transformed, columns=data.columns)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    data["population"].hist(ax=axes[0], bins=50)
    axes[0].set_title("Original Data")
    data_frame["population"].hist(bins=50, ax=axes[1])
    axes[1].set_title("Transformed Data")
    data_frame.hist(bins=50, figsize=(12, 8))

# RBF - radial basis function
def rbf_kernel_char():
    ages = np.linspace(housing["housing_median_age"].min(),
                       housing["housing_median_age"].max(),
                       500).reshape(-1, 1)
    gamma1 = 0.1
    gamma2 = 0.03
    rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
    rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Housing median age")
    ax1.set_ylabel("Number of districts")
    ax1.hist(housing["housing_median_age"], bins=50)

    ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
    color = "blue"
    ax2.plot(ages, rbf1, color=color, label="gamma = 0.10")
    ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel("Age similarity", color=color)

    plt.legend(loc="upper left")

def transforming_multimodal_distribution_using_rbf_kernel(data):
    print("\nTransforming Multimodal Distribution using RBF Kernel")
    housing["age_35"] = rbf_kernel(data[["housing_median_age"]], [[35]], gamma=0.1)
    #print(f"age_35: {age_35}")
    #correlate_housing_age = housing.corr(numeric_only=True)
    #print(f"Correlation: {correlate_housing_age["age_35"].sort_values(ascending=False)}")
    rbf_kernel_char()


housing_num = housing.select_dtypes(include=[np.number])
min_max_scaling(housing_num)
standard_scaling(housing_num)
standard_scaling_with_mean(housing_num)
log_transform(housing_num)
transforming_multimodal_distribution_using_rbf_kernel(housing)
plt.show()