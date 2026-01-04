import get_data
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import numpy as np

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

def sklearn_imputation(data):
    imputer = SimpleImputer(strategy="median")
    data_numeric = data.select_dtypes(include=[np.number])
    imputer.fit(data_numeric)
    print(f"Median by SimpleImputer: {imputer.statistics_}")
    print(f"Median by pandas: {data_numeric.median().values}")
    X = imputer.transform(data_numeric)

sklearn_imputation(housing)