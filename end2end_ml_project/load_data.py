from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def standard_operating_procedure(housing_data: any):
    # Get first 5 rows
    print(housing_data.head())
    # Print Info - Class, Row count, Column infos like name, non-null count, datatype
    housing_data.info()
    # Get Distinct value along with count for column 'ocean_proximity'
    print(housing_data["ocean_proximity"].value_counts())
    # Mathematical Statistics for numerical columns
    print(housing_data.describe())

def hist(housing_data: any):
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    housing_data.hist(bins=50, figsize=(12,8))
    plt.show()

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    print(shuffled_indices)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def sklearn_shuffle_and_split_data(data, test_ratio):
    return train_test_split(data, test_size=test_ratio, random_state=42)

def data_volume(source, train, test):
    print(f"Source: {len(source)}")
    print(f"Train: {len(train)}")
    print(f"Test: {len(test)}")

def create_feature_stratum(data):
    data["income_cat"] = pd.cut(data["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    #data["income_cat"].hist()
    # cat_counts = data["income_cat"].value_counts().sort_index()
    # cat_counts.plot.bar(rot=0, grid=True, xlabel="Income Category", ylabel="Count")
    # plt.show()

# .split comes with iterator of the train and test data sets
def stratified_shuffle_split():
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    strat_splits = []
    for train_index, test_index in splitter.split(housing_full, housing_full["income_cat"]):
        strat_train_set = housing_full.loc[train_index]
        strat_test_set = housing_full.loc[test_index]
        strat_splits.append([strat_train_set, strat_test_set])

# see if data is evening distributed using the new feature 'income_cat'
def train_test_split_with_stratification():
    train_set, test_set = train_test_split(housing_full, stratify=housing_full["income_cat"],
                                           test_size=0.2, random_state=42)
    print(f"train_test_split_with_stratification: value count: {test_set["income_cat"].value_counts()}")
    print(f"Distribution: {test_set["income_cat"].value_counts() / len(test_set)}")
    return train_set, test_set

def delete_income_cat(data, test):
    for set_ in (data, test):
        set_.drop("income_cat", axis=1, inplace=True)

housing_full = load_housing_data()
create_feature_stratum(housing_full)
#train_set, test_set = shuffle_and_split_data(housing_full, 0.2)
#train_set, test_set = sklearn_shuffle_and_split_data(housing_full, 0.2)
train_set, test_set = train_test_split_with_stratification()

data_volume(housing_full, train_set, test_set)
standard_operating_procedure(train_set)
hist(train_set)
delete_income_cat(train_set, test_set)