from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

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

housing_full = load_housing_data()
train_set, test_set = shuffle_and_split_data(housing_full, 0.2)
#train_set, test_set = sklearn_shuffle_and_split_data(housing_full, 0.2)
data_volume(housing_full, train_set, test_set)
standard_operating_procedure(train_set)
hist(train_set)