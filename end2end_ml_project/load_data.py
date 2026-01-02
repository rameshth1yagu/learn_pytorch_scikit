from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt

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

housing_full = load_housing_data()
standard_operating_procedure(housing_full)
hist(housing_full)