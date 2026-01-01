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

housing_full = load_housing_data()
# Get first 5 rows
print(housing_full.head())
# Print Info - Class, Row count, Column infos like name, non-null count, datatype
housing_full.info()
# Get Distinct value along with count for column 'ocean_proximity'
print(housing_full["ocean_proximity"].value_counts())