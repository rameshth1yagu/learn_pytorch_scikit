import get_data
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
from pandas.plotting import scatter_matrix

### Exploration Phase
# Ideally, when train_set is large, we take a sample out of it during this phase.
# since, our current data set is small, we can take the entire data set as sample
strat_train_set, _ = get_data.train_test_split_with_stratification()

housing = strat_train_set.copy()

housing.plot(title="Normal", kind="scatter", x="longitude", y="latitude", grid=True)
housing.plot(title="Density", kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
housing.plot(title="Color", kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
housing.plot(title="ColorBlind", kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="viridis", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))

def correlation_matrix(data, with_additional_attributes=False):
    corr_matrix = data.corr(numeric_only=True)
    print("\nCorrelation Matrix")
    #print(corr_matrix)
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    if not with_additional_attributes:
        attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
    else:
        attributes = ["median_house_value", "bedrooms_ratio"]
    scatter_matrix(housing[attributes], figsize=(12, 8))

def with_california_map():
    # Download the California image
    filename = "california.png"
    filepath = Path(f"my_{filename}")
    if not filepath.is_file():
        homlp_root = "https://github.com/ageron/handson-mlp/raw/main/"
        url = homlp_root + "images/end_to_end_project/" + filename
        print("Downloading", filename)
        urllib.request.urlretrieve(url, filepath)

    housing_renamed = housing.rename(columns={
        "latitude": "Latitude", "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)"})
    housing_renamed.plot(
        kind="scatter", x="Longitude", y="Latitude",
        s=housing_renamed["Population"] / 100, label="Population",
        c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
        legend=True, sharex=False, figsize=(10, 7))

    california_img = plt.imread(filepath)
    axis = -124.55, -113.95, 32.45, 42.05
    plt.axis(axis)
    plt.imshow(california_img, extent=axis)

def attribute_combinations():
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["people_per_household"] = housing["population"] / housing["households"]
    correlation_matrix(housing, True)

correlation_matrix(housing)
housing.plot(title="Income", kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
#with_california_map()
attribute_combinations()
plt.show()
