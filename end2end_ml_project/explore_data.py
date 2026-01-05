import get_data
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
from pandas.plotting import scatter_matrix

### 1. Exploration Phase Setup
# We have already split the data. Now we strictly use the TRAINING set.
# Ideally, if the dataset is massive, we would sample it down for speed.
# Since this dataset is small (~16k rows), we can explore the whole training set.
strat_train_set, _ = get_data.train_test_split_with_stratification()

# CONCEPT: Data Copying
# We create a copy of the dataframe.
# WHY? If we modify 'housing' (e.g., adding test columns), we don't want
# to accidentally corrupt the original 'strat_train_set' variable.
housing = strat_train_set.copy()

### 2. Geographical Visualization
# The goal is to verify if location affects price (e.g., coastal vs inland).

# Simple Plot: Just draws the dots. Hard to see patterns where points overlap.
housing.plot(title="Normal", kind="scatter", x="longitude", y="latitude", grid=True)

# CONCEPT: Alpha Channel (Density)
# We set alpha=0.2 (20% opacity).
# WHY? This handles "Overplotting". In high-density areas (like LA or Bay Area),
# many transparent dots pile on top of each other, creating a dark color.
# It instantly reveals high-density clusters.
housing.plot(title="Density", kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)

# CONCEPT: 4-Dimensional Plotting
# We display 4 dimensions on a 2D screen:
# 1. X-axis: Longitude
# 2. Y-axis: Latitude
# 3. Size (s): Population (Larger circle = More people)
# 4. Color (c): Price (Red = Expensive, Blue = Cheap)
housing.plot(title="Color", kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, # Divide by 100 to keep circle sizes reasonable
             label="population",
             c="median_house_value",        # The column used for color
             cmap="jet",                    # 'jet' ranges from Blue (Low) to Red (High)
             colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))

# CONCEPT: Colorblind Friendliness
# 'jet' is often criticized because it can be misleading visually.
# 'viridis' is a perceptually uniform colormap that is readable by colorblind users
# and prints well in black & white.
housing.plot(title="ColorBlind", kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="viridis", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))


def correlation_matrix(data, with_additional_attributes=False):
    """
    Calculates and displays the Standard Correlation Coefficient (Pearson's r).
    Values range from -1 (inverse relationship) to 1 (direct relationship).
    """

    # CONCEPT: Correlation
    # We check relationships. A high positive number means "If X goes up, Y goes up."
    # numeric_only=True is required to ignore text columns like 'ocean_proximity'.
    corr_matrix = data.corr(numeric_only=True)

    print("\nCorrelation Matrix")
    # We sort by 'median_house_value' to see what matters most for price.
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # CONCEPT: Scatter Matrix
    # The correlation number (e.g., 0.68) can hide things (like curved relationships).
    # A Scatter Matrix plots every attribute against every other attribute.
    # The diagonal (Top-Left to Bottom-Right) displays a histogram of that variable
    # because plotting a variable against itself would just be a straight line.
    if not with_additional_attributes:
        attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
    else:
        # Later, we check if our new engineered features are better than the old ones.
        attributes = ["median_house_value", "bedrooms_ratio"]

    scatter_matrix(housing[attributes], figsize=(12, 8))


def with_california_map():
    """
    Advanced Visualization: Overlays the scatter plot on top of a real
    geographical image of California for context.
    """
    # 1. Download the background image if we don't have it
    filename = "california.png"
    filepath = Path(f"my_{filename}")
    if not filepath.is_file():
        homlp_root = "https://github.com/ageron/handson-mlp/raw/main/"
        url = homlp_root + "images/end_to_end_project/" + filename
        print("Downloading", filename)
        urllib.request.urlretrieve(url, filepath)

    # Rename columns simply for prettier labels on the final graph
    housing_renamed = housing.rename(columns={
        "latitude": "Latitude", "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)"})

    # Create the standard scatter plot
    housing_renamed.plot(
        kind="scatter", x="Longitude", y="Latitude",
        s=housing_renamed["Population"] / 100, label="Population",
        c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
        legend=True, sharex=False, figsize=(10, 7))

    # CONCEPT: Image Overlay
    # 2. Load the image file
    california_img = plt.imread(filepath)

    # 3. Define the bounding box (Extent)
    # These numbers correspond to the Lat/Long edges of the image.
    # We must map the pixels of the image to the coordinates of the graph.
    axis = -124.55, -113.95, 32.45, 42.05
    plt.axis(axis)

    # 4. Draw the image behind the scatter plot
    plt.imshow(california_img, extent=axis)


def attribute_combinations():
    """
    CONCEPT: Feature Engineering
    Raw data is often noisy or irrelevant. We combine columns to create
    new metrics that might correlate better with the target (Price).
    """

    # "Total Rooms" is useless because it depends on district population.
    # "Rooms per house" is better: Big houses are expensive.
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]

    # "Total Bedrooms" is useless.
    # "Bedroom Ratio" is better: Houses with FEWER bedrooms (relative to total rooms)
    # tend to be fancier/more open-concept, or perhaps office spaces?
    # Actually, lower ratio usually implies larger living areas.
    housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]

    # "Population" is useless.
    # "People per house" is better: High density per house usually means lower income/price.
    housing["people_per_household"] = housing["population"] / housing["households"]

    # Run correlation again to see if our new features beat the old ones.
    # Spoiler: 'bedrooms_ratio' usually has a much stronger negative correlation
    # than the raw 'total_bedrooms' count.
    correlation_matrix(housing, True)

# --- EXECUTION ---

# 1. Check baseline correlations
correlation_matrix(housing)

# 2. Zoom in on the most promising attribute
# We see a strong linear trend between Income and Value.
# We also see horizontal lines (at $500k, $350k) which indicate data capping quirks.
housing.plot(title="Income", kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)

# 3. (Optional) Run the map visualization
# with_california_map()

# 4. Create and test new features
attribute_combinations()

plt.show()