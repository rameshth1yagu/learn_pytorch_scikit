import get_data
import matplotlib.pyplot as plt

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
plt.show()
