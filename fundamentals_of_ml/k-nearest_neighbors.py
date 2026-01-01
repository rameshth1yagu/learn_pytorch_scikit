import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

data_root = "https://raw.githubusercontent.com/ageron/data/main"
life_sat = pd.read_csv(data_root + "/lifesat/lifesat.csv")
X = life_sat[["GDP per capita (USD)"]].values
y = life_sat[["Life satisfaction"]].values


life_sat.plot(kind="scatter", grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
#plt.show()
#model = LinearRegression()
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

x_new = [[33442.8]]
print(f"KNeighborsRegressor: {model.predict(x_new)}")

life_sat.plot(title="KNeighborsRegressor...", kind="scatter", grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([30_500, 35_500, 4, 9])
plt.show()
