import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

data_root = "https://raw.githubusercontent.com/ageron/data/main"
life_sat = pd.read_csv(data_root + "/lifesat/lifesat.csv")
X = life_sat[["GDP per capita (USD)"]].values
y = life_sat[["Life satisfaction"]].values

poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X, y)

X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = poly_model.predict(X_range)
life_sat.plot(title="Polynomial Regression", kind="scatter", grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.plot(X_range, y_range, color="red")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

