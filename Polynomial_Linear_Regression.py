import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis = 0)
y = X**3 + np.random.randn(80, 1) * 30

data = pd.DataFrame(data = np.column_stack([X, y]), columns = ["X", "y"])
data.to_csv("polynomial_data.csv")

data = pd.read_csv("polynomial_data.csv")

X = data[["X"]].values
y = data[["y"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

model = LinearRegression()
model.fit(X_poly_train, y_train)

y_pred = model.predict(X_poly_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.scatter(X_test, y_test, color = "red", label = "Gerçek Değerler")
plt.plot(X_test, y_pred, color = "blue", label = "Polinomal Regresyon Modeli")
plt.title("Polinomal Regresyon Modeli")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
