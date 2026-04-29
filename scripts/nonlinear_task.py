import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Generate data
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

# Polynomial Model
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Prediction range
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_poly = poly_model.predict(X_new_poly)

# Decision Tree Model
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

y_tree = tree.predict(X_new)

# WINDOW 1 → Polynomial
plt.figure()
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_new, y_poly, color='red', label='Polynomial')
plt.legend()
plt.title("Polynomial Regression")
plt.show(block=False)

# WINDOW 2 → Decision Tree
plt.figure()
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_new, y_tree, color='green', label='Decision Tree')
plt.legend()
plt.title("Decision Tree Regression")
plt.show(block=False)

# WINDOW 3 → COMBINED
plt.figure()
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_new, y_poly, color='red', label='Polynomial')
plt.plot(X_new, y_tree, color='green', label='Decision Tree')
plt.legend()
plt.title("Polynomial vs Decision Tree")
plt.show()