import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Load dataset
data = fetch_california_housing()

X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Parameter grid (what to test)
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0, 500.0]
}

# Grid Search
grid_search = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='r2'
)

# Run search
grid_search.fit(X_train_scaled, y_train)

# Results
print("Best Alpha Found:", grid_search.best_params_)
print("Best R2 Score:", round(grid_search.best_score_, 4))