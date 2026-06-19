from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
import joblib
import numpy as np

# 1. LOAD DATA
data = load_breast_cancer()
X, y = data.data, data.target

# 2. BASELINE MODEL
baseline_model = RandomForestClassifier(random_state=42)

baseline_scores = cross_val_score(baseline_model, X, y, cv=3)

print("\n Baseline Model:")
print("Mean Accuracy:", round(np.mean(baseline_scores), 4))

# 3. SEARCH SPACE (IMPORTANT)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# 4. GRID SEARCH
rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1
)

# 5. RUN SEARCH
grid_search.fit(X, y)

print("\n Best Parameters:")
print(grid_search.best_params_)

print("\n Best Cross-Validation Score:")
print(round(grid_search.best_score_, 4))

# 6. TOTAL MODELS TRAINED
total_models = (
    len(param_grid['n_estimators']) *
    len(param_grid['max_depth']) *
    len(param_grid['min_samples_split'])
) * 3   # CV folds

print(f"\n Total Models Trained: {total_models}")

# 7. SAVE BEST MODEL
joblib.dump(grid_search.best_estimator_, "best_model.pkl")

print("\n Model saved as best_model.pkl")