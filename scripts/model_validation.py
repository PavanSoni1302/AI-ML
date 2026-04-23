# =========================================
# DAY 16: MODEL VALIDATION & K-FOLD
# =========================================

from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import numpy as np

# 1. LOAD DATA
digits = load_digits()
X, y = digits.data, digits.target

# 2. DEFINE MODEL
model = RandomForestClassifier(n_estimators=50, random_state=42)

# 3. 5-FOLD CROSS VALIDATION
scores = cross_val_score(model, X, y, cv=5)

print("\n Cross-Validation Scores:")
print(scores)

print(f"\n Mean Accuracy: {np.mean(scores):.4f}")
print(f" Standard Deviation: {np.std(scores):.4f}")

# 4. OVERFITTING CHECK
model.fit(X, y)
train_score = model.score(X, y)

print(f"\n Training Accuracy: {train_score:.4f}")
print(f" Validation Accuracy: {np.mean(scores):.4f}")

# 5. STABILITY TEST (SHUFFLE vs NO SHUFFLE)

print("\n Stability Test:")

# Without shuffle
kf_no_shuffle = KFold(n_splits=10, shuffle=False)
scores_no_shuffle = cross_val_score(model, X, y, cv=kf_no_shuffle)

# With shuffle
kf_shuffle = KFold(n_splits=10, shuffle=True, random_state=42)
scores_shuffle = cross_val_score(model, X, y, cv=kf_shuffle)

print("\nWithout Shuffle STD:", np.std(scores_no_shuffle))
print("With Shuffle STD:", np.std(scores_shuffle))