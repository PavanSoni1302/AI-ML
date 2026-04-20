# =========================================
# MUX INTELLIGENCE PROTOTYPE (CAPSTONE)
# User Compatibility Prediction Engine
# =========================================

# 1. IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# 2. DATA GENERATION (LOGICAL)
np.random.seed(42)

data_size = 300

df = pd.DataFrame({
    "age_diff": np.random.randint(0, 10, data_size),
    "common_interests": np.random.randint(0, 5, data_size),
    "availability_overlap": np.random.rand(data_size) * 10,
    "activity_match": np.random.rand(data_size) * 10
})

# Compatibility logic
df["compatible"] = (
    (df["common_interests"] >= 2) &
    (df["availability_overlap"] > 5) &
    (df["activity_match"] > 6)
).astype(int)

print("\n Dataset Preview:")
print(df.head())

# 3. FEATURES & TARGET
X = df.drop("compatible", axis=1)
y = df["compatible"]

# 4. POLYNOMIAL FEATURES
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# 5. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# 6. RANDOM FOREST MODEL
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# 7. PREDICTIONS
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {accuracy:.4f}")

# 8. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("User Compatibility Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show(block=False)   

# 9. FEATURE IMPORTANCE
importances = model.feature_importances_
feature_names = poly.get_feature_names_out()

feat_imp = pd.Series(importances, index=feature_names)
top_features = feat_imp.nlargest(5)

print("\n Top Features Influencing Compatibility:")
print(top_features)

# Plot Feature Importance
plt.figure()
top_features.sort_values().plot(kind='barh')

plt.title("Top Features Driving Compatibility Decisions")
plt.xlabel("Importance Score")

plt.show()   



# 10. SAMPLE PREDICTION
sample_user_pair = np.array([[3, 3, 7.5, 8]])

sample_poly = poly.transform(sample_user_pair)

prediction = model.predict(sample_poly)

print("\n Sample User Compatibility Prediction:")
print("Compatible " if prediction[0] == 1 else "Not Compatible ❌")