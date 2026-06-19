from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
data = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Target column
df['MedHouseVal'] = data.target

print("Dataset Overview:")
print(df.head())

print("\nBasic Statistics:")
print(df.describe())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features & target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
predictions = model.predict(X_test_scaled)

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nSample Predictions:")
print(predictions[:5])

print("\nActual Values:")
print(y_test.values[:5])

print(f"Mean Absolute Error: ${mae * 100000:.2f}")
print(f"R-Squared Score: {r2:.2f}")