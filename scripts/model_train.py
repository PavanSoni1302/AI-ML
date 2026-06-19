from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [35, 40, 55, 60, 68, 72, 81, 88, 92, 95]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Score']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test set
predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Actual:", y_test.values)

# Bonus: predict for 11 hours
future_score = model.predict([[11]])
print(f"Predicted score for 11 hours: {future_score[0]:.2f}")