import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    'Hours_Sleep': [8, 7, 6, 5, 8, 4, 9, 5, 6, 4],
    'Coffee_Cups': [1, 2, 2, 4, 0, 5, 1, 4, 3, 6],
    'Passed': [1, 1, 1, 0, 1, 0, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

X = df[['Hours_Sleep', 'Coffee_Cups']]
y = df['Passed']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)

print("Predictions for test data:", predictions)
print("Actual values:", y_test.values)

# Manual custom prediction
manual_input = pd.DataFrame({
    'Hours_Sleep': [3],
    'Coffee_Cups': [7]
})

manual_prediction = clf.predict(manual_input)

print("Manual Prediction (3 hrs sleep, 7 coffees):", manual_prediction)