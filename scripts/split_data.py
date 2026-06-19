from sklearn.model_selection import train_test_split
import pandas as pd

# Sample dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [35, 40, 55, 60, 68, 72, 81, 88, 92, 95]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training items: {len(X_train)}")
print(f"Testing items: {len(X_test)}")