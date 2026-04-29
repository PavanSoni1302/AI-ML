import pandas as pd

# Load dataset
df = pd.read_csv("data/data.csv")

# 1. Identify missing values
print("Missing Values:\n", df.isnull().sum())

# 2. Fill missing Age with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# 3. Fill missing Score with 0
df['Score'] = df['Score'].fillna(0)

print("\nCleaned Data:\n", df)