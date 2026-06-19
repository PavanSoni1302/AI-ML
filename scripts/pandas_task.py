import pandas as pd

df = pd.read_csv("data/data.csv")

print("Preview:\n", df.head())
print("\nStats:\n", df.describe())
print("\nAverage Score:", df['Score'].mean())