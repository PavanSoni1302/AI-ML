import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

feature_store = os.path.join(
    BASE_DIR,
    "feature_store",
    "feature_store.csv"
)

print("="*70)
print("EDGE CASE VALIDATION")
print("="*70)

df = pd.read_csv(feature_store)

print("\nChecking Missing Values...")

print(df.isnull().sum())

print("\nChecking Duplicate Records...")

print(df.duplicated().sum())

print("\nChecking Invalid Scores...")

invalid = df[
    (df["Score"] < 0) |
    (df["Score"] > 100)
]

print(len(invalid))

print("\nChecking Empty Skills...")

empty = df[
    df["Skill"].astype(str).str.strip() == ""
]

print(len(empty))

print("\nChecking Empty Candidate Names...")

empty_candidate = df[
    df["Candidate"].astype(str).str.strip() == ""
]

print(len(empty_candidate))

print("\nEDGE CASE STATUS")

if (
    len(invalid)==0 and
    len(empty)==0 and
    len(empty_candidate)==0
):
    print("PASS")
else:
    print("WARNING")

print("="*70)