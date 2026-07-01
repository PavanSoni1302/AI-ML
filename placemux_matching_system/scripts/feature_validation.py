import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

feature_store_path = os.path.join(
    BASE_DIR,
    "feature_store",
    "feature_store.csv"
)

print("=" * 70)
print("FEATURE STORE VALIDATION ENGINE")
print("=" * 70)

# Check if feature store exists
if not os.path.exists(feature_store_path):
    print("ERROR: Feature Store not found!")
    exit()

# Load Feature Store
df = pd.read_csv(feature_store_path)

print("\nFeature Store Loaded Successfully")

# Validation Checks
missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()

invalid_scores = 0

if "Score" in df.columns:
    invalid_scores = ((df["Score"] < 0) | (df["Score"] > 100)).sum()

empty_candidates = 0

if "Candidate" in df.columns:
    empty_candidates = df["Candidate"].astype(str).str.strip().eq("").sum()

verified_count = 0

if "Verified" in df.columns:
    verified_count = (df["Verified"] == "Yes").sum()

print("\nVALIDATION REPORT")
print("-" * 40)

print(f"Total Records      : {len(df)}")
print(f"Missing Values     : {missing_values}")
print(f"Duplicate Records  : {duplicate_rows}")
print(f"Invalid Scores     : {invalid_scores}")
print(f"Empty Candidates   : {empty_candidates}")
print(f"Verified Features  : {verified_count}")

print("\nOVERALL STATUS")

if (
    missing_values == 0
    and duplicate_rows == 0
    and invalid_scores == 0
    and empty_candidates == 0
):
    print("PASS")
else:
    print("FAIL")

print("=" * 70)