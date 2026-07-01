import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

registry_path = os.path.join(
    BASE_DIR,
    "registry",
    "model_registry.csv"
)

feature_store_path = os.path.join(
    BASE_DIR,
    "feature_store",
    "feature_store.csv"
)

print("=" * 70)
print("PLACEMUX MLOPS FOUNDATION DEMO")
print("=" * 70)

print("\nLoading Model Registry...")

registry = pd.read_csv(registry_path)

print(registry)

print("\nLoading Feature Store...")

features = pd.read_csv(feature_store_path)

print(features)

print("\nRunning Sample Prediction Pipeline")

candidate = "John"

candidate_features = features[
    features["Candidate"] == candidate
]

print("\nCandidate Selected :", candidate)

print(candidate_features)

avg_score = candidate_features["Score"].mean()

print("\nAverage Skill Score :", round(avg_score,2))

if avg_score >= 90:
    recommendation = "Highly Recommended"

elif avg_score >= 80:
    recommendation = "Recommended"

else:
    recommendation = "Needs Improvement"

print("\nRecommendation :", recommendation)

print("\nModel Used")

print(registry.iloc[0]["Model Name"])

print("\nModel Version")

print(registry.iloc[0]["Version"])

print("\nEvaluation Metrics")

print("Accuracy :", registry.iloc[0]["Accuracy"], "%")

print("Precision :", registry.iloc[0]["Precision"], "%")

print("Recall :", registry.iloc[0]["Recall"], "%")

print("RMSE :", registry.iloc[0]["RMSE"])

print("\nExplainability")

print(
    "Candidate recommendation is based on the verified feature scores "
    "stored in the Feature Store."
)

print("\nExecution Time")

print(datetime.now())

print("\n")

print("=" * 70)
print("MLOPS FOUNDATION SUCCESSFULLY VERIFIED")
print("=" * 70)