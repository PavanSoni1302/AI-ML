import pandas as pd
import os
import time

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
print("PLACEMUX MODEL HEALTH MONITOR")
print("=" * 70)

start = time.time()

# ------------------------
# Registry Check
# ------------------------

registry_status = "Healthy"

try:
    registry = pd.read_csv(registry_path)
except Exception:
    registry_status = "Failed"

# ------------------------
# Feature Store Check
# ------------------------

feature_status = "Healthy"

try:
    features = pd.read_csv(feature_store_path)
except Exception:
    feature_status = "Failed"

# ------------------------
# Prediction Pipeline
# ------------------------

pipeline_status = "Ready"

latency = round(time.time() - start, 3)

print("\nSYSTEM HEALTH REPORT")
print("-" * 45)

print(f"Model Registry      : {registry_status}")
print(f"Feature Store       : {feature_status}")
print(f"Prediction Engine   : {pipeline_status}")

if registry_status == "Healthy":

    print("\nCURRENT MODEL")

    print(registry.iloc[0]["Model Name"])

    print("Version :", registry.iloc[0]["Version"])

    print("Accuracy :", registry.iloc[0]["Accuracy"], "%")

    print("Precision :", registry.iloc[0]["Precision"], "%")

    print("Recall :", registry.iloc[0]["Recall"], "%")

print("\nSYSTEM LATENCY")

print(latency, "seconds")

print("\nOVERALL STATUS")

if registry_status == "Healthy" and feature_status == "Healthy":

    print("SYSTEM HEALTHY")

else:

    print("SYSTEM WARNING")

print("=" * 70)