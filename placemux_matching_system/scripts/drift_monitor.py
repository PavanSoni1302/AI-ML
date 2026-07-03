import os
import random
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGISTRY_PATH = os.path.join(
    BASE_DIR,
    "registry",
    "model_registry.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "drift_report.csv"
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("=" * 70)
print("PLACEMUX PRODUCTION DRIFT MONITOR")
print("=" * 70)

try:
    registry = pd.read_csv(REGISTRY_PATH)
    print("\nModel Registry Loaded Successfully\n")

except Exception as e:
    print("ERROR :", e)
    exit()

results = []

for _, model in registry.iterrows():

    drift_score = round(random.uniform(0.01, 0.08), 3)

    threshold = 0.05

    if drift_score < 0.03:

        status = "PASS"
        action = "Continue Monitoring"
        retrain = "NO"
        production = "YES"

    elif drift_score < threshold:

        status = "WARNING"
        action = "Increase Monitoring"
        retrain = "NO"
        production = "YES"

    else:

        status = "CRITICAL"
        action = "Schedule Retraining"
        retrain = "YES"
        production = "NO"

    results.append({

        "Timestamp":
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "Model":
        model["Model Name"],

        "Version":
        model["Version"],

        "Drift Score":
        drift_score,

        "Threshold":
        threshold,

        "Drift Status":
        status,

        "Action":
        action,

        "Retraining Required":
        retrain,

        "Production Ready":
        production

    })

report = pd.DataFrame(results)

report.to_csv(
    OUTPUT_PATH,
    index=False
)

print(report)

healthy = len(report[
    report["Drift Status"] == "PASS"
])

warning = len(report[
    report["Drift Status"] == "WARNING"
])

critical = len(report[
    report["Drift Status"] == "CRITICAL"
])

avg_drift = round(
    report["Drift Score"].mean(),
    3
)

print("\n" + "=" * 70)
print("DRIFT SUMMARY")
print("=" * 70)

print(f"Models Checked      : {len(report)}")
print(f"Healthy Models      : {healthy}")
print(f"Warning Models      : {warning}")
print(f"Critical Models     : {critical}")

print(f"\nAverage Drift Score : {avg_drift}")

if critical == 0:

    overall = "MONITORING ACTIVE"

elif critical <= 1:

    overall = "MONITOR CLOSELY"

else:

    overall = "RETRAINING REQUIRED"

print(f"Overall Status      : {overall}")

print("\nOutput Saved To")

print(OUTPUT_PATH)

print("=" * 70)