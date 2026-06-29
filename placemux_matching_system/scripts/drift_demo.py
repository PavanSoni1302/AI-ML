import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

report = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "drift_report.csv"
    )
)

log = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "retraining_log.csv"
    )
)

print("="*60)
print("DRIFT MONITORING DEMO")
print("="*60)
print("\nDrift Summary")
print(report["status"].value_counts())
print("\nRetraining Actions")
print(log["action"].value_counts())
print("\nSample Records")
print(report.head(10))