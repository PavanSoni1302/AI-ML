import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

report = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "fairness_report.csv"
    )
)
metrics = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "fairness_metrics.csv"
    )
)
print("="*60)
print("FAIRNESS AUDIT DEMO")
print("="*60)
print("\nMetrics")
print(metrics)
print("\nSample Audit")
print(report.head(10))