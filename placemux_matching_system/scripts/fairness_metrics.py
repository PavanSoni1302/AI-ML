import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPORT = os.path.join(
    BASE_DIR,
    "outputs",
    "fairness_report.csv"
)
OUTPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "fairness_metrics.csv"
)
df = pd.read_csv(REPORT)
metrics = pd.DataFrame([{
    "Total Candidates": len(df),
    "Low Risk":
    len(df[df["fairness_level"]=="LOW RISK"]),
    "Medium Risk":
    len(df[df["fairness_level"]=="MEDIUM RISK"]),
    "High Risk":
    len(df[df["fairness_level"]=="HIGH RISK"]),
    "Average Recommendation Score":
    round(df["recommendation_score"].mean(),2)
}])
metrics.to_csv(
    OUTPUT,
    index=False
)
print(metrics)