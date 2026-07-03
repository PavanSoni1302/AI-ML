import os
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIVE_MONITOR = os.path.join(
    BASE_DIR,
    "outputs",
    "live_monitoring.csv"
)

DRIFT_REPORT = os.path.join(
    BASE_DIR,
    "outputs",
    "drift_report.csv"
)

FAIRNESS = os.path.join(
    BASE_DIR,
    "outputs",
    "fairness_metrics.csv"
)

SIGNOFF = os.path.join(
    BASE_DIR,
    "outputs",
    "model_signoff.csv"
)

OUTPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "production_health_summary.csv"
)

print("=" * 90)
print("PLACEMUX PRODUCTION HEALTH ENGINE")
print("=" * 90)

monitor = pd.read_csv(LIVE_MONITOR)
drift = pd.read_csv(DRIFT_REPORT)
fairness = pd.read_csv(FAIRNESS)
signoff = pd.read_csv(SIGNOFF)
accuracy = fairness.loc[
    fairness["Metric"] == "Model Accuracy",
    "Value"
].iloc[0]
precision = fairness.loc[
    fairness["Metric"] == "Precision",
    "Value"
].iloc[0]
recall = fairness.loc[
    fairness["Metric"] == "Recall",
    "Value"
].iloc[0]
availability = round(
    monitor["Availability (%)"].mean(),
    2
)
cpu = round(
    monitor["CPU Usage (%)"].mean(),
    2
)
memory = round(
    monitor["Memory Usage (%)"].mean(),
    2
)
latency = round(
    monitor["Latency (sec)"].mean(),
    3
)
healthy_models = len(
    monitor[
        monitor["Health"] == "Healthy"
    ]
)
critical_drift = len(
    drift[
        drift["Drift Status"] == "CRITICAL"
    ]
)
approved_models = len(
    signoff[
        signoff["Status"] == "APPROVED"
    ]
)
health_score = round(
    (
        accuracy +
        precision +
        recall +
        availability
    ) / 4,
    2
)
if critical_drift == 0:
    drift_health = "PASS"
else:
    drift_health = "WARNING"
if approved_models == len(signoff):
    signoff_status = "PASS"
else:
    signoff_status = "FAIL"
if health_score >= 90 and drift_health == "PASS":
    production_status = "READY FOR GO-LIVE"
elif health_score >= 80:
    production_status = "MONITOR CLOSELY"
else:

    production_status = "REVIEW REQUIRED"
summary = pd.DataFrame([{
    "Timestamp":
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Models":
    len(monitor),
    "Healthy Models":
    healthy_models,
    "Average Accuracy":
    accuracy,
    "Average Precision":
    precision,
    "Average Recall":
    recall,
    "Average Availability":
    availability,
    "Average CPU":
    cpu,
    "Average Memory":
    memory,
    "Average Latency":
    latency,
    "Health Score":
    health_score,
    "Drift Health":
    drift_health,
    "Model Sign-off":
    signoff_status,
    "Production Status":
    production_status
}])
summary.to_csv(
    OUTPUT,
    index=False
)
print("\nPRODUCTION HEALTH SUMMARY\n")
print(summary)
print("\n" + "=" * 90)
print("SYSTEM HEALTH")
print("=" * 90)
print(f"Models Monitored      : {len(monitor)}")
print(f"Healthy Models        : {healthy_models}")
print(f"Average Accuracy      : {accuracy}%")
print(f"Average Precision     : {precision}%")
print(f"Average Recall        : {recall}%")
print(f"Average Availability  : {availability}%")
print(f"Average CPU           : {cpu}%")
print(f"Average Memory        : {memory}%")
print(f"Average Latency       : {latency} sec")
print(f"Health Score          : {health_score}")
print(f"Drift Status          : {drift_health}")
print(f"Model Sign-off        : {signoff_status}")
print(f"Production Status     : {production_status}")
print("\nOutput Saved To")
print(OUTPUT)
print("=" * 90)