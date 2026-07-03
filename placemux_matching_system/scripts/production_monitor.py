import os
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGISTRY = os.path.join(
    BASE_DIR,
    "registry",
    "model_registry.csv"
)

MONITOR = os.path.join(
    BASE_DIR,
    "outputs",
    "live_monitoring.csv"
)

FAIRNESS = os.path.join(
    BASE_DIR,
    "outputs",
    "fairness_metrics.csv"
)

DRIFT = os.path.join(
    BASE_DIR,
    "outputs",
    "drift_report.csv"
)

OUTPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "production_health.csv"
)

print("="*90)
print("PLACEMUX PRODUCTION MONITOR")
print("="*90)

registry = pd.read_csv(REGISTRY)
monitor = pd.read_csv(MONITOR)
fairness = pd.read_csv(FAIRNESS)
drift = pd.read_csv(DRIFT)

accuracy = fairness.loc[
    fairness["Metric"]=="Model Accuracy",
    "Value"
].iloc[0]

precision = fairness.loc[
    fairness["Metric"]=="Precision",
    "Value"
].iloc[0]

recall = fairness.loc[
    fairness["Metric"]=="Recall",
    "Value"
].iloc[0]
rows=[]
for _,model in registry.iterrows():
    live = monitor[
        monitor["Model"]==model["Model Name"]
    ].iloc[0]
    drift_info = drift[
        drift["Model"]==model["Model Name"]
    ].iloc[0]
    cpu = live["CPU Usage (%)"]
    memory = live["Memory Usage (%)"]
    availability = live["Availability (%)"]
    latency = live["Latency (sec)"]
    health_score = round(
        (
            accuracy +
            precision +
            recall +
            availability
        ) / 4,
        2
    )
    production = "READY"
    if drift_info["Production Ready"] == "NO":
        production = "REVIEW"
    rows.append({
        "Timestamp":
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model":
        model["Model Name"],
        "Version":
        model["Version"],
        "Accuracy":
        accuracy,
        "Precision":
        precision,
        "Recall":
        recall,
        "Latency":
        latency,
        "CPU":
        cpu,
        "Memory":
        memory,
        "Availability":
        availability,
        "Drift":
        drift_info["Drift Score"],
        "Drift Status":
        drift_info["Drift Status"],
        "Health Score":
        health_score,
        "Production":
        production
    })
report = pd.DataFrame(rows)
report.to_csv(
    OUTPUT,
    index=False
)
print("\nPRODUCTION REPORT\n")
print(report)
print("\n"+"="*90)
print("PRODUCTION SUMMARY")
print("="*90)
print("Models                  :",len(report))
print("Average Health Score    :",
      round(report["Health Score"].mean(),2))
print("Average Latency         :",
      round(report["Latency"].mean(),3),"sec")
print("Average CPU             :",
      round(report["CPU"].mean(),2),"%")
print("Average Memory          :",
      round(report["Memory"].mean(),2),"%")
print("Average Availability    :",
      round(report["Availability"].mean(),2),"%")
print("Ready Models            :",
      len(report[
          report["Production"]=="READY"
      ]))
print("Review Required         :",
      len(report[
          report["Production"]=="REVIEW"
      ]))
overall="GO LIVE READY"
if len(report[
    report["Production"]=="REVIEW"
])>0:
    overall="MANUAL REVIEW REQUIRED"
print("\nOverall Production Status")
print(overall)
print("\nOutput Saved To")
print(OUTPUT)
print("="*90)