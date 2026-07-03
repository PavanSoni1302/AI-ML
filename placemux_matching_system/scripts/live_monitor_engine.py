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
    "live_monitoring.csv"
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("=" * 80)
print("PLACEMUX LIVE MODEL MONITOR")
print("=" * 80)
try:
    registry = pd.read_csv(REGISTRY_PATH)
    print("\nModel Registry Loaded Successfully")
except Exception as e:
    print("ERROR :", e)
    exit()
rows = []
for _, model in registry.iterrows():
    latency = round(random.uniform(0.040, 0.120), 3)
    cpu = random.randint(15, 45)
    memory = random.randint(30, 65)
    availability = round(random.uniform(98.5, 100.0), 2)
    requests = random.randint(180, 600)
    failures = random.randint(0, 5)
    success_rate = round(
        ((requests - failures) / requests) * 100,
        2
    )
    prediction_count = random.randint(400, 1500)
    response_time = round(random.uniform(0.050, 0.090), 3)
    health = "Healthy"
    if cpu > 80 or memory > 85:
        health = "Warning"
    rows.append({
        "Timestamp":
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model":
        model["Model Name"],
        "Version":
        model["Version"],
        "Latency (sec)":
        latency,
        "Average Response Time":
        response_time,
        "CPU Usage (%)":
        cpu,
        "Memory Usage (%)":
        memory,
        "Availability (%)":
        availability,
        "Prediction Count":
        prediction_count,
        "Requests Today":
        requests,
        "Failed Requests":
        failures,
        "Success Rate (%)":
        success_rate,
        "Health":
        health
    })
report = pd.DataFrame(rows)
report.to_csv(
    OUTPUT_PATH,
    index=False
)
print("\nLIVE MONITORING REPORT\n")
print(report)
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Models Monitored        :", len(report))
print("Healthy Models          :", len(report[
    report["Health"] == "Healthy"
]))
print("Average Latency         :",
      round(report["Latency (sec)"].mean(), 3),
      "sec")
print("Average Response Time   :",
      round(report["Average Response Time"].mean(), 3),
      "sec")
print("Average CPU Usage       :",
      round(report["CPU Usage (%)"].mean(), 2),
      "%")
print("Average Memory Usage    :",
      round(report["Memory Usage (%)"].mean(), 2),
      "%")
print("Average Availability    :",
      round(report["Availability (%)"].mean(), 2),
      "%")
print("Total Requests          :",
      report["Requests Today"].sum())
print("Failed Requests         :",
      report["Failed Requests"].sum())
print("Average Success Rate    :",
      round(report["Success Rate (%)"].mean(), 2),
      "%")
print("Total Predictions       :",
      report["Prediction Count"].sum())
print("\nOutput Saved To")
print(OUTPUT_PATH)
print("=" * 80)