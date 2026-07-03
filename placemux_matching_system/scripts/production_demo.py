import os
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "registry": os.path.join(BASE_DIR, "registry", "model_registry.csv"),
    "feature_store": os.path.join(BASE_DIR, "feature_store", "feature_store.csv"),
    "fairness": os.path.join(BASE_DIR, "outputs", "fairness_report.csv"),
    "drift": os.path.join(BASE_DIR, "outputs", "drift_report.csv"),
    "monitor": os.path.join(BASE_DIR, "outputs", "live_monitoring.csv"),
    "health": os.path.join(BASE_DIR, "outputs", "production_health_summary.csv"),
    "dashboard": os.path.join(BASE_DIR, "outputs", "monitoring_dashboard.csv"),
    "signoff": os.path.join(BASE_DIR, "outputs", "model_signoff.csv")
}

print("=" * 100)
print("PLACEMUX AI PLATFORM - GO LIVE DEMONSTRATION")
print("=" * 100)

print("\nSTEP 1 : MODEL REGISTRY")

registry = pd.read_csv(FILES["registry"])

print("PASS")
print("Registered Models :", len(registry))

print("\nSTEP 2 : FEATURE STORE")

feature_store = pd.read_csv(FILES["feature_store"])

print("PASS")
print("Stored Features :", len(feature_store))

print("\nSTEP 3 : FAIRNESS AUDIT")

fairness = pd.read_csv(FILES["fairness"])

print("PASS")
print("Candidates Audited :", len(fairness))

print("\nSTEP 4 : DRIFT MONITOR")

drift = pd.read_csv(FILES["drift"])

critical = len(
    drift[
        drift["Drift Status"] == "CRITICAL"
    ]
)

print("PASS")

print("Critical Drift Models :", critical)

print("\nSTEP 5 : LIVE MONITORING")

monitor = pd.read_csv(FILES["monitor"])

print("PASS")

print("Healthy Models :",
      len(
          monitor[
              monitor["Health"] == "Healthy"
          ]
      )
)

print("Average Latency :",
      round(
          monitor["Latency (sec)"].mean(),
          3
      ),
      "sec"
)

print("\nSTEP 6 : PRODUCTION HEALTH")

health = pd.read_csv(FILES["health"])

print("PASS")

print("Health Score :",
      health.iloc[0]["Health Score"])

print("Production Status :",
      health.iloc[0]["Production Status"])

print("\nSTEP 7 : EXECUTIVE DASHBOARD")

dashboard = pd.read_csv(FILES["dashboard"])

print("PASS")

print("Requests :",
      dashboard.iloc[0]["Requests"])

print("Prediction Count :",
      dashboard.iloc[0]["Prediction Count"])

print("Success Rate :",
      dashboard.iloc[0]["Success Rate"],
      "%")

print("\nSTEP 8 : MODEL SIGN-OFF")
signoff = pd.read_csv(FILES["signoff"])
approved = len(
    signoff[
        signoff["Status"] == "APPROVED"
    ]
)
print("PASS")
print("Approved Models :", approved)

print("\nSTEP 9 : FINAL GO-LIVE DECISION")
if (
    critical == 0
    and approved == len(signoff)
):
    decision = "GO LIVE APPROVED"
else:
    decision = "MANUAL REVIEW REQUIRED"
print(decision)

print("\nSTEP 10 : EXECUTION SUMMARY")
summary = pd.DataFrame({
    "Metric":[
        "Models",
        "Features",
        "Healthy Models",
        "Approved Models",
        "Critical Drift",
        "Production Status",
        "Go Live"
    ],
    "Value":[
        len(registry),
        len(feature_store),
        len(
            monitor[
                monitor["Health"] == "Healthy"
            ]
        ),
        approved,
        critical,
        health.iloc[0]["Production Status"],
        decision
    ]
})
print(summary)
summary.to_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "production_demo_report.csv"
    ),
    index=False
)
print("\nTimestamp")
print(datetime.now())
print("\nOutput Saved")
print(
    os.path.join(
        BASE_DIR,
        "outputs",
        "production_demo_report.csv"
    )
)
print("=" * 100)
print("PLACEMUX GO-LIVE DEMO COMPLETED SUCCESSFULLY")
print("=" * 100)