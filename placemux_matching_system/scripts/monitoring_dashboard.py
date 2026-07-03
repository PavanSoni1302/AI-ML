import os
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIVE_MONITOR = os.path.join(
    BASE_DIR,
    "outputs",
    "live_monitoring.csv"
)

PRODUCTION_HEALTH = os.path.join(
    BASE_DIR,
    "outputs",
    "production_health_summary.csv"
)

DRIFT_REPORT = os.path.join(
    BASE_DIR,
    "outputs",
    "drift_report.csv"
)

OUTPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "monitoring_dashboard.csv"
)

print("=" * 90)
print("PLACEMUX EXECUTIVE MONITORING DASHBOARD")
print("=" * 90)

monitor = pd.read_csv(LIVE_MONITOR)
health = pd.read_csv(PRODUCTION_HEALTH)
drift = pd.read_csv(DRIFT_REPORT)
healthy_models = len(
    monitor[
        monitor["Health"] == "Healthy"
    ]
)
warning_models = len(
    monitor[
        monitor["Health"] == "Warning"
    ]
)
critical_models = len(
    drift[
        drift["Drift Status"] == "CRITICAL"
    ]
)
dashboard = pd.DataFrame([{
    "Timestamp":
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Models":
    len(monitor),
    "Healthy Models":
    healthy_models,
    "Warning Models":
    warning_models,
    "Critical Models":
    critical_models,
    "Average CPU":
    round(
        monitor["CPU Usage (%)"].mean(),
        2
    ),
    "Average Memory":
    round(
        monitor["Memory Usage (%)"].mean(),
        2
    ),
    "Average Latency":
    round(
        monitor["Latency (sec)"].mean(),
        3
    ),
    "Average Response":
    round(
        monitor["Average Response Time"].mean(),
        3
    ),
    "Availability":
    round(
        monitor["Availability (%)"].mean(),
        2
    ),
    "Prediction Count":
    monitor["Prediction Count"].sum(),
    "Requests":
    monitor["Requests Today"].sum(),
    "Failures":
    monitor["Failed Requests"].sum(),
    "Success Rate":
    round(
        monitor["Success Rate (%)"].mean(),
        2
    ),
    "Health Score":
    health.iloc[0]["Health Score"],
    "Production Status":
    health.iloc[0]["Production Status"]
}])
dashboard.to_csv(
    OUTPUT,
    index=False
)
print("\nEXECUTIVE DASHBOARD\n")
print(dashboard)
print("\n" + "=" * 90)
print("SYSTEM SUMMARY")
print("=" * 90)
print(f"Models                  : {dashboard.iloc[0]['Models']}")
print(f"Healthy Models          : {dashboard.iloc[0]['Healthy Models']}")
print(f"Warning Models          : {dashboard.iloc[0]['Warning Models']}")
print(f"Critical Models         : {dashboard.iloc[0]['Critical Models']}")
print(f"Prediction Count        : {dashboard.iloc[0]['Prediction Count']}")
print(f"Requests                : {dashboard.iloc[0]['Requests']}")
print(f"Failures                : {dashboard.iloc[0]['Failures']}")
print(f"Success Rate            : {dashboard.iloc[0]['Success Rate']}%")
print(f"Health Score            : {dashboard.iloc[0]['Health Score']}")
print(f"Production Status       : {dashboard.iloc[0]['Production Status']}")
print("\nDashboard Saved To")
print(OUTPUT)
print("=" * 90)