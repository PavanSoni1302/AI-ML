import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {

    "Registry":
    os.path.join(BASE_DIR,"registry","model_registry.csv"),

    "Feature Store":
    os.path.join(BASE_DIR,"feature_store","feature_store.csv"),

    "Fairness Report":
    os.path.join(BASE_DIR,"outputs","fairness_report.csv"),

    "Fairness Metrics":
    os.path.join(BASE_DIR,"outputs","fairness_metrics.csv"),

    "Live Monitoring":
    os.path.join(BASE_DIR,"outputs","live_monitoring.csv"),

    "Drift Report":
    os.path.join(BASE_DIR,"outputs","drift_report.csv"),

    "Production Health":
    os.path.join(BASE_DIR,"outputs","production_health_summary.csv"),

    "Dashboard":
    os.path.join(BASE_DIR,"outputs","monitoring_dashboard.csv"),

    "Model Signoff":
    os.path.join(BASE_DIR,"outputs","model_signoff.csv"),

    "Experiment Log":
    os.path.join(BASE_DIR,"outputs","experiment_log.csv")

}

print("="*100)
print("PLACEMUX FINAL GO-LIVE VALIDATION")
print("="*100)

results=[]

for name,path in FILES.items():

    if os.path.exists(path):

        status="PASS"

    else:

        status="FAIL"

    results.append({

        "Component":name,

        "Status":status

    })

report=pd.DataFrame(results)

print(report)

passed=len(report[
    report["Status"]=="PASS"
])

failed=len(report[
    report["Status"]=="FAIL"
])

print("\n"+"="*100)

print("VALIDATION SUMMARY")

print("="*100)

print("Components Checked :",len(report))

print("Passed :",passed)

print("Failed :",failed)

if failed==0:

    print("\nFINAL RESULT")

    print("PROJECT READY FOR SUBMISSION")

else:

    print("\nFINAL RESULT")

    print("PLEASE FIX MISSING COMPONENTS")

print("\nTimestamp")

print(datetime.now())

print("="*100)