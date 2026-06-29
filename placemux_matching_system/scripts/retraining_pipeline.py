import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "drift_report.csv"
)

OUTPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "retraining_log.csv"
)

def retrain():
    drift = pd.read_csv(INPUT)
    log = []
    for _, row in drift.iterrows():
        if row["status"] == "Retrain":
            action = "Model Retrained"
        elif row["status"] == "Monitor":
            action = "Monitoring"
        else:
            action = "No Action"
        log.append({
            "student_id": row["student_id"],
            "job_role": row["job_role"],
            "drift_status": row["status"],
            "action": action,
            "timestamp": datetime.now()
        })
    log = pd.DataFrame(log)
    log.to_csv(
        OUTPUT,
        index=False
    )
    return log

if __name__ == "__main__":
    log = retrain()
    print("="*60)
    print("RETRAINING PIPELINE")
    print("="*60)
    print(log.head())
    print("\nSaved To")
    print(OUTPUT)