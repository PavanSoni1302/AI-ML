import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "validated_recommendations.csv"
)

OUTPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "drift_report.csv"
)

def monitor():
    df = pd.read_csv(INPUT)
    baseline = 85.0
    report = []
    for _, row in df.iterrows():
        current = row["recommendation_score"]
        drift = round(current - baseline, 2)
        if abs(drift) <= 5:
            status = "Stable"
        elif abs(drift) <= 15:
            status = "Monitor"
        else:
            status = "Retrain"
        report.append({
            "student_id": row["student_id"],
            "student_name": row["student_name"],
            "job_role": row["job_role"],
            "baseline_score": baseline,
            "current_score": current,
            "drift": drift,
            "status": status
        })
    report = pd.DataFrame(report)
    report.to_csv(
        OUTPUT,
        index=False
    )
    return report

if __name__ == "__main__":
    report = monitor()
    print("="*60)
    print("MODEL DRIFT MONITOR")
    print("="*60)
    print(report.head())
    print("\nSaved To")
    print(OUTPUT)