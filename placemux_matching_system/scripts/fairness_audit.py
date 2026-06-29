import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(BASE_DIR, "outputs", "validated_recommendations.csv")
OUTPUT = os.path.join(BASE_DIR, "outputs", "fairness_report.csv")

def run_audit():
    df = pd.read_csv(INPUT)
    report = []
    for _, row in df.iterrows():
        score = row["recommendation_score"]
        status = row["validation_status"]
        if score >= 90:
            fairness = "LOW RISK"
        elif score >= 75:
            fairness = "MEDIUM RISK"
        else:
            fairness = "HIGH RISK"
        report.append({
            "student_id": row["student_id"],
            "student_name": row["student_name"],
            "job_role": row["job_role"],
            "recommendation_score": score,
            "validation_status": status,
            "fairness_level": fairness
        })
    report = pd.DataFrame(report)
    report.to_csv(
        OUTPUT,
        index=False
    )
    return report

if __name__ == "__main__":
    report = run_audit()
    print("="*60)
    print("FAIRNESS AUDIT")
    print("="*60)
    print(report.head())
    print("\nTotal Records :", len(report))
    print("\nSaved To")
    print(OUTPUT)