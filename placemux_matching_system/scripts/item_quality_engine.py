import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JOBS = os.path.join(BASE_DIR, "data", "jobs.csv")
OUTPUT = os.path.join(BASE_DIR, "outputs", "weak_items.csv")

def analyze_jobs():
    jobs = pd.read_csv(JOBS)
    weak = []
    for _, row in jobs.iterrows():
        issues = []
        if row["python_req"] < 50:
            issues.append("Low Python requirement")
        if row["ml_req"] < 50:
            issues.append("Low ML requirement")
        if row["sql_req"] < 50:
            issues.append("Low SQL requirement")
        if row["comm_req"] < 50:
            issues.append("Low Communication requirement")
        if row["exp_req"] == 0:
            issues.append("No experience requirement")
        if len(issues) > 0:
            weak.append({
                "job_id": row["job_id"],
                "role": row["role"],
                "issues": " | ".join(issues),
                "status": "Needs Review"
            })
    weak_df = pd.DataFrame(weak)
    weak_df.to_csv(
        OUTPUT,
        index=False
    )
    return weak_df

if __name__ == "__main__":
    report = analyze_jobs()
    print("=" * 60)
    print("ITEM QUALITY ENGINE")
    print("=" * 60)
    if report.empty:
        print("No weak job postings found.")
    else:
        print(report)
    print("\nTotal Weak Jobs :", len(report))
    print("\nReport saved to:")
    print(OUTPUT)