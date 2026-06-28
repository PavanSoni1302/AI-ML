import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JOBS_FILE = os.path.join(BASE_DIR, "data", "jobs.csv")
WEAK_FILE = os.path.join(BASE_DIR, "outputs", "weak_items.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "recruiter_dashboard.csv")

def build_dashboard():
    jobs = pd.read_csv(JOBS_FILE)
    if os.path.exists(WEAK_FILE):
        weak = pd.read_csv(WEAK_FILE)
        weak_jobs = len(weak)
    else:
        weak_jobs = 0
    dashboard = pd.DataFrame([{
        "Total Jobs": len(jobs),
        "Weak Jobs": weak_jobs,
        "Healthy Jobs": len(jobs) - weak_jobs,
        "Quality Score (%)": round(((len(jobs)-weak_jobs)/len(jobs))*100,2)
    }])
    dashboard.to_csv(OUTPUT_FILE, index=False)
    return dashboard

if __name__ == "__main__":
    dashboard = build_dashboard()
    print("="*60)
    print("RECRUITER DASHBOARD")
    print("="*60)
    print(dashboard)
    print("\nSaved To")
    print(OUTPUT_FILE)