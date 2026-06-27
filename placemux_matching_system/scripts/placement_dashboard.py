import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RECOMMENDATIONS = os.path.join(
    BASE_DIR,
    "outputs",
    "recommendations.csv"
)

OUTPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "placement_dashboard.csv"
)

def build_dashboard():
    df = pd.read_csv(RECOMMENDATIONS)
    dashboard = []
    total_students = df["student_id"].nunique()
    total_jobs = df["job_role"].nunique()
    average_score = round(
        df["recommendation_score"].mean(),
        2
    )
    highest_score = round(
        df["recommendation_score"].max(),
        2
    )
    placement_percentage = round(
        (len(df[df["recommendation_score"] >= 80]) /
         len(df)) * 100,
        2
    )
    dashboard.append({
        "Total Students": total_students,
        "Total Jobs": total_jobs,
        "Average Score": average_score,
        "Highest Score": highest_score,
        "Placement %": placement_percentage                 
    }) 
    dashboard_df = pd.DataFrame(dashboard)
    dashboard_df.to_csv(
        OUTPUT,
        index=False
    )
    return dashboard_df

if __name__ == "__main__":
    dashboard = build_dashboard()
    print("=" * 60)
    print("PLACEMENT DASHBOARD")
    print("=" * 60)
    print(dashboard)
    print("\nSaved To")
    print(OUTPUT)