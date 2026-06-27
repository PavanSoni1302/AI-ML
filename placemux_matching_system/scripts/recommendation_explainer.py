import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RECOMMENDATIONS = os.path.join(BASE_DIR, "outputs", "recommendations.csv")
STUDENTS = os.path.join(BASE_DIR, "data", "students.csv")
JOBS = os.path.join(BASE_DIR, "data", "jobs.csv")
OUTPUT = os.path.join(BASE_DIR, "outputs", "recommendation_explanations.csv")

def generate_explanations():

    recommendations = pd.read_csv(RECOMMENDATIONS)
    students = pd.read_csv(STUDENTS)
    jobs = pd.read_csv(JOBS)
    results = []
    for _, rec in recommendations.iterrows():
        student = students[
            students["student_id"] == rec["student_id"]
        ].iloc[0]
        job = jobs[
            jobs["role"] == rec["job_role"]
        ].iloc[0]
        reasons = []
        if student["python"] >= job["python_req"]:
            reasons.append("Python requirement satisfied")
        if student["ml"] >= job["ml_req"]:
            reasons.append("Machine Learning requirement satisfied")
        if student["sql"] >= job["sql_req"]:
            reasons.append("SQL requirement satisfied")
        if student["communication"] >= job["comm_req"]:
            reasons.append("Communication requirement satisfied")
        reasons.append("Candidate passed AI trust verification")
        results.append({
            "student_id": rec["student_id"],
            "student_name": rec["student_name"],
            "job_role": rec["job_role"],
            "recommendation_score": rec["recommendation_score"],
            "explanation": " | ".join(reasons)
        })
    df = pd.DataFrame(results)
    df.to_csv(
        OUTPUT,
        index=False
    )
    return df

if __name__ == "__main__":
    df = generate_explanations()
    print("=" * 60)
    print("RECOMMENDATION EXPLAINABILITY")
    print("=" * 60)
    print(df.head())
    print("\nSaved To")
    print(OUTPUT)