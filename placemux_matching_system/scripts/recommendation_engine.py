import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUDENTS = os.path.join(BASE_DIR, "data", "students.csv")
JOBS = os.path.join(BASE_DIR, "data", "jobs.csv")
TRUST = os.path.join(BASE_DIR, "outputs", "trust_signoff.csv")
OUTPUT = os.path.join(BASE_DIR, "outputs", "recommendations.csv")

def calculate_score(student, job):
    score = 0
    score += max(0, 100 - abs(student["python"] - job["python_req"]))
    score += max(0, 100 - abs(student["ml"] - job["ml_req"]))
    score += max(0, 100 - abs(student["sql"] - job["sql_req"]))
    score += max(0, 100 - abs(student["communication"] - job["comm_req"]))
    return round(score / 4, 2)

def generate_recommendations():
    students = pd.read_csv(STUDENTS)
    jobs = pd.read_csv(JOBS)
    trust = pd.read_csv(TRUST)
    recommendations = []
    for _, student in students.iterrows():
        trust_row = trust[trust["student_id"] == student["student_id"]]
        if trust_row.empty:
            continue
        if trust_row.iloc[0]["trust_status"] != "APPROVED":
            continue
        for _, job in jobs.iterrows():
            score = calculate_score(student, job)
            recommendations.append({
                "student_id": student["student_id"],
                "student_name": student["name"],
                "job_role": job["role"],
                "recommendation_score": score,
                "trust_status": trust_row.iloc[0]["trust_status"],
                "reason": "High skill similarity with verified trust."
            })
    df = pd.DataFrame(recommendations)
    df = df.sort_values(
        by="recommendation_score",
        ascending=False
    )
    df.to_csv(
        OUTPUT,
        index=False
    )
    return df

if __name__ == "__main__":
    result = generate_recommendations()
    print("=" * 60)
    print("PLACEMUX RECOMMENDATION ENGINE")
    print("=" * 60)
    print(result.head(10))
    print("\nRecommendations Generated :", len(result))
    print("\nSaved To")
    print(OUTPUT)