import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students = pd.read_csv(
    os.path.join(BASE_DIR, "data", "students.csv")
)

jobs = pd.read_csv(
    os.path.join(BASE_DIR, "data", "jobs.csv")
)

student = students.iloc[0]   # Pavan

results = []

for _, job in jobs.iterrows():

    score = (
        (student["python"] - job["python_req"]) +
        (student["ml"] - job["ml_req"]) +
        (student["sql"] - job["sql_req"]) +
        (student["communication"] - job["comm_req"])
    )

    results.append({
        "Job": job["role"],
        "Match Score": score
    })

ranked_jobs = pd.DataFrame(results)

ranked_jobs = ranked_jobs.sort_values(
    by="Match Score",
    ascending=False
)

print("=" * 50)
print("JOB SEARCH RESULTS")
print("=" * 50)

print(f"\nStudent: {student['name']}\n")

print(ranked_jobs)