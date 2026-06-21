import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students = pd.read_csv(
    os.path.join(BASE_DIR, "data", "students.csv")
)

jobs = pd.read_csv(
    os.path.join(BASE_DIR, "data", "jobs.csv")
)

job = jobs.iloc[0]     # AI Engineer

results = []

for _, student in students.iterrows():

    score = (
        (student["python"] - job["python_req"]) +
        (student["ml"] - job["ml_req"]) +
        (student["sql"] - job["sql_req"]) +
        (student["communication"] - job["comm_req"])
    )

    results.append({
        "Candidate": student["name"],
        "Match Score": score
    })

ranked_candidates = pd.DataFrame(results)

ranked_candidates = ranked_candidates.sort_values(
    by="Match Score",
    ascending=False
)

print("=" * 50)
print("CANDIDATE SEARCH RESULTS")
print("=" * 50)

print(f"\nJob: {job['role']}\n")

print(ranked_candidates)