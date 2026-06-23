import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students = pd.read_csv(
    os.path.join(BASE_DIR, "data", "students.csv")
)

jobs = pd.read_csv(
    os.path.join(BASE_DIR, "data", "jobs.csv")
)

job = jobs.iloc[0]

results = []

for _, student in students.iterrows():

    score = (
        (student["python"] - job["python_req"]) * 1.5 +
        (student["ml"] - job["ml_req"]) * 1.5 +
        (student["sql"] - job["sql_req"]) * 1.2 +
        (student["communication"] - job["comm_req"]) * 1.2 +
        student["projects"] * 5 +
        student["experience"] * 8 +
        student["cgpa"] * 2
    )

    results.append({
        "Candidate": student["name"],
        "Conversion Score": round(score,2)
    })

ranked = pd.DataFrame(results)

ranked = ranked.sort_values(
    by="Conversion Score",
    ascending=False
)

print("="*60)
print("CONVERSION TUNED RANKING")
print("="*60)

print(ranked)