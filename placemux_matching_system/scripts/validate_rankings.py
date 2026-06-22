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
        (student["python"] - job["python_req"])
        + (student["ml"] - job["ml_req"])
        + (student["sql"] - job["sql_req"])
        + (student["communication"] - job["comm_req"])
    )

    results.append(
        {
            "Candidate": student["name"],
            "Score": score
        }
    )

ranked = pd.DataFrame(results)

ranked = ranked.sort_values(
    by="Score",
    ascending=False
)

print("=" * 60)
print("RANKING VALIDATION")
print("=" * 60)

print("\nJob:", job["role"])

print("\nTop Candidates")

print(ranked)

print("\nValidation Result")

if ranked.iloc[0]["Score"] > ranked.iloc[-1]["Score"]:
    print("PASS")
else:
    print("FAIL")