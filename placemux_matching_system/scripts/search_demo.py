import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students = pd.read_csv(
    os.path.join(BASE_DIR, "data", "students.csv")
)

jobs = pd.read_csv(
    os.path.join(BASE_DIR, "data", "jobs.csv")
)

student = students.iloc[0]

results = []

for _, job in jobs.iterrows():

    score = (
        (student["python"] - job["python_req"])
        + (student["ml"] - job["ml_req"])
        + (student["sql"] - job["sql_req"])
        + (student["communication"] - job["comm_req"])
    )

    results.append(
        {
            "Job": job["role"],
            "Score": score
        }
    )

ranked_jobs = pd.DataFrame(results)

ranked_jobs = ranked_jobs.sort_values(
    by="Score",
    ascending=False
)

top_job = ranked_jobs.iloc[0]

output_path = os.path.join(
    BASE_DIR,
    "outputs",
    "search_results.csv"
)

ranked_jobs.to_csv(
    output_path,
    index=False
)

print("=" * 60)
print("PLACEMUX SEARCH DISCOVERY")
print("=" * 60)

print("\nStudent:")
print(student["name"])

print("\nTop 5 Recommended Jobs")
print(ranked_jobs.head())

print("\nBest Match")
print(top_job["Job"])

print("\nScore")
print(top_job["Score"])

print("\nSearch Results Saved To:")
print(output_path)