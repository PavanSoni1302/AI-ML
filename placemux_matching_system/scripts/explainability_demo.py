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

    results.append((job, score))

best_job, best_score = max(
    results,
    key=lambda x: x[1]
)

python_gap = student["python"] - best_job["python_req"]
ml_gap = student["ml"] - best_job["ml_req"]
sql_gap = student["sql"] - best_job["sql_req"]
communication_gap = (
    student["communication"]
    - best_job["comm_req"]
)

report = pd.DataFrame([
    {
        "student": student["name"],
        "recommended_job": best_job["role"],
        "match_score": best_score,
        "python_gap": python_gap,
        "ml_gap": ml_gap,
        "sql_gap": sql_gap,
        "communication_gap": communication_gap
    }
])

output_path = os.path.join(
    BASE_DIR,
    "outputs",
    "explainability_report.csv"
)

report.to_csv(
    output_path,
    index=False
)

print("=" * 60)
print("MATCH EXPLAINABILITY")
print("=" * 60)

print(f"\nStudent: {student['name']}")
print(f"Recommended Job: {best_job['role']}")
print(f"Match Score: {best_score}")

print("\nExplanation:")

print(f"Python Gap: {python_gap}")
print(f"ML Gap: {ml_gap}")
print(f"SQL Gap: {sql_gap}")
print(f"Communication Gap: {communication_gap}")

print("\nDecision:")

if best_score > 40:
    print("Strong Match")
elif best_score > 20:
    print("Good Match")
else:
    print("Average Match")

print("\nExplainability Report Saved To:")
print(output_path)