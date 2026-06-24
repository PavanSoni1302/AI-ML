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

scores = []

for _, student in students.iterrows():

    score = (
        (student["python"] - job["python_req"])
        + (student["ml"] - job["ml_req"])
        + (student["sql"] - job["sql_req"])
        + (student["communication"] - job["comm_req"])
    )

    scores.append(score)

average_score = round(sum(scores) / len(scores), 2)

best_score = max(scores)

worst_score = min(scores)

status = "APPROVED"

if average_score < 0:
    status = "REJECTED"

report = pd.DataFrame([
    {
        "job": job["role"],
        "average_score": average_score,
        "best_score": best_score,
        "worst_score": worst_score,
        "status": status
    }
])

output_path = os.path.join(
    BASE_DIR,
    "outputs",
    "quality_signoff.csv"
)

report.to_csv(
    output_path,
    index=False
)

print("=" * 60)
print("QUALITY SIGNOFF")
print("=" * 60)

print(report)

print("\nSaved To")

print(output_path)