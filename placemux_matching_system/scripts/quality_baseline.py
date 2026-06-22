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

average_score = sum(scores) / len(scores)

print("=" * 60)
print("MATCH QUALITY BASELINE")
print("=" * 60)

print("Job:", job["role"])

print("\nAverage Match Score:")
print(round(average_score, 2))

print("\nBaseline Recorded Successfully")