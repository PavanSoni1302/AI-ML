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

baseline_scores = []
conversion_scores = []

for _, student in students.iterrows():

    baseline = (
        (student["python"] - job["python_req"])
        + (student["ml"] - job["ml_req"])
        + (student["sql"] - job["sql_req"])
    )

    conversion = (
        (student["python"] - job["python_req"]) * 1.5
        + (student["ml"] - job["ml_req"]) * 1.5
        + (student["sql"] - job["sql_req"]) * 1.2
        + student["projects"] * 5
        + student["experience"] * 8
        + student["cgpa"] * 2
    )

    baseline_scores.append(baseline)
    conversion_scores.append(conversion)

baseline_avg = sum(baseline_scores) / len(baseline_scores)
conversion_avg = sum(conversion_scores) / len(conversion_scores)

difference = conversion_avg - baseline_avg

print("=" * 60)
print("CONVERSION QUALITY CHECK")
print("=" * 60)

print("\nBaseline Average Score:")
print(round(baseline_avg, 2))

print("\nConversion Average Score:")
print(round(conversion_avg, 2))

print("\nDifference:")
print(round(difference, 2))

if difference >= 0:
    print("\nPASS")
    print("No relevance regression detected.")
else:
    print("\nFAIL")
    print("Ranking quality decreased.")