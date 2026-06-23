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
job = jobs.iloc[0]

score = (
    (student["python"] - job["python_req"])
    + (student["ml"] - job["ml_req"])
    + (student["sql"] - job["sql_req"])
)

print("="*60)
print("SPEND QUALITY GUARDRAIL")
print("="*60)

print("Student:", student["name"])
print("Job:", job["role"])

print("\nMatch Score:", score)

if score < 20:
    print("\nWARNING")
    print("Low Fit")
    print("Application may not be worth paying for.")
else:
    print("\nGOOD FIT")
    print("Application recommended.")