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

python_gap = student["python"] - job["python_req"]
ml_gap = student["ml"] - job["ml_req"]
sql_gap = student["sql"] - job["sql_req"]

print("=" * 50)
print("MATCH VECTOR")
print("=" * 50)

print(f"Student: {student['name']}")
print(f"Job: {job['role']}")

print("\nMatch Vector")

print(f"Python Gap: {python_gap}")
print(f"ML Gap: {ml_gap}")
print(f"SQL Gap: {sql_gap}")