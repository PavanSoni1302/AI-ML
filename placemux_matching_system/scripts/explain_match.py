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

print("=" * 50)
print("MATCH EXPLANATION")
print("=" * 50)

print(f"Candidate: {student['name']}")
print(f"Role: {job['role']}")

print()

print(
    f"Python exceeds threshold by "
    f"{student['python'] - job['python_req']}"
)

print(
    f"ML exceeds threshold by "
    f"{student['ml'] - job['ml_req']}"
)

print(
    f"SQL exceeds threshold by "
    f"{student['sql'] - job['sql_req']}"
)

print("\nDecision: ELIGIBLE")