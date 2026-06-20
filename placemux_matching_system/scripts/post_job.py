import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

jobs = pd.read_csv(
    os.path.join(BASE_DIR, "data", "jobs.csv")
)

print("=" * 50)
print("POSTED JOBS")
print("=" * 50)

for _, job in jobs.iterrows():
    print(f"\nJob ID: {job['job_id']}")
    print(f"Role: {job['role']}")
    print(f"Python Threshold: {job['python_req']}")
    print(f"ML Threshold: {job['ml_req']}")
    print(f"SQL Threshold: {job['sql_req']}")