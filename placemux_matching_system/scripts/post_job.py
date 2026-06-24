import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

jobs_file = os.path.join(
    BASE_DIR,
    "data",
    "jobs.csv"
)

jobs = pd.read_csv(jobs_file)

new_job = {
    "role": "Data Analyst",
    "python_req": 70,
    "ml_req": 60,
    "sql_req": 80,
    "comm_req": 75
}

jobs = pd.concat(
    [jobs, pd.DataFrame([new_job])],
    ignore_index=True
)

jobs.to_csv(
    jobs_file,
    index=False
)

print("=" * 60)
print("JOB POSTING")
print("=" * 60)

print("\nJob Posted Successfully")

print(new_job)

print("\nTotal Jobs")

print(len(jobs))