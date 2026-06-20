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
print("THRESHOLD VALIDATION")
print("=" * 50)

python_pass = student["python"] >= job["python_req"]
ml_pass = student["ml"] >= job["ml_req"]
sql_pass = student["sql"] >= job["sql_req"]

print("Python:", "PASS" if python_pass else "FAIL")
print("ML:", "PASS" if ml_pass else "FAIL")
print("SQL:", "PASS" if sql_pass else "FAIL")

if python_pass and ml_pass and sql_pass:
    print("\nFINAL RESULT: ELIGIBLE")
else:
    print("\nFINAL RESULT: NOT ELIGIBLE")