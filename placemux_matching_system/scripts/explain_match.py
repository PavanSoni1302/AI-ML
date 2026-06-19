import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students = pd.read_csv(os.path.join(BASE_DIR, "data", "students.csv"))
jobs = pd.read_csv(os.path.join(BASE_DIR, "data", "jobs.csv"))

student = students.iloc[0]
job = jobs.iloc[0]

print("=" * 50)
print("MATCH EXPLANATION")
print("=" * 50)

print(f"Student : {student['name']}")
print(f"Job     : {job['role']}")
print()

if student["python"] >= job["python_req"]:
    print("✓ Python requirement satisfied")

if student["ml"] >= job["ml_req"]:
    print("✓ Machine Learning requirement satisfied")

if student["sql"] >= job["sql_req"]:
    print("✓ SQL requirement satisfied")

if student["projects"] >= job["projects_req"]:
    print("✓ Project requirement satisfied")

if student["experience"] >= job["exp_req"]:
    print("✓ Experience requirement satisfied")