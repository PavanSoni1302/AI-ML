import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students_path = os.path.join(BASE_DIR, "data", "students.csv")
jobs_path = os.path.join(BASE_DIR, "data", "jobs.csv")

students = pd.read_csv(students_path)
jobs = pd.read_csv(jobs_path)

student = students.iloc[0]
job = jobs.iloc[0]

score = (
    student["python"] / job["python_req"] +
    student["ml"] / job["ml_req"] +
    student["sql"] / job["sql_req"]
) / 3

print("="*50)
print("PLACEMUX BASELINE MATCHING")
print("="*50)
print(f"Student: {student['name']}")
print(f"Job: {job['role']}")
print(f"Match Score: {score*100:.2f}%")
print("="*50)