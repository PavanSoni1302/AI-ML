import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students = pd.read_csv(
    os.path.join(BASE_DIR, "data", "students.csv")
)

jobs = pd.read_csv(
    os.path.join(BASE_DIR, "data", "jobs.csv")
)

student = students.iloc[0]  # Pavan
job = jobs.iloc[0]          # AI Engineer

python_gap = student["python"] - job["python_req"]
ml_gap = student["ml"] - job["ml_req"]
sql_gap = student["sql"] - job["sql_req"]
comm_gap = student["communication"] - job["comm_req"]

match_score = (
    python_gap +
    ml_gap +
    sql_gap +
    comm_gap
)

print("=" * 60)
print("MATCH EXPLANATION PAYLOAD")
print("=" * 60)

print(f"\nCandidate : {student['name']}")
print(f"Job       : {job['role']}")
print(f"Score     : {match_score}")

print("\nExplanation")

print(f"✓ Python exceeds requirement by {python_gap}")
print(f"✓ ML exceeds requirement by {ml_gap}")
print(f"✓ SQL exceeds requirement by {sql_gap}")
print(f"✓ Communication exceeds requirement by {comm_gap}")

print("\nDecision")

if match_score > 40:
    print("Strong Match")
elif match_score > 20:
    print("Good Match")
else:
    print("Average Match")