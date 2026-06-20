import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students = pd.read_csv(
    os.path.join(BASE_DIR, "data", "students.csv")
)

job_python = 80
job_ml = 85
job_sql = 60

students["match_score"] = (
    (students["python"] - job_python) +
    (students["ml"] - job_ml) +
    (students["sql"] - job_sql)
)

ranked = students.sort_values(
    by="match_score",
    ascending=False
)

print("=" * 50)
print("CANDIDATE RANKING")
print("=" * 50)

print(
    ranked[
        ["name", "match_score"]
    ]
)