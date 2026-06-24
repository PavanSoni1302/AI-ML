import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students = pd.read_csv(
    os.path.join(BASE_DIR, "data", "students.csv")
)

jobs = pd.read_csv(
    os.path.join(BASE_DIR, "data", "jobs.csv")
)

warnings = []

for _, student in students.iterrows():

    for _, job in jobs.iterrows():

        score = (
            (student["python"] - job["python_req"])
            + (student["ml"] - job["ml_req"])
            + (student["sql"] - job["sql_req"])
        )

        status = "GOOD FIT"

        if score < 20:
            status = "LOW FIT WARNING"

        warnings.append(
            {
                "Student": student["name"],
                "Job": job["role"],
                "Score": score,
                "Status": status
            }
        )

df = pd.DataFrame(warnings)

print("=" * 60)
print("SPEND QUALITY GUARDRAIL")
print("=" * 60)

print(df.head(10))

df.to_csv(
    os.path.join(BASE_DIR, "outputs", "guardrail_report.csv"),
    index=False
)

print("\nGuardrail report saved.")