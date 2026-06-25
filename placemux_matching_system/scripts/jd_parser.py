import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

jd_path = os.path.join(
    BASE_DIR,
    "samples",
    "job_description.txt"
)

with open(jd_path, "r", encoding="utf-8") as file:
    jd = file.read().lower()

skills = [
    "python",
    "machine learning",
    "sql",
    "flask",
    "git",
    "pandas",
    "numpy",
    "scikit-learn",
    "communication",
    "problem solving"
]

parsed = []

for skill in skills:

    parsed.append(
        {
            "required_skill": skill,
            "required": skill in jd
        }
    )

df = pd.DataFrame(parsed)

output = os.path.join(
    BASE_DIR,
    "outputs",
    "parsed_job.csv"
)

df.to_csv(output, index=False)

print("="*60)
print("JOB DESCRIPTION PARSER")
print("="*60)

print(df)

print("\nSaved To")

print(output)