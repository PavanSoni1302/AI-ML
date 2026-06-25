import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

resume_path = os.path.join(
    BASE_DIR,
    "samples",
    "resume.txt"
)

with open(resume_path, "r", encoding="utf-8") as file:
    resume = file.read().lower()

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
            "skill": skill,
            "found": skill in resume
        }
    )

df = pd.DataFrame(parsed)

output = os.path.join(
    BASE_DIR,
    "outputs",
    "parsed_resume.csv"
)

df.to_csv(output, index=False)

print("="*60)
print("RESUME PARSER")
print("="*60)

print(df)

print("\nSaved To")

print(output)