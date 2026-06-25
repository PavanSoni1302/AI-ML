import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

resume = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "parsed_resume.csv"
    )
)

job = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "parsed_job.csv"
    )
)

matched = []

for skill in resume["skill"]:

    resume_has = bool(
        resume.loc[
            resume["skill"] == skill,
            "found"
        ].values[0]
    )

    job_requires = bool(
        job.loc[
            job["required_skill"] == skill,
            "required"
        ].values[0]
    )

    if resume_has and job_requires:
        matched.append(skill)

print("="*60)
print("PARSER DEMO")
print("="*60)

print("\nMatched Skills")

for skill in matched:
    print("-", skill)

print("\nTotal Skills Matched")

print(len(matched))