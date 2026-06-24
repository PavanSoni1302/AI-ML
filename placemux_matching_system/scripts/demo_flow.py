import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

companies = pd.read_csv(
    os.path.join(BASE_DIR, "data", "companies.csv")
)

jobs = pd.read_csv(
    os.path.join(BASE_DIR, "data", "jobs.csv")
)

students = pd.read_csv(
    os.path.join(BASE_DIR, "data", "students.csv")
)

company = companies.iloc[0]
job = jobs.iloc[0]

results = []

for _, student in students.iterrows():

    score = (
        (student["python"] - job["python_req"])
        + (student["ml"] - job["ml_req"])
        + (student["sql"] - job["sql_req"])
        + (student["communication"] - job["comm_req"])
    )

    results.append(
        {
            "Candidate": student["name"],
            "Score": score
        }
    )

ranked = pd.DataFrame(results)

ranked = ranked.sort_values(
    by="Score",
    ascending=False
)

print("=" * 60)
print("PLACEMUX END TO END FLOW")
print("=" * 60)

print("\nCompany:")
print(company["company_name"])

print("\nJob:")
print(job["role"])

print("\nTop Candidate:")
print(ranked.iloc[0]["Candidate"])

print("\nMatch Score:")
print(ranked.iloc[0]["Score"])

ranked.to_csv(
    os.path.join(BASE_DIR, "outputs", "final_ranking.csv"),
    index=False
)

print("\nFinal ranking saved.")