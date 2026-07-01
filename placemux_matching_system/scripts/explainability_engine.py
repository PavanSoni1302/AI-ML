import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

feature_store_path = os.path.join(
    BASE_DIR,
    "feature_store",
    "feature_store.csv"
)

output_path = os.path.join(
    BASE_DIR,
    "outputs",
    "explainability_report.csv"
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

print("=" * 70)
print("PLACEMUX EXPLAINABILITY ENGINE")
print("=" * 70)

df = pd.read_csv(feature_store_path)

results = []

for candidate in df["Candidate"].unique():

    data = df[df["Candidate"] == candidate]

    avg_score = data["Score"].mean()

    skills = ", ".join(data["Skill"].tolist())

    if avg_score >= 90:
        recommendation = "Highly Recommended"

    elif avg_score >= 80:
        recommendation = "Recommended"

    else:
        recommendation = "Needs Improvement"

    reason = (
        f"{candidate} has verified skills in {skills}. "
        f"Average score is {avg_score:.1f}, therefore the recommendation is '{recommendation}'."
    )

    results.append({

        "Candidate": candidate,

        "Average Score": round(avg_score,2),

        "Recommendation": recommendation,

        "Reason": reason

    })

report = pd.DataFrame(results)

report.to_csv(output_path,index=False)

print(report)

print("\nExplainability Report Saved Successfully")

print(output_path)

print("=" * 70)