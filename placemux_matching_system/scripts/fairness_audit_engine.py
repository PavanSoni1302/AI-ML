import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

feature_store = os.path.join(
    BASE_DIR,
    "feature_store",
    "feature_store.csv"
)

output = os.path.join(
    BASE_DIR,
    "outputs",
    "fairness_report.csv"
)

os.makedirs(os.path.dirname(output), exist_ok=True)

df = pd.read_csv(feature_store)

results = []

for candidate in df["Candidate"].unique():

    data = df[df["Candidate"] == candidate]

    avg = data["Score"].mean()

    if avg >= 90:
        recommendation = "Highly Recommended"

    elif avg >= 80:
        recommendation = "Recommended"

    else:
        recommendation = "Needs Improvement"

    results.append({

        "Candidate": candidate,

        "Average Score": round(avg,2),

        "Recommendation": recommendation

    })

report = pd.DataFrame(results)

distribution = report["Recommendation"].value_counts()

fairness = "PASS"

if distribution.max()/len(report) > 0.80:
    fairness = "WARNING"

report["Fairness Status"] = fairness

report.to_csv(output,index=False)

print("="*70)
print("FAIRNESS AUDIT")
print("="*70)

print(report)

print("\nRecommendation Distribution")

print(distribution)

print("\nOverall Fairness")

print(fairness)

print("\nSaved To")

print(output)