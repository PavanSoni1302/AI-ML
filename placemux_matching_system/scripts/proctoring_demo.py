import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

report = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "proctoring_report.csv"
    )
)

verified = report[
    report["status"] == "Verified"
]

review = report[
    report["status"] == "Review Required"
]

rejected = report[
    report["status"] == "Rejected"
]

print("=" * 60)
print("PROCTORING DEMO")
print("=" * 60)

print("\nVerified Candidates")

print(len(verified))

print("\nNeed Review")

print(len(review))

print("\nRejected")

print(len(rejected))

print("\nVerification Rate")

rate = len(verified) / len(report) * 100

print(round(rate,2),"%")