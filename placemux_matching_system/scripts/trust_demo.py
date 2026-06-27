import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPORT = os.path.join(
    BASE_DIR,
    "outputs",
    "trust_signoff.csv"
)

def show_demo():
    df = pd.read_csv(REPORT)
    approved = len(df[df["trust_status"] == "APPROVED"])
    review = len(df[df["trust_status"] == "REVIEW"])
    rejected = len(df[df["trust_status"] == "REJECTED"])
    print("=" * 60)
    print("PLACEMUX AI TRUST DEMO")
    print("=" * 60)
    print(f"Approved Candidates : {approved}")
    print(f"Review Candidates   : {review}")
    print(f"Rejected Candidates : {rejected}")
    print("\nDetailed Report")
    print("-" * 60)
    print(df)
    print("\nTrust Score Statistics")
    print("-" * 60)
    print(df["trust_score"].describe())
if __name__ == "__main__":
    show_demo()