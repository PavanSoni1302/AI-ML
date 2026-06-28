import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "recommendations.csv"
)

VALIDATED = os.path.join(
    BASE_DIR,
    "outputs",
    "validated_recommendations.csv"
)

METRICS = os.path.join(
    BASE_DIR,
    "outputs",
    "validation_metrics.csv"
)

def validate():
    df = pd.read_csv(INPUT)
    status = []
    remarks = []
    for _, row in df.iterrows():
        score = row["recommendation_score"]
        trust = row["trust_status"]
        if trust == "APPROVED" and score >= 85:
            status.append("VALID")
            remarks.append("High confidence recommendation")
        elif trust == "APPROVED" and score >= 70:
            status.append("REVIEW")
            remarks.append("Requires recruiter confirmation")
        else:
            status.append("REJECTED")
            remarks.append("Recommendation rejected")
    df["validation_status"] = status
    df["remarks"] = remarks
    df.to_csv(
        VALIDATED,
        index=False
    )
    metrics = pd.DataFrame([{
        "Total Recommendations": len(df),
        "Validated":
        len(df[df["validation_status"] == "VALID"]),
        "Review":
        len(df[df["validation_status"] == "REVIEW"]),
        "Rejected":
        len(df[df["validation_status"] == "REJECTED"])
    }])
    metrics.to_csv(
        METRICS,
        index=False
    )
    return df, metrics

if __name__ == "__main__":
    recommendations, metrics = validate()
    print("=" * 60)
    print("RECOMMENDATION VALIDATION")
    print("=" * 60)
    print(metrics)
    print("\nValidated recommendations saved.")