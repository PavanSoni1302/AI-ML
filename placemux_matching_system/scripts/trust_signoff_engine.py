import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCTORING = os.path.join(
    BASE_DIR,
    "outputs",
    "fp_reduction_report.csv"
)

ONTOLOGY = os.path.join(
    BASE_DIR,
    "outputs",
    "ontology_mapping.csv"
)

OUTPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "trust_signoff.csv"
)


def generate_trust():

    proctor = pd.read_csv(PROCTORING)
    ontology = pd.read_csv(ONTOLOGY)

    ontology_ready = len(ontology) > 0

    trust = []

    for _, row in proctor.iterrows():

        score = row["confidence_score"]
        violations = row["violations"]
        status = row["improved_status"]

        trust_score = score

        if violations > 0:
            trust_score -= violations * 10

        if status == "Verified" and ontology_ready:

            trust_status = "APPROVED"
            reason = "Verified candidate with parsed ontology."

        elif status == "Manual Review":

            trust_status = "REVIEW"
            reason = "Requires manual verification."

        else:

            trust_status = "REJECTED"
            reason = "Failed trust validation."

        trust.append(
            {
                "student_id": row["student_id"],
                "name": row["name"],
                "trust_score": round(trust_score, 2),
                "trust_status": trust_status,
                "reason": reason,
            }
        )

    trust_df = pd.DataFrame(trust)

    trust_df.to_csv(
        OUTPUT,
        index=False
    )

    return trust_df


if __name__ == "__main__":

    df = generate_trust()

    print("=" * 60)
    print("PLACEMUX AI TRUST SIGN-OFF")
    print("=" * 60)

    print(df)

    print("\nSummary")
    print("-" * 60)

    print("Approved :", len(df[df["trust_status"] == "APPROVED"]))
    print("Review   :", len(df[df["trust_status"] == "REVIEW"]))
    print("Rejected :", len(df[df["trust_status"] == "REJECTED"]))

    print("\nTrust report saved to:")
    print(OUTPUT)