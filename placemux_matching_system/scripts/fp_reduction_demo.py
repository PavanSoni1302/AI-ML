import os
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
report = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "fp_reduction_report.csv"
    )
)
verified = len(
    report[
        report["improved_status"] == "Verified"
    ]
)
review = len(
    report[
        report["improved_status"] == "Manual Review"
    ]
)
rejected = len(
    report[
        report["improved_status"] == "Rejected"
    ]
)
print("=" * 60)
print("FALSE POSITIVE REDUCTION DEMO")
print("=" * 60)
print(f"Verified Candidates : {verified}")
print(f"Manual Review      : {review}")
print(f"Rejected           : {rejected}")
print("\nDecision Summary")
print(
    report[
        [
            "student_id",
            "name",
            "confidence_score",
            "violations",
            "improved_status"
        ]
    ]
)