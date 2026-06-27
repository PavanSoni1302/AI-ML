import os
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "proctoring_report.csv"
) 
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "fp_reduction_report.csv"
)

def reduce_false_positives():
    report = pd.read_csv(INPUT_FILE) 
    baseline_fp = len(
        report[
            report["status"] == "Review Required"
        ]
    ) 
    improved_status = []
    reasons = []
    for _, row in report.iterrows(): 
        score = row["confidence_score"]
        violations = row["violations"] 
        if score >= 90 and violations == 0:
            status = "Verified"
            reason = "High confidence with no violations."
        elif score >= 80 and violations == 0: 
            status = "Verified"
            reason = "Confidence threshold satisfied."
        elif score >= 75 and violations <= 1:
            status = "Manual Review"
            reason = "Borderline candidate requires manual review."
        else:
            status = "Rejected"
            reason = "Low confidence or excessive violations."
        improved_status.append(status)
        reasons.append(reason)
    report["improved_status"] = improved_status
    report["decision_reason"] = reasons
    final_fp = len(
        report[
            report["improved_status"] == "Manual Review"
        ]
    )
    reduction = baseline_fp - final_fp
    reduction_percent = round(
        (reduction / baseline_fp) * 100,
        2
    ) if baseline_fp != 0 else 100
    report.to_csv(
        OUTPUT_FILE,
        index=False
    )
    metrics = {
        "Baseline False Positives": baseline_fp,
        "Reduced False Positives": final_fp,
        "Reduction": reduction,
        "Reduction (%)": reduction_percent
    }
    return report, metrics

if __name__ == "__main__":
    report, metrics = reduce_false_positives()
    print("=" * 60)
    print("PLACEMUX FALSE POSITIVE REDUCTION")
    print("=" * 60)
    print(report)
    print("\nPerformance Metrics")
    print("-" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("\nReport Saved")
    print(OUTPUT_FILE)