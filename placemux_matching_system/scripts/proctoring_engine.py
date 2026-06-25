import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students = pd.read_csv(
    os.path.join(BASE_DIR, "data", "students.csv")
)

results = []

for _, student in students.iterrows():

    # Calculate confidence score from student profile
    confidence = min(
        100,
        int(
            (
                student["python"] +
                student["ml"] +
                student["communication"]
            ) / 3
        )
    )

    # Simple rule-based violation detection
    violations = 0

    if student["cgpa"] < 7:
        violations = 2
    elif student["projects"] < 2:
        violations = 1

    # Verification decision
    if confidence >= 90 and violations == 0:
        status = "Verified"
    elif confidence >= 80:
        status = "Review Required"
    else:
        status = "Rejected"

    results.append(
        {
            "student_id": student["student_id"],
            "name": student["name"],
            "confidence_score": confidence,
            "violations": violations,
            "status": status
        }
    )

report = pd.DataFrame(results)

output_path = os.path.join(
    BASE_DIR,
    "outputs",
    "proctoring_report.csv"
)

report.to_csv(
    output_path,
    index=False
)

print("=" * 60)
print("PLACEMUX PROCTORING ENGINE")
print("=" * 60)

print(report)

verified = len(report[report["status"] == "Verified"])
review = len(report[report["status"] == "Review Required"])
rejected = len(report[report["status"] == "Rejected"])

print("\nSummary")
print("-" * 60)
print(f"Verified Candidates : {verified}")
print(f"Review Required     : {review}")
print(f"Rejected Candidates : {rejected}")

verification_rate = round((verified / len(report)) * 100, 2)

print(f"\nVerification Rate : {verification_rate}%")

print("\nReport Saved To:")
print(output_path)