import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

recommendations = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "validated_recommendations.csv"
    )
)

metrics = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "validation_metrics.csv"
    )
)

print("=" * 60)
print("PORTAL VALIDATION DEMO")
print("=" * 60)
print("\nValidation Metrics")
print(metrics)
print("\nTop Valid Recommendations")
valid = recommendations[
    recommendations["validation_status"] == "VALID"
]
print(valid.head(10))