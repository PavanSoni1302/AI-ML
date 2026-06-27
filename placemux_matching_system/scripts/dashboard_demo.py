import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dashboard = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "placement_dashboard.csv"
    )
)
recommendations = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "recommendations.csv"
    )
)
print("=" * 60)
print("PLACEMUX PLACEMENT DASHBOARD")
print("=" * 60)
print(dashboard)
print("\nTop 10 Recommendations")
print(
    recommendations.sort_values(
        by="recommendation_score",
        ascending=False
    ).head(10)
)