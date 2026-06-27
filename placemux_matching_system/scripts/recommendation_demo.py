import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPORT = os.path.join(
    BASE_DIR,
    "outputs",
    "recommendations.csv"
)
df = pd.read_csv(REPORT)
print("=" * 60)
print("PLACEMUX RECOMMENDATION DEMO")
print("=" * 60)
print("\nTop 10 Recommendations")
print(df.head(10))
print("\nAverage Recommendation Score")
print(round(df["recommendation_score"].mean(), 2))
print("\nHighest Recommendation Score")
print(df["recommendation_score"].max())
print("\nLowest Recommendation Score")
print(df["recommendation_score"].min())