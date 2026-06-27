import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RECOMMENDATIONS = os.path.join(
    BASE_DIR,
    "outputs",
    "recommendations.csv"
)

def get_student_recommendations(student_id):
    df = pd.read_csv(RECOMMENDATIONS)
    result = df[
        df["student_id"] == student_id
    ]
    return result.sort_values(
        by="recommendation_score",
        ascending=False
    )

if __name__ == "__main__":
    student = int(input("Enter Student ID : "))
    recommendations = get_student_recommendations(student)
    print(recommendations)