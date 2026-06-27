import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "recommendation_explanations.csv"
)

OUTPUT = os.path.join(
    BASE_DIR,
    "outputs",
    "review_queue.csv"
)

def build_review_queue():

    df = pd.read_csv(INPUT)
    queue = df[
        (df["recommendation_score"] >= 70) &
        (df["recommendation_score"] < 85)
    ].copy()
    queue["review_status"] = "Pending"
    queue.to_csv(
        OUTPUT,
        index=False
    )
    
    return queue

if __name__ == "__main__":
    queue = build_review_queue()
    print("=" * 60)
    print("ADMIN REVIEW QUEUE")
    print("=" * 60)
    print(queue)
    print("\nSaved To")
    print(OUTPUT)