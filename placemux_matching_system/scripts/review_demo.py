import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

queue = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "review_queue.csv"
    )
)

print("=" * 60)
print("ADMIN REVIEW DASHBOARD")
print("=" * 60)

print(queue)

print("\nTotal Pending Reviews")

print(len(queue))