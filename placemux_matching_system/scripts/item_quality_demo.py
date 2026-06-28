import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

weak = pd.read_csv(
    os.path.join(BASE_DIR,"outputs","weak_items.csv")
)
dashboard = pd.read_csv(
    os.path.join(BASE_DIR,"outputs","recruiter_dashboard.csv")
)
print("="*60)
print("ITEM QUALITY DEMO")
print("="*60)
print("\nRecruiter Dashboard")
print(dashboard)
print("\nWeak Job Listings")
print(weak)
print("\nTotal Weak Jobs :", len(weak))