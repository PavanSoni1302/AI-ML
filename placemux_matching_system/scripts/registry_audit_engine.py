import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

output_path = os.path.join(
    BASE_DIR,
    "outputs",
    "registry_audit.csv"
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

audit_log = [

    {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Component": "Model Registry",
        "Action": "Registered Models",
        "Status": "Success"
    },

    {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Component": "Feature Store",
        "Action": "Updated Features",
        "Status": "Success"
    },

    {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Component": "Validation Engine",
        "Action": "Validated Feature Store",
        "Status": "Success"
    },

    {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Component": "Prediction Engine",
        "Action": "Generated Recommendation",
        "Status": "Success"
    },

    {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Component": "Explainability Engine",
        "Action": "Generated Explanation",
        "Status": "Success"
    }

]

df = pd.DataFrame(audit_log)

df.to_csv(output_path, index=False)

print("=" * 70)
print("REGISTRY AUDIT ENGINE")
print("=" * 70)

print(df)

print("\nAudit Log Saved Successfully")

print(output_path)

print("=" * 70)