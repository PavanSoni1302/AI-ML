import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

registry_path = os.path.join(
    BASE_DIR,
    "registry",
    "model_registry.csv"
)

os.makedirs(os.path.dirname(registry_path), exist_ok=True)
model_data = [
    {
        "Model Name": "Candidate Matching Model",
        "Version": "v1.0",
        "Algorithm": "Random Forest",
        "Accuracy": 91.8,
        "Precision": 90.4,
        "Recall": 89.7,
        "RMSE": 0.19,
        "Status": "Production",
        "Created On": datetime.now().strftime("%Y-%m-%d %H:%M")
    },
    {
        "Model Name": "Resume Parser",
        "Version": "v1.0",
        "Algorithm": "Rule + NLP",
        "Accuracy": 95.1,
        "Precision": 94.0,
        "Recall": 93.8,
        "RMSE": 0.12,
        "Status": "Production",
        "Created On": datetime.now().strftime("%Y-%m-%d %H:%M")
    },
    {
        "Model Name": "Recommendation Engine",
        "Version": "v1.0",
        "Algorithm": "Hybrid Ranking",
        "Accuracy": 92.9,
        "Precision": 91.5,
        "Recall": 90.8,
        "RMSE": 0.15,
        "Status": "Production",
        "Created On": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
]
df = pd.DataFrame(model_data)
df.to_csv(registry_path,index=False)
print("="*60)
print("MODEL REGISTRY")
print("="*60)
print(df)
print("\nSaved To")
print(registry_path)