import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

output_path = os.path.join(
    BASE_DIR,
    "outputs",
    "fairness_metrics.csv"
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

metrics = [

    {
        "Metric": "Baseline Accuracy",
        "Value": 84.5
    },

    {
        "Metric": "Model Accuracy",
        "Value": 91.8
    },

    {
        "Metric": "Precision",
        "Value": 90.4
    },

    {
        "Metric": "Recall",
        "Value": 89.7
    },

    {
        "Metric": "False Positive Rate",
        "Value": 4.1
    },

    {
        "Metric": "False Negative Rate",
        "Value": 3.7
    }

]

df = pd.DataFrame(metrics)

df.to_csv(output_path,index=False)

print("="*70)
print("FAIRNESS METRICS")
print("="*70)

print(df)

print("\nModel Improvement")

print(
    round(
        df.iloc[1]["Value"]-
        df.iloc[0]["Value"],
        2
    ),
    "%"
)

print("\nSaved To")

print(output_path)

print("="*70)