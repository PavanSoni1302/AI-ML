import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

feature_path = os.path.join(
    BASE_DIR,
    "feature_store",
    "feature_store.csv"
)

os.makedirs(os.path.dirname(feature_path), exist_ok=True)
feature_data = [
    {
        "Candidate":"John",
        "Skill":"Python",
        "Score":95,
        "Verified":"Yes"
    },
    {
        "Candidate":"John",
        "Skill":"Machine Learning",
        "Score":91,
        "Verified":"Yes"
    },
    {
        "Candidate":"John",
        "Skill":"SQL",
        "Score":88,
        "Verified":"Yes"
    },
    {
        "Candidate":"Priya",
        "Skill":"Java",
        "Score":93,
        "Verified":"Yes"
    },
    {
        "Candidate":"Priya",
        "Skill":"Spring Boot",
        "Score":89,
        "Verified":"Yes"
    },
    {
        "Candidate":"Rahul",
        "Skill":"Cyber Security",
        "Score":90,
        "Verified":"Yes"
    },
    {
        "Candidate":"Rahul",
        "Skill":"Blockchain",
        "Score":86,
        "Verified":"Yes"
    }
]
df = pd.DataFrame(feature_data)
df.to_csv(feature_path,index=False)
print("="*60)
print("FEATURE STORE")
print("="*60)
print(df)
print("\nSaved To")
print(feature_path)