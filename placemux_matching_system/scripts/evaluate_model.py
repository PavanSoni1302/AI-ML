import pandas as pd
import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students = pd.read_csv(os.path.join(BASE_DIR, "data", "students.csv"))
jobs = pd.read_csv(os.path.join(BASE_DIR, "data", "jobs.csv"))

data = []

for _, s in students.iterrows():
    for _, j in jobs.iterrows():

        python_gap = s["python"] - j["python_req"]
        ml_gap = s["ml"] - j["ml_req"]
        sql_gap = s["sql"] - j["sql_req"]
        comm_gap = s["communication"] - j["comm_req"]
        project_gap = s["projects"] - j["projects_req"]
        exp_gap = s["experience"] - j["exp_req"]

        match = 1 if (
            python_gap >= 0 and
            ml_gap >= 0 and
            sql_gap >= 0
        ) else 0

        data.append([
            python_gap,
            ml_gap,
            sql_gap,
            comm_gap,
            project_gap,
            exp_gap,
            match
        ])

df = pd.DataFrame(
    data,
    columns=[
        "python_gap",
        "ml_gap",
        "sql_gap",
        "comm_gap",
        "project_gap",
        "exp_gap",
        "match"
    ]
)

X = df.drop("match", axis=1)
y = df["match"]

model = joblib.load(
    os.path.join(BASE_DIR, "outputs", "matching_model.pkl")
)

pred = model.predict(X)

accuracy = accuracy_score(y, pred)
precision = precision_score(y, pred)
recall = recall_score(y, pred)

cm = confusion_matrix(y, pred)

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print()
print("Confusion Matrix")
print(cm)