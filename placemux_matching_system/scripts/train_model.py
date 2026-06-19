import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

students = pd.read_csv(os.path.join(BASE_DIR, "data", "students.csv"))
jobs = pd.read_csv(os.path.join(BASE_DIR, "data", "jobs.csv"))

training_data = []

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

        training_data.append([
            python_gap,
            ml_gap,
            sql_gap,
            comm_gap,
            project_gap,
            exp_gap,
            match
        ])

df = pd.DataFrame(
    training_data,
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(
    model,
    os.path.join(BASE_DIR, "outputs", "matching_model.pkl")
)

print("=" * 50)
print("MODEL TRAINED SUCCESSFULLY")
print("=" * 50)
print("Model saved in outputs folder")