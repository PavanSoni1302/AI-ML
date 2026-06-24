from flask import Flask, jsonify
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

students = pd.read_csv(
    os.path.join(BASE_DIR, "data", "students.csv")
)

jobs = pd.read_csv(
    os.path.join(BASE_DIR, "data", "jobs.csv")
)


@app.route("/")
def home():

    return jsonify(
        {
            "project": "PlaceMux Matching System",
            "status": "running",
            "students": len(students),
            "jobs": len(jobs)
        }
    )


@app.route("/students")
def get_students():

    return jsonify(
        students.to_dict(
            orient="records"
        )
    )


@app.route("/jobs")
def get_jobs():

    return jsonify(
        jobs.to_dict(
            orient="records"
        )
    )


@app.route("/health")
def health():

    return jsonify(
        {
            "status": "healthy"
        }
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )