import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "parsed_resume.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "ontology_mapping.csv"
)

ONTOLOGY = {
    "python": "Programming",
    "sql": "Database",
    "flask": "Backend",
    "git": "Version Control",
    "machine learning": "Artificial Intelligence",
    "scikit-learn": "Artificial Intelligence",
    "numpy": "Data Science",
    "pandas": "Data Science",
    "communication": "Soft Skills",
    "problem solving": "Soft Skills"
}

def map_skills():
    df = pd.read_csv(INPUT_FILE)
    categories = []
    for _, row in df.iterrows():
        skill = str(row["skill"]).lower()
        category = ONTOLOGY.get(
            skill,
            "Other"
        )
        categories.append(category)
    df["category"] = categories
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )
    return df

if __name__ == "__main__":
    mapped = map_skills()
    print("=" * 60)
    print("ONTOLOGY ENGINE")
    print("=" * 60)
    print(mapped)
    print("\nSaved To")
    print(OUTPUT_FILE)