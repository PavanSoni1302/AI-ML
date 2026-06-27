import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ontology = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "ontology_mapping.csv"
    )
)
print("=" * 60)
print("ONTOLOGY DEMO")
print("=" * 60)
print("\nSkill Categories")
print(
    ontology[
        [
            "skill",
            "category"
        ]
    ]
)
print("\nCategory Summary")
print(
    ontology["category"].value_counts()
)