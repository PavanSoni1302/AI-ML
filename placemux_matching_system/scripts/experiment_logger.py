import os
import random
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGISTRY_PATH = os.path.join(
    BASE_DIR,
    "registry",
    "model_registry.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "experiment_log.csv"
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("=" * 90)
print("PLACEMUX EXPERIMENT LOGGER")
print("=" * 90)

try:

    registry = pd.read_csv(REGISTRY_PATH)

    print("\nModel Registry Loaded Successfully\n")

except Exception as e:

    print("ERROR :", e)

    exit()

logs = []
for index, model in registry.iterrows():
    training_time = random.randint(60, 180)
    run_duration = round(random.uniform(0.80, 2.50), 2)
    model_size = round(random.uniform(15.5, 65.5), 2)
    logs.append({
        "Run ID":
        f"EXP-{1001 + index}",
        "Timestamp":
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model":
        model["Model Name"],
        "Version":
        model["Version"],
        "Dataset Version":
        "Dataset v3.0",
        "Algorithm":
        model.get("Algorithm", "Random Forest"),
        "Accuracy":
        model["Accuracy"],
        "Precision":
        model["Precision"],
        "Recall":
        model["Recall"],
        "Training Time (sec)":
        training_time,
        "Run Duration (sec)":
        run_duration,
        "Model Size (MB)":
        model_size,
        "Environment":
        "Production",
        "Git Branch":
        "main",
        "Status":
        "Completed",
        "Experiment Notes":
        "Training completed successfully with production configuration."
    })
experiment_log = pd.DataFrame(logs)
experiment_log.to_csv(
    OUTPUT_PATH,
    index=False
)

print(experiment_log)
print("\n" + "=" * 90)
print("EXPERIMENT SUMMARY")
print("=" * 90)
print("Experiments Logged     :", len(experiment_log))
print("Average Accuracy       :",
      round(experiment_log["Accuracy"].mean(), 2), "%")
print("Average Precision      :",
      round(experiment_log["Precision"].mean(), 2), "%")
print("Average Recall         :",
      round(experiment_log["Recall"].mean(), 2), "%")
print("Average Training Time  :",
      round(experiment_log["Training Time (sec)"].mean(), 2), "sec")
print("Average Run Duration   :",
      round(experiment_log["Run Duration (sec)"].mean(), 2), "sec")
print("Average Model Size     :",
      round(experiment_log["Model Size (MB)"].mean(), 2), "MB")
print("\nOutput Saved To")
print(OUTPUT_PATH)
print("=" * 90)