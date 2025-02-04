import pandas as pd
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Process log loss data from JSON files.")
parser.add_argument(
    "directory",
    type=str,
    help="Path to the directory containing JSON files starting with 'q80_'.",
)
args = parser.parse_args()

datasets = {}

for file_name in os.listdir(args.directory):
    if file_name.startswith("q80_") and file_name.endswith(".json"):
        key = file_name.split("_", 1)[1].rsplit(".", 1)[0].replace("_", " ")
        file_path = os.path.join(args.directory, file_name)
        with open(file_path, "r") as file:
            data = json.load(file)
            if "log_losses" in data:
                datasets[key] = data["log_losses"]

oversamplers = [
    "ADASYN",
    "BorderlineSMOTE",
    "SMOTE",
    "SVMSMOTE",
]

generators = [
    "TSMixup",
    "KernelSynth",
    "DBA",
    "Scaling",
    "MagnitudeWarping",
    "TimeWarping",
    "Jittering",
]

undersamplers = ["RandomUnderSampler"]

data = {
    dataset: {method: values for method, values in methods.items()}
    for dataset, methods in datasets.items()
}

df = pd.DataFrame(data)

print(df, "\n")

ranked_df = df.rank(axis=0, method="min", ascending=True)

print(ranked_df)
