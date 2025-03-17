import pandas as pd
import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

parser = argparse.ArgumentParser(description="Process log loss data from JSON files.")
parser.add_argument(
    "directory",
    type=str,
    help="Path to the directory containing JSON files.",
)
parser.add_argument(
    "prefix",
    type=str,
    help="Prefix of the JSON files to process (e.g., 'q80_').",
)
args = parser.parse_args()

datasets = {}

for file_name in os.listdir(args.directory):
    if file_name.startswith(args.prefix) and file_name.endswith(".json"):
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

undersamplers = ["RandomUnderSampler"]

generators = [
    "TSMixup",
    "DBA",
    "Scaling",
    "Jittering",
    "TimeWarping",
    "MagnitudeWarping",
]


def get_color(model_name):
    if model_name in oversamplers:
        return "forestgreen"
    elif model_name in undersamplers:
        return "firebrick"
    elif model_name in generators:
        return "darkorange"
    else:
        return "gray"


legend_elements = [
    Patch(
        facecolor="forestgreen", edgecolor="forestgreen", label="Oversampling Features"
    ),
    Patch(facecolor="firebrick", edgecolor="firebrick", label="Undersampling Features"),
    Patch(
        facecolor="darkorange", edgecolor="darkorange", label="Generating Time Series"
    ),
    Patch(facecolor="gray", edgecolor="gray", label="No Resampling"),
]

rankings = {}

for dataset, methods in datasets.items():
    df = pd.DataFrame(list(methods.items()), columns=["Method", "Log Loss"]).set_index(
        "Method"
    )
    df["Log Loss Rank"] = df["Log Loss"].rank()
    rankings[dataset] = df["Log Loss Rank"].astype("int")

overall_ranking = pd.DataFrame(rankings).mean(axis=1).sort_values()
overall_ranking = overall_ranking.reset_index()
overall_ranking.columns = ["Method", "Average Rank"]
overall_ranking["Average Rank"] = overall_ranking["Average Rank"].round(2)
overall_ranking["Overall Rank"] = overall_ranking["Average Rank"].rank().astype(int)
overall_ranking.set_index("Method", inplace=True)

overall_ranking["Color"] = overall_ranking.index.map(get_color)

plt.rcParams.update({"font.size": 13})
plt.figure(figsize=(10, 8))
bars = plt.bar(
    overall_ranking.index,
    overall_ranking["Average Rank"],
    color=overall_ranking["Color"],
    edgecolor="black",
)

plt.xlabel("Method", fontsize=12)
plt.ylabel("Average Rank", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=14)

plt.legend(
    handles=legend_elements,
    title="Method Type",
    loc="upper center",  # Change legend location to top
    bbox_to_anchor=(0.5, 1.3),  # Adjust the position to be above the plot
    ncol=2,  # Arrange legend items in 3 columns
)

plt.tight_layout()
os.makedirs("figs", exist_ok=True)
# windows
# path = f"figs/{args.directory.split('\\', 1)[1].rsplit('.', 1)[0].replace('\\', '_')}_average_rank_plot.pdf"
# mac os
path = f"figs/{args.directory.split('/', 1)[1].rsplit('.', 1)[0].replace('/', '_')}_average_rank_plot.pdf"
plt.savefig(path, format="pdf", dpi=1000)
plt.show()
