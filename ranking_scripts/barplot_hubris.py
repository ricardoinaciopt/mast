import os
import json
import argparse
import matplotlib.pyplot as plt


def generate_plot(average_ranks, output_file):
    sorted_ranks = sorted(average_ranks.items(), key=lambda x: x[1])
    methods, ranks = zip(*sorted_ranks)

    categories = {
        "Oversampling Features": ["ADASYN", "SMOTE", "BorderlineSMOTE", "SVMSMOTE"],
        "Undersampling Features": ["RandomUnderSampler"],
        "Generating Time Series": [
            "DBA",
            "Jittering",
            "Scaling",
            "MagnitudeWarping",
            "TimeWarping",
            "TSMixup",
        ],
        "No Resampling": ["No Resampling"],
    }

    colors = []
    for method in methods:
        if method in categories["Oversampling Features"]:
            colors.append("green")
        elif method in categories["Undersampling Features"]:
            colors.append("red")
        elif method in categories["Generating Time Series"]:
            colors.append("orange")
        elif method in categories["No Resampling"]:
            colors.append("gray")
        else:
            colors.append("black")

    plt.figure(figsize=(12, 8))
    plt.bar(methods, ranks, color=colors, edgecolor="black")
    plt.xlabel("Method", fontsize=14)
    plt.ylabel("Average Rank", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color="green", lw=4, label="Oversampling Features"),
            plt.Line2D([0], [0], color="red", lw=4, label="Undersampling Features"),
            plt.Line2D([0], [0], color="orange", lw=4, label="Generating Time Series"),
            plt.Line2D([0], [0], color="gray", lw=4, label="No Resampling"),
        ],
        title="Resampling Method",
        title_fontsize=12,
        fontsize=13,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=4,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=1000, format="pdf", bbox_inches="tight")
    plt.close()


def generate_plot_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if "lc_60_40" not in data or not isinstance(data["lc_60_40"], dict):
        print("Error: JSON file must contain a 'lc_60_40' key with a dictionary.")
        return

    data = data["lc_60_40"]

    filename = os.path.basename(json_path)
    task_name = "lc"

    output_folder = os.path.join("figs_ranks", task_name)
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, f"{task_name}.pdf")
    generate_plot(data, output_file)
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bar plot from a JSON file.")
    parser.add_argument("json_path", type=str, help="Path to the input JSON file.")
    args = parser.parse_args()

    generate_plot_from_json(args.json_path)
