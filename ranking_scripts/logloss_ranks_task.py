import json
import sys
import os
from collections import defaultdict
from tabulate import tabulate
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
    plt.bar(methods, ranks, color=colors, edgecolor=("black"))
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
        title="Method",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=1000, format="pdf", bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.close()


def calculate_ranks_across_files(all_log_losses):
    method_ranks = defaultdict(list)

    for file_log_losses in all_log_losses:
        scores = [(method, log_loss) for method, log_loss in file_log_losses.items()]
        sorted_scores = sorted(scores, key=lambda x: x[1])
        ranks = {}
        current_rank = 1
        while sorted_scores:
            min_score = sorted_scores[0][1]
            same_score = [m for m, s in sorted_scores if s == min_score]
            avg_rank = current_rank + (len(same_score) - 1) / 2
            for method in same_score:
                ranks[method] = avg_rank
            sorted_scores = [item for item in sorted_scores if item[1] > min_score]
            current_rank += len(same_score)

        for method, rank in ranks.items():
            method_ranks[method].append(rank)

    avg_ranks = {
        method: sum(ranks) / len(ranks) for method, ranks in method_ranks.items()
    }

    return avg_ranks


def process_folder(folder_path):
    all_log_losses = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "r") as file:
                    results = json.load(file)
                    if "log_losses" in results:
                        all_log_losses.append(results["log_losses"])
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Error processing file: {file_path}")
                continue

    overall_average_ranks = calculate_ranks_across_files(all_log_losses)

    sorted_ranks = sorted(overall_average_ranks.items(), key=lambda x: x[1])
    table = [(method, f"{rank:.2f}") for method, rank in sorted_ranks]
    print(tabulate(table, headers=["Method", "Overall Average Rank"], tablefmt="grid"))

    output_dir = "figs_task"
    os.makedirs(output_dir, exist_ok=True)

    grandparent_folder = os.path.basename(
        os.path.dirname(os.path.normpath(folder_path))
    )
    parent_folder = os.path.basename(os.path.normpath(folder_path))
    output_file = os.path.join(
        output_dir,
        f"{grandparent_folder}_{parent_folder}_overall_average_rank_plot.pdf",
    )

    generate_plot(overall_average_ranks, output_file)


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        sys.exit(1)

    process_folder(folder_path)


if __name__ == "__main__":
    main()
