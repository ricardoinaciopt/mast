import json
import sys
import os
from collections import defaultdict
from tabulate import tabulate
import matplotlib.pyplot as plt


def calculate_average_ranks(results):
    method_ranks = defaultdict(list)

    for _, methods in results["by_id"].items():
        scores = [(method, data["log_loss"]) for method, data in methods.items()]
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


def generate_plot(average_ranks, input_file, parent_folder):
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
        fontsize=12,
    )

    plt.tight_layout()

    output_dir = "figs_uid"
    os.makedirs(output_dir, exist_ok=True)

    base_file_name = os.path.splitext(input_file)[0]
    output_file = os.path.join(
        output_dir, f"{parent_folder}_{base_file_name}_average_rank_plot.pdf"
    )

    plt.savefig(output_file, dpi=1000, format="pdf", bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_json_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]

    try:
        with open(json_file_path, "r") as file:
            results = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{json_file_path}' is not a valid JSON file.")
        sys.exit(1)

    average_ranks = calculate_average_ranks(results)
    sorted_ranks = sorted(average_ranks.items(), key=lambda x: x[1])

    table = [(method, f"{rank:.2f}") for method, rank in sorted_ranks]
    print(tabulate(table, headers=["Method", "Average Rank"], tablefmt="grid"))

    parent_folder = os.path.basename(os.path.dirname(json_file_path))
    generate_plot(average_ranks, os.path.basename(json_file_path), parent_folder)


if __name__ == "__main__":
    main()
