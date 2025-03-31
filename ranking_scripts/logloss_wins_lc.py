import os
import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt


def process_lc_folder(lc_path):
    method_wins = defaultdict(int)
    for file_name in os.listdir(lc_path):
        if not file_name.endswith(".json"):
            continue
        print(f"\tProcessing {file_name}...")
        file_path = os.path.join(lc_path, file_name)
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                by_id = data.get("by_id", {})
                print("\t\tProcessing", len(by_id), "time series...")
                for uid_losses in by_id.values():
                    losses = {
                        method: info["log_loss"] for method, info in uid_losses.items()
                    }
                    best_loss = min(losses.values())
                    for method, loss in losses.items():
                        if loss == best_loss:
                            method_wins[method] += 1
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {file_path}: {e}")
            continue
    return dict(method_wins)


def plot_method_vs_baseline(wins_data, baseline_method, output_path):
    if baseline_method not in wins_data:
        print(f"Baseline method '{baseline_method}' not found in wins data.")
        return

    total_wins = sum(wins_data.values())
    if total_wins == 0:
        print("No wins to plot.")
        return

    methods = [m for m in wins_data if m != baseline_method]

    method_percentages = [wins_data[m] / total_wins * 100 for m in methods]
    baseline_percentage = wins_data[baseline_method] / total_wins * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        method_pct = method_percentages[i]
        lower = min(method_pct, baseline_percentage)
        upper = max(method_pct, baseline_percentage) - lower
        colors = [
            "green" if method_pct < baseline_percentage else "red",
            "red" if method_pct < baseline_percentage else "green",
        ]
        ax.bar(method, lower, color=colors[0], edgecolor="black")
        ax.bar(method, upper, bottom=lower, color=colors[1], edgecolor="black")

    ax.set_ylabel("Percentage of Total Wins")
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{int(tick)}%" for tick in ax.get_yticks()])
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="green", label="Resampling Methods"),
        plt.Rectangle((0, 0), 1, 1, color="red", label="No Resampling"),
    ]
    ax.legend(handles=legend_handles)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute wins counts and generate wins comparison plot"
    )
    parser.add_argument("root_dir", help="Directory containing LC folders")
    parser.add_argument("lc_folder", help="Name of the LC folder (e.g., lc_60_40)")
    args = parser.parse_args()

    lc_path = os.path.join(args.root_dir, args.lc_folder)
    if not os.path.isdir(lc_path):
        print(f"LC folder '{args.lc_folder}' not found in root directory.")
        exit(1)

    print(f"Processing LC folder: {args.lc_folder}...")
    wins_data = process_lc_folder(lc_path)

    wins_counts_dir = os.path.join(args.root_dir, "wins_counts")
    os.makedirs(wins_counts_dir, exist_ok=True)

    wins_count_filename = f"wins_count_{args.lc_folder}.json"
    wins_count_path = os.path.join(wins_counts_dir, wins_count_filename)
    with open(wins_count_path, "w") as f:
        json.dump({args.lc_folder: wins_data}, f, indent=2)
    print(f"Wins count JSON saved at: {wins_count_path}")

    baseline_method = "No Resampling"
    plot_output_path = os.path.join(
        wins_counts_dir, f"stacked_vs_baseline_{args.lc_folder}.pdf"
    )
    plot_method_vs_baseline(wins_data, baseline_method, plot_output_path)
    print(f"Stacked bar plot against baseline saved at: {plot_output_path}")
