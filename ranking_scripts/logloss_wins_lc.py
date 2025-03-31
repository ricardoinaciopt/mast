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

    baseline_wins = wins_data[baseline_method]
    methods = [m for m in wins_data if m != baseline_method]

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        method_wins = wins_data[method]
        pair_total = method_wins + baseline_wins
        if pair_total == 0:
            method_pct = 0
            baseline_pct = 0
        else:
            method_pct = method_wins / pair_total * 100
            baseline_pct = baseline_wins / pair_total * 100

        ax.bar(method, method_pct, color="green", edgecolor="black")
        ax.bar(method, baseline_pct, bottom=method_pct, color="red", edgecolor="black")

    ax.set_ylabel("Win / Loss Percentage Ratio")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 110, 10))
    ax.set_yticklabels([f"{t}%" for t in range(0, 110, 10)])

    ax.axhline(50, color="black", linestyle="--", linewidth=1)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="green", label="Resampling Method"),
        plt.Rectangle((0, 0), 1, 1, color="red", label="Baseline (No Resampling)"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2
    )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout(rect=[0, 0, 1, 1])
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
