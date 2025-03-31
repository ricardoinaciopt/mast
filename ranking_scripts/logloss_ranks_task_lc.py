import os
import json
import argparse
from collections import defaultdict
import statistics
from scipy.stats import rankdata


def calculate_ranks(log_loss_dict):
    methods = list(log_loss_dict.keys())
    log_losses = [log_loss_dict[m]["log_loss"] for m in methods]
    ranks_array = rankdata(log_losses, method="average")
    ranks = {methods[i]: ranks_array[i] for i in range(len(methods))}
    return ranks


def process_lc_folders(root_dir):
    results = {}

    for lc_folder in os.listdir(root_dir):
        lc_path = os.path.join(root_dir, lc_folder)

        if not os.path.isdir(lc_path) or not lc_folder.startswith("lc"):
            continue
        print(f"Processing {lc_folder}...")

        method_ranks = defaultdict(list)

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
                        ranks = calculate_ranks(uid_losses)
                        for method, rank in ranks.items():
                            method_ranks[method].append(rank)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing {file_path}: {e}")
                continue

        avg_ranks = {
            method: statistics.mean(ranks) for method, ranks in method_ranks.items()
        }

        results[lc_folder] = avg_ranks

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate method performance ranks across LC folders"
    )
    parser.add_argument("root_dir", help="Directory containing LC folders")

    args = parser.parse_args()

    results = process_lc_folders(args.root_dir)

    sorted_results = {
        key: dict(sorted(subdict.items(), key=lambda item: item[1]))
        for key, subdict in results.items()
    }

    print(json.dumps(sorted_results, indent=2))

    output_path = os.path.join(args.root_dir, "average_ranks_hubris.json")
    with open(output_path, "w") as f:
        json.dump(sorted_results, f, indent=2)

    print(f"Results successfully written to {output_path}")
