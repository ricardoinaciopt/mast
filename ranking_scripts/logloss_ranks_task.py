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


def process_q_folders(root_dir, task_name):
    results = {}

    for q_folder in os.listdir(root_dir):
        q_path = os.path.join(root_dir, q_folder)

        if not os.path.isdir(q_path) or not q_folder.startswith("q"):
            continue
        print(f"Processing {q_folder}...")

        task_path = os.path.join(q_path, task_name)
        if not os.path.exists(task_path):
            continue
        print(f"\tProcessing {task_path}...")

        method_ranks = defaultdict(list)

        for file_name in os.listdir(task_path):
            if not file_name.endswith(".json"):
                continue
            print(f"\t\tProcessing {file_name}...")
            file_path = os.path.join(task_path, file_name)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    by_id = data.get("by_id", {})

                    print("\t\t\tProcessing", len(by_id), " time series...")
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

        results[q_folder] = avg_ranks

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate method performance ranks across Q folders"
    )
    parser.add_argument("root_dir", help="Directory containing Q folders")
    parser.add_argument("task", help="Task subfolder name (le/lu)")

    args = parser.parse_args()

    results = process_q_folders(args.root_dir, args.task)

    sorted_results = {
        key: dict(sorted(subdict.items(), key=lambda item: item[1]))
        for key, subdict in results.items()
    }

    print(json.dumps(sorted_results, indent=2))

    filename = f"average_ranks_{args.task}.json"
    output_path = os.path.join(args.root_dir, filename)

    with open(output_path, "w") as f:
        json.dump(sorted_results, f, indent=2)

    print(f"Results successfully written to {output_path}")
