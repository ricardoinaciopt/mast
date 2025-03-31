import os
import json
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python generate_tables.py <folder_path> <avg_columns>")
    sys.exit(1)

folder_path = sys.argv[1]

avg_columns = False
if len(sys.argv) > 2:
    try:
        avg_columns = sys.argv[2].lower() in ("true", "1", "yes")
    except ValueError:
        print("Error: avg_columns should be a boolean value (true/false).")
        sys.exit(1)


data = {}

for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        dataset_name = file_name.split(".json")[0].split("_", 2)[-1].replace("_", " ")
        with open(os.path.join(folder_path, file_name), "r") as file:
            json_data = json.load(file)
            log_losses = json_data.get("log_losses", {})
            for method, value in log_losses.items():
                if method not in data:
                    data[method] = {}
                data[method][dataset_name] = value

df = pd.DataFrame(data).T
datasets = sorted(df.columns)
df = df.reindex(columns=datasets)

rus_row = df.loc["RandomUnderSampler"]
df = df.drop("RandomUnderSampler")
no_resampling_row = df.loc["No Resampling"]
df = df.drop("No Resampling")
df = pd.concat([df, pd.DataFrame([rus_row], index=["RandomUnderSampler"])])
df = pd.concat([df, pd.DataFrame([no_resampling_row], index=["No Resampling"])])

if avg_columns:
    df["Avg Log Loss"] = df.mean(axis=1)

ranks = df[datasets].rank(
    axis=0, method="dense", ascending=True
)  # dense ranking accounts for ties

if avg_columns:
    df["Avg Rank"] = ranks.mean(axis=1)

latex_rows = []
for method, row in df.iterrows():
    formatted_row = [method]
    for col in datasets:
        column_values = df[col]
        ranks_col = column_values.rank(ascending=True, method="dense")

        min_value = column_values[ranks_col == 1].min()
        second_min_value = column_values[ranks_col == 2].min()

        if round(row[col], 3) == round(min_value, 3):
            formatted_row.append(f"\\textbf{{{row[col]:.3f}}}")
        elif round(row[col], 3) in round(column_values[ranks_col == 2], 3).tolist():
            formatted_row.append(f"\\underline{{{row[col]:.3f}}}")
        else:
            formatted_row.append(f"{row[col]:.3f}")

    if avg_columns:
        avg_log_loss_rank = df["Avg Log Loss"].rank(ascending=True, method="dense")
        min_avg_log_loss = df["Avg Log Loss"][avg_log_loss_rank == 1].min()
        second_min_avg_log_loss = df["Avg Log Loss"][avg_log_loss_rank == 2].min()

        if row["Avg Log Loss"] == min_avg_log_loss:
            formatted_row.append(f"\\textbf{{{row['Avg Log Loss']:.3f}}}")
        elif row["Avg Log Loss"] == second_min_avg_log_loss:
            formatted_row.append(f"\\underline{{{row['Avg Log Loss']:.3f}}}")
        else:
            formatted_row.append(f"{row['Avg Log Loss']:.3f}")

        avg_rank_rank = df["Avg Rank"].rank(ascending=True, method="dense")
        min_avg_rank = df["Avg Rank"][avg_rank_rank == 1].min()
        second_min_avg_rank = df["Avg Rank"][avg_rank_rank == 2].min()

        if row["Avg Rank"] == min_avg_rank:
            formatted_row.append(f"\\textbf{{{row['Avg Rank']:.1f}}}")
        elif row["Avg Rank"] == second_min_avg_rank:
            formatted_row.append(f"\\underline{{{row['Avg Rank']:.1f}}}")
        else:
            formatted_row.append(f"{row['Avg Rank']:.1f}")

    latex_rows.append(" & ".join(formatted_row) + " \\\\")

if avg_columns:
    latex_table = (
        """
    \\begin{table}
    \\centering
    \\setlength{\\tabcolsep}{5pt}
    \\renewcommand{\\arraystretch}{1.2}
    \\scalebox{0.7}{
    \\begin{tabular}{|l|"""
        + "c" * len(datasets)
        + """|c|c|}
    \\hline
    Method & """
        + " & ".join(datasets)
        + """ & Avg. Log Loss & Avg. Rank \\\\
    \\hline
    """
        + "\n".join(latex_rows[:-2])
        + """
    \\hline
    """
        + "\n".join(latex_rows[-2:])
        + """
    \\hline
    \\end{tabular}
    }
    \\end{table}
    """
    )
else:
    latex_table = (
        """
    \\begin{table}
    \\centering
    \\setlength{\\tabcolsep}{5pt}
    \\renewcommand{\\arraystretch}{1.2}
    \\scalebox{0.75}{
    \\begin{tabular}{|l|"""
        + "c" * len(datasets)
        + """|}
    \\hline
    Method & """
        + " & ".join(datasets)
        + """ \\\\
    \\hline
    """
        + "\n".join(latex_rows[:-2])
        + """
    \\hline
    """
        + "\n".join(latex_rows[-2:])
        + """
    \\hline
    \\end{tabular}
    }
    \\end{table}
    """
    )

print(latex_table)

# Extract folder structure relative to "results/"
results_index = folder_path.find("results" + os.sep)
if results_index != -1:
    folder_name = (
        folder_path[results_index + len("results" + os.sep) :]
        .strip(os.sep)
        .replace(os.sep, "_")
        .replace(" ", "_")
    )
else:
    folder_name = os.path.basename(os.path.normpath(folder_path)).replace(" ", "_")

file_name = f"{folder_name}.tex"

output_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(folder_path)))),
    "tables_task",
)
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, file_name)

with open(output_file_path, "w") as tex_file:
    tex_file.write(latex_table)
