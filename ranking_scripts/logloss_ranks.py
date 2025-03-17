import pandas as pd
import os
import json
import argparse

parser = argparse.ArgumentParser(
    description="Generate LaTeX table for Log Loss analysis from a directory of JSON files."
)
parser.add_argument(
    "directory",
    type=str,
    help="Path to the directory containing JSON files.",
)
args = parser.parse_args()

datasets = {}
for file_name in os.listdir(args.directory):
    if file_name.endswith(".json"):
        key = (
            file_name.split("_", 1)[1]
            .rsplit(".", 1)[0]
            .replace("_", " ")
            .replace("90 ", "")
        )
        file_path = os.path.join(args.directory, file_name)
        with open(file_path, "r") as file:
            data = json.load(file)
            if "log_losses" in data:
                datasets[key] = data["log_losses"]


def format_column(df_formatted, df_numeric, column):
    col_values = df_numeric[column]
    smallest_value = col_values.min()
    second_smallest_value = col_values[col_values > smallest_value].min()

    # bold all smallest values
    smallest_indices = col_values[col_values == smallest_value].index
    for idx in smallest_indices:
        df_formatted.loc[idx, column] = f"\\textbf{{{df_formatted.loc[idx, column]}}}"

    # underline all second smallest values
    second_smallest_indices = col_values[col_values == second_smallest_value].index
    for idx in second_smallest_indices:
        df_formatted.loc[idx, column] = (
            f"\\underline{{{df_formatted.loc[idx, column]}}}"
        )


def generate_latex(datasets):
    df = pd.DataFrame(datasets)
    df_numeric = df.astype(float)
    avg_log_loss = df_numeric.mean(axis=1).round(3)
    ranked_df = df_numeric.rank(axis=0, method="min", ascending=True)
    avg_rank = ranked_df.mean(axis=1).round(2)

    # 3 decimal places for log loss, 1 decimal place for rank
    df_numeric_formatted = df_numeric.map("{:.3f}".format)
    df_numeric["Avg. Log Loss"] = avg_log_loss
    df_numeric["Avg. Rank"] = avg_rank
    df_numeric_formatted["Avg. Log Loss"] = avg_log_loss.map("{:.3f}".format)
    df_numeric_formatted["Avg. Rank"] = avg_rank.map("{:.1f}".format)

    df_formatted = df_numeric_formatted.copy()
    column_order = [
        "M3 Monthly",
        "M3 Quarterly",
        "M3 Yearly",
        "M4 Monthly",
        "M4 Quarterly",
        "M4 Yearly",
        "Avg. Log Loss",
        "Avg. Rank",
    ]
    df_formatted = df_formatted[column_order]

    for column in df_numeric.columns:
        format_column(df_formatted, df_numeric, column)

    latex_code = df_formatted.to_latex(
        float_format="%.3f",
        caption=(
            "Metamodel performance in each dataset according to \\textit{Log Loss}. "
            "Bold font denotes the best method in each dataset, while underlined font denotes second best. "
            "Average Log Loss and Rank analysis are also included as columns."
        ),
        label="tab:results2",
        escape=False,
        header=True,
        index=True,
        column_format="|l|" + "c|" * len(df_formatted.columns),
    )

    latex_code = (
        latex_code.replace("\\toprule", "\\hline")
        .replace("\\midrule", "\\hline")
        .replace("\\bottomrule", "\\hline")
    )
    return latex_code


if __name__ == "__main__":
    latex_table = generate_latex(datasets)
    print(latex_table)
