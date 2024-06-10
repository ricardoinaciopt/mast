import os


def errors_to_latex_table(data):
    """
    Convert error data to LaTeX table format.

    Args:
        data (dict): A dictionary containing error data.

    Returns:
        str: LaTeX table representation of the error data.
    """
    table = "\\begin{table}[htbp]\n\\centering\n"
    table += "\\begin{tabular}{|l|c|c|}\n"
    table += "\\hline\n"
    table += "Dataset & SeasonalNaive & ForecastingModel \\\\\n"
    table += "\\hline\n"

    for key, value in data.items():
        lines = []
        for inner_list in value:
            for inner_dict in inner_list:
                if inner_dict["metric"] == "smape":
                    lines.append(
                        "{:.4f} & {:.4f} \\\\\n".format(
                            inner_dict["SeasonalNaive"], inner_dict["ForecastingModel"]
                        )
                    )
        if lines:
            table += "{} (Phase I) & {} \n".format(
                key.replace("_errors", "")
                .replace("_roc_auc", "")
                .replace("_", " ")
                .upper(),
                lines[0],
            )
            if len(lines) > 1:
                table += "{} (Phase II) & {} \n".format(
                    key.replace("_errors", "")
                    .replace("_roc_auc", "")
                    .replace("_", " ")
                    .upper(),
                    lines[1],
                )
            else:
                table += "{} (Phase II) & \\\\\n".format(
                    key.replace("_errors", "")
                    .replace("_roc_auc", "")
                    .replace("_", " ")
                    .upper()
                )
            for line in lines[2:]:
                table += "& {} \n".format(line)
            table += "\\hline\n"

    table += "\\end{tabular}\n"
    table += "\\end{table}\n\n"

    return table


def save_scores_table(data):
    """
    Convert score data to LaTeX table format.

    Args:
        data (dict): A dictionary containing score data.

    Returns:
        str: LaTeX table representation of the score data.
    """
    columns = set()
    for inner_dict in data.values():
        columns.update(inner_dict.keys())

    table = ""

    for key, inner_dict in data.items():
        table += "\\begin{table}[htbp]\n\\centering\n"
        table += "\\begin{tabular}{|l" + "|c" * (len(columns) + 1) + "|}\n"

        table += "\\hline\n"
        table += (
            " & "
            + " & ".join(
                column.replace("_errors", "").replace("_roc_auc", "").replace("_", " ")
                for column in columns
            )
            + " & No Augmentation\\\\\n"
        )
        table += "\\hline\n"

        table += (
            key.replace("_errors", "").replace("_roc_auc", "").replace("_", " ").upper()
            + " & "
        )
        base_value = None
        for column in columns:
            if column in inner_dict:
                table += f"{inner_dict[column][1]:.4f}" + " & "
                if base_value is None:
                    base_value = inner_dict[column][0]
            else:
                table += " & "
        if base_value is not None:
            table += f"{float(base_value):.4f}" + " \\\\\n"
        else:
            table += " \\\\\n"

        table += "\\hline\n"
        table += "\\end{tabular}\n"
        table += "\\end{table}\n\n"

    return table


def save_latex_table(data, function_name, folder="analysis"):
    """
    Save LaTeX table to a file.

    Args:
        data (dict): A dictionary containing the data to be converted to LaTeX table.
        function_name (str): Name of the function generating the LaTeX table.
        folder (str, optional): Name of the folder to save the LaTeX table file. Defaults to "analysis".
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    if function_name == "save_scores_table":
        latex_tables = save_scores_table(data)
        filename = "auc_scores_"
    elif function_name == "errors_to_latex_table":
        latex_tables = errors_to_latex_table(data)
        filename = "errors_"
    else:
        raise ValueError("Invalid function name provided.")

    keys_str = "_".join(
        key.replace("_errors", "").replace("_roc_auc", "").replace("_", "")
        for key in data.keys()
    )
    filename += f"{keys_str}.tex"

    filepath = os.path.join(folder, filename)
    with open(filepath, "w") as file:
        file.write(latex_tables)
    print(f"\n{filepath} saved successfully.\n")
