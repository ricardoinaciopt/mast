import importlib
import sys


def run_scripts(scripts, resamplers):
    errors = {}
    scores = {}

    for script_name in scripts:
        script_module = importlib.import_module(script_name)
        name_errors = script_name + "_errors"
        name_scores = script_name + "_roc_auc"
        errors[name_errors] = {}
        scores[name_scores] = {}

        for resampler in resamplers:
            try:
                print(f"\n\nStarting {script_name} with {resampler} resampling.\n")
                errors1, errors2, roc_auc_score_1, roc_auc_score_2 = script_module.main(
                    resampler
                )
                errors[name_errors] = (errors1, errors2)
                scores[name_scores][resampler] = (roc_auc_score_1, roc_auc_score_2)
            except Exception as e:
                print(f"Error processing {script_name} with {resampler}: {e}")

    return errors, scores


if __name__ == "__main__":
    scripts = ["m3_M", "m4_M", "tourism_M"]
    resamplers = ["SMOTE", "ADASYN"]

    errors, scores = run_scripts(scripts, resamplers)

    save_tables = importlib.import_module("json2latex")
    save_tables.save_latex_table(errors, "errors_to_latex_table")
    save_tables.save_latex_table(scores, "save_scores_table")

    sys.exit(0)
