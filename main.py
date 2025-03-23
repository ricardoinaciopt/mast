import subprocess

"""
The following code generates the pipelines for the experiments.
Experiments are divided into three groups: Monthly, Quarterly, and Yearly.
Each group has a different number of periods for the horizon.
"""

cases = {
    "M3": ["Monthly", "Quarterly", "Yearly"],
    "M4": ["Monthly", "Quarterly", "Yearly"],
}
for data, groups in cases.items():
    for group in groups:
        h = {"Monthly": "12", "Quarterly": "4", "Yearly": "4"}.get(group)

        command = [
            "python",
            # ./errors_uid_pipeline.py for large errors experiment
            # ./uncertainty_uid_pipeline.py for uncertainty experiment
            # ./hubris_uid_pipeline.py for hubris (large error but large certainty) experiment
            "./errors_uid_pipeline.py",
            "--data",
            data,
            "--group",
            group,
            "--horizon",
            h,
            "--models",
            "LGBM",
            "--quantile",
            "0.80",
            "--level",
            "90",
        ]
        subprocess.run(command, check=True)
