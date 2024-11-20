import subprocess

cases = {
    "M3": ["Monthly", "Quarterly", "Yearly"],
    "M4": ["Monthly", "Quarterly", "Yearly"],
}


for data, groups in cases.items():
    for group in groups:
        h = {"Monthly": "12", "Quarterly": "4", "Yearly": "4"}.get(group)

        command = [
            "python",
            "./aug_pipeline.py",
            "--data",
            data,
            "--group",
            group,
            "--horizon",
            h,
            "--models",
            "LGBM",
            # "Ridge",
            # "XGB",
            # "Lasso",
            # "LinearRegression",
            # "ElasticNet",
            # "CatBoost",
        ]
        subprocess.run(command, check=True)
