import subprocess
import concurrent.futures
import os

cases = {
    # "M3": ["Monthly", "Quarterly", "Yearly"],
    "M4": ["Monthly", "Quarterly", "Yearly"],
}

MODELS = "LGBM"
QUANTILE = "0.95"
LEVEL = "90"


def run_command(script, data, group, h):
    command = [
        "python",
        script,
        "--data",
        data,
        "--group",
        group,
        "--horizon",
        h,
        "--models",
        MODELS,
        "--quantile",
        QUANTILE,
        "--level",
        LEVEL,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command {command}: {e}")


scripts = [
    # "./errors_uid_pipeline.py",
    "./uncertainty_uid_pipeline.py",
    # "./hubris_uid_pipeline.py",
]

tasks = [
    (script, data, group, {"Monthly": "12", "Quarterly": "4", "Yearly": "4"}.get(group))
    for data in cases
    for group in cases[data]
    for script in scripts
]

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(run_command, *task) for task in tasks}
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Task failed with exception: {e}")
