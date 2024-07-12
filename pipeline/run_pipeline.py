import subprocess
import sys
import os


def run_script(script_name):
    try:
        # Use sys.executable to get the path to the current Python interpreter
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e)


def check_files(files):
    for file in files:
        if not os.path.exists(file):
            return False
    return True


scripts_and_outputs = [
    (
        "data_processing.py",
        ["../data/processed/processed_data.npy"],
    ),  # Add the actual output file(s) for data_processing.py
    (
        "feature_extraction.py",
        ["../data/processed/features.npy", "../data/processed/labels.npy"],
    ),
    (
        "train.py",
        ["../models/emotion_recognition_model.h5", "../models/training_history.pkl"],
    ),
    (
        "evaluate.py",
        ["../results/evaluation_report.txt"],
    ),
]

for script, output_files in scripts_and_outputs:
    print(f"Checking outputs for {script}...")
    if check_files(output_files):
        print(f"Outputs for {script} already exist. Skipping...")
    else:
        print(f"Running {script}...")
        run_script(os.path.join(os.path.dirname(__file__), script))
