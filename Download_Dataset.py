import os
import subprocess


def download_datasets():

    # Create .kaggle directory and move kaggle.json (API key) there
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    kaggle_api_path = os.path.expanduser("~/.kaggle/kaggle.json")

    if not os.path.exists(kaggle_api_path):
        raise FileNotFoundError(
            "Kaggle API key not found. Please place kaggle.json in ~/.kaggle/"
        )

    os.chmod(kaggle_api_path, 0o600)

    # Define the datasets and their Kaggle paths
    datasets = {
        "ravdess": "uwrfkaggler/ravdess-emotional-speech-audio",
        "tess": "ejlok1/toronto-emotional-speech-set-tess",
    }

    # Define the raw data directory
    raw_data_dir = "data/raw"
    os.makedirs(raw_data_dir, exist_ok=True)

    # Download the datasets using Kaggle API
    for name, kaggle_path in datasets.items():
        print(f"Downloading {name} dataset...")
        subprocess.call(
            [
                "kaggle",
                "datasets",
                "download",
                "--unzip",
                "--path",
                raw_data_dir,
                kaggle_path,
            ]
        )

    print("Datasets downloaded successfully.")


# Run the function to download data
download_datasets()
