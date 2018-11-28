import os
import requests
import zipfile
import subprocess
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent.parent.resolve()
RAW_DATA_PATH = PROJECT_ROOT.joinpath("data/raw")
DOWNLOAD_PATH = PROJECT_ROOT.joinpath("data/raw/download")
TRAIN_PATH    = PROJECT_ROOT.joinpath("data/raw/train")
TEST_PATH     = PROJECT_ROOT.joinpath("data/raw/test")

def configure():
    """
    Make the project directory structure.
    """
    for path in [DOWNLOAD_PATH,TRAIN_PATH,TEST_PATH,RAW_DATA_PATH]:
        path.mkdir(parents=True,exist_ok=True)

def download_data():
    """
    Download the Protein Atlas Dataset by calling the Kaggle command line API.
    """
    download_cmd = ["kaggle", "competitions", "download", "-c",
                    "human-protein-atlas-image-classification", "-p",
                    "{}".format(DOWNLOAD_PATH)]
    subprocess.run(download_cmd)

def unzip_data():
    """
    Unzip test and train datasets from the Kaggle download.
    """
    zip_paths = [DOWNLOAD_PATH.joinpath(name) for name in ["train.zip","test.zip"]]
    zip_dests = [TRAIN_PATH, TEST_PATH]

    for zip_path, zip_dest in zip(zip_paths,zip_dests):
        print("Unzipping {}...".format(zip_path))
        print("\tdestination: {}".format(zip_dest),end="\n")
        dest_size = zip_dest.stat().st_size
        if dest_size < 100:
            zf = zipfile.ZipFile(zip_path)
            zf.extractall(zip_dest)
            zf.close()
            print("\tdone.")
        else:
            print("\talready has size {}, skipping.".format(dest_size))

def move_data():
    """
    Move data into place.
    """
    for csv_name in ["train.csv","test.csv"]:
        csv_path = DOWNLOAD_PATH.joinpath(csv_name)
        if csv_path.exists():
            csv_path.rename(RAW_DATA_PATH.joinpath(csv_name))

if __name__ == "__main__":
    configure()
    unzip_data()
    move_data()
