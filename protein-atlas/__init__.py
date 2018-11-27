import os
import requests
import zipfile
import subprocess
from pathlib import Path
from clint.textui import progress


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
DATA_PATH     = os.path.join(PROJECT_ROOT,"data")
TRAIN_PATH    = os.path.join(PROJECT_ROOT,"data/raw/train")
TEST_PATH     = os.path.join(PROJECT_ROOT,"data/raw/test")

def configure():
    """
    Make the project directory structure.
    """
    for path in [DATA_PATH,TRAIN_PATH,TEST_PATH]:
        Path(path).mkdir(parents=True,exist_ok=True)


def get_data():
    """
    Download the Protein Atlas Dataset
    """
    download_cmd = ["kaggle", "competitions", "download", "-c",
                    "human-protein-atlas-image-classification", "-p",
                    "{}".format(DATA_PATH)]
    subprocess.run(download_cmd)

if __name__ == "__main__":
    print(PROJECT_ROOT)
    configure()
    get_data()
    
