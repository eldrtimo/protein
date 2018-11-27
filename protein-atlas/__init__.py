import os
import requests
from pathlib import Path
from clint.textui import progress

PROJECT_ROOT       = os.path.join(os.path.dirname(__file__),os.pardir)

DATA_PATH     = os.path.join(PROJECT_ROOT,"data")
TRAIN_PATH    = os.path.join(PROJECT_ROOT,"data/raw/train")
TEST_PATH     = os.path.join(PROJECT_ROOT,"data/raw/test")

KAGGLE_DATA_URL    = "https://www.kaggle.com/c/10418/download-all"
DOWNLOAD_FILE = os.path.join(DATA_PATH,"download.zip")


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
    r = requests.get(KAGGLE_DATA_URL, stream=True)
    with open(DOWNLOAD_FILE, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
            if chunk:
                f.write(chunk)
                f.flush()

if __name__ == "__main__":
    configure()
    get_data()
    
