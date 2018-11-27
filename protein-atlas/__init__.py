import os

ROOT = os.path.join(os.path.dirname(__file__),os.pardir)
DATA_URL = "https://www.kaggle.com/c/10418/download-all"

def configure():
    """
    Make the directory structure for project.
    """
    dirs = [
        "data",
        "data/raw",
        "data/processed",
    ]
    for dir in dirs:
        os.makedirs(os.path.join(ROOT,dir))

def get_data():
    """
    Download the Protein Atlas Dataset
    """
    pass



if __name__ == "__main__":
    configure()

