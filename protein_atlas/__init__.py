import os
import subprocess
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from clint.textui import progress

# Absolute path to project root directory
ROOT  = Path(__file__).parent.parent.resolve()

PATH = {
    "root"  : ROOT,
    "data"  : ROOT.joinpath("data"),
    "raw"   : ROOT.joinpath("data/raw"),
    "test"  : ROOT.joinpath("data/raw/test"),
    "train" : ROOT.joinpath("data/raw/train"),
}

KAGGLE_FILE_TO_LOCAL = {
    "train.csv"    : PATH["raw"].joinpath("train.csv"),
    "test.zip"     : PATH["raw"].joinpath("test.zip"),
    "train.zip"    : PATH["raw"].joinpath("train.zip")
}

ZIPFILE_TO_OUTPUTDIR = {
    "test.zip" : PATH["test"],
    "train.zip" : PATH["train"],
}


def get_protein_atlas_zip(zipname,clean=False,force=False):
    """
    Download a Kaggle competition data .zip archive named `zipname` and store it in
    its own directory.

    Parameters:
        zipname : str
            name of the kaggle .zip file
        clean : bool, default False
            If true, delete the .zip file after expanding
        force : bool, default False
            If true, download and decompress the .zip file even if the directory 
            ZIPFILE_TO_DESTIONATION[zipfile] already exists.

    Throws: 
        subprocess.CalledProcessError:
            Subprocess call to kaggle returned with nonzero exit status.
        KeyError :
            Lookup to KAGGLE_FILE_TO_LOCAL[zipname] or ZIPFILE_TO_OUTPUTDIR[zipname] 
            failed.
    """
    outputdir = ZIPFILE_TO_OUTPUTDIR[zipname]
    if outputdir.exists() and not force:
        print("{}: Skipping, output directory {} already exists"
              .format(zipname,str(outputdir)))
    else:
        download_cmd = [
            "kaggle", "competitions", "download",
            "human-protein-atlas-image-classification",
            "-f", zipname,
            "-p", str(KAGGLE_FILE_TO_LOCAL[zipname].parent)
        ]
        print(" ".join(download_cmd))
        subprocess.run(download_cmd,check=True)
           
        with ZipFile(KAGGLE_FILE_TO_LOCAL[zipname]) as f:
            outputdir = ZIPFILE_TO_OUTPUTDIR[zipname]
            if not outputdir.exists(): outputdir.mkdir(parents=True,exist_ok=True)
            members = f.infolist()
            n_members = len(members)
            label = "unzipping {} ".format(zipname)
            with progress.Bar(label=label, expected_size=n_members) as bar:
                for i, member in enumerate(members):
                    if not outputdir.joinpath(member.filename).exists():
                        f.extract(member, path=outputdir)
                    bar.show(i)

        if clean:
            KAGGLE_FILE_TO_LOCAL[zipname].unlink()

def get_protein_atlas_csv(name):
    """Download a Kaggle competition .csv file name `name` and store it in
    PATH["raw"]
    """
    outputdir = KAGGLE_FILE_TO_LOCAL[name].parent
    if not outputdir.exists():
        outputdir.mkdir(parents=True,exist_ok=True)
        
    download_cmd = [
        "kaggle", "competitions", "download",
        "human-protein-atlas-image-classification",
        "-f", name,
        "-p", str(outputdir)
    ]
    print(" ".join(download_cmd))
    subprocess.run(download_cmd,check=True)

def get_protein_atlas_file(name, **kwds):
    suffix = Path(name).suffix
    if suffix == ".csv":
        get_protein_atlas_csv(name)
    elif suffix == ".zip":
        get_protein_atlas_zip(name, **kwds)
    else:
        print("skipping {}: extension {} not recognized".format(name,suffix))

if __name__ == "__main__":
    for name in KAGGLE_FILE_TO_LOCAL.keys():
        get_protein_atlas_file(name, clean=True)
    
