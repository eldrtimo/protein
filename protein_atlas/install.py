# Standard Library Imports
import os
import subprocess
from zipfile import ZipFile

# Package imports
from pathlib import Path
from clint.textui import progress

# Root project directory, one level above the module directory.
ROOT = Path(__file__).parent.parent.resolve()

# Project data paths
PATH = {
    "root"      : ROOT,
    "data"      : ROOT.joinpath("data"),
    "raw"       : ROOT.joinpath("data/raw"),
    "test"      : ROOT.joinpath("data/raw/test"),
    "train"     : ROOT.joinpath("data/raw/train"),
    "train.csv" : ROOT.joinpath("data/raw/train.csv"),
    "test.zip"  : ROOT.joinpath("data/raw/test.zip"),
    "train.zip" : ROOT.joinpath("data/raw/train.zip")
}

KAGGLE_FILES = ["train.zip", "train.csv"]

def get_protein_atlas_zip(zipname,clean=False,force=False):
    """Download a Kaggle competition data .zip archive named `zipname` and store it
    in its own directory whose name is the stem of `zipname`.

    Parameters:

        zipname : str

            name of the kaggle .zip file. ***UNDEFINED BEHAVIOR WHEN zipname HAS NO FILE EXTENSION***

        clean : bool, default False

            If true, delete the .zip file after expanding

        force : bool, default False

            If true, download and decompress the .zip file even if the directory
            ZIPFILE_TO_DESTIONATION[zipfile] already exists.

    Throws: 

        subprocess.CalledProcessError:

            Subprocess call to kaggle returned with nonzero exit status.

        KeyError :

            PATH lookup failed.

    """
    outputdir_name = Path(zipname).stem
    outputdir = PATH[ outputdir_name ]
    if outputdir.exists() and not force:
        print("{}: Skipping, output directory {} already exists"
              .format(zipname,str(outputdir)))
    else:
        download_cmd = [
            "kaggle", "competitions", "download",
            "human-protein-atlas-image-classification",
            "-f", zipname,
            "-p", str(PATH[zipname].parent)
        ]
        print(" ".join(download_cmd))
        subprocess.run(download_cmd,check=True)
           
        with ZipFile(PATH[zipname]) as f:
            outputdir = PATH[zipname]
            if not outputdir.exists():
                outputdir.mkdir(parents=True,exist_ok=True)
            members   = f.infolist()
            n_members = len(members)
            label     = "unzipping {} ".format(zipname)
            with progress.Bar(label=label, expected_size=n_members) as bar:
                for i, member in enumerate(members):
                    if not outputdir.joinpath(member.filename).exists():
                        f.extract(member, path=outputdir)
                    bar.show(i)

        if clean:
            PATH[zipname].unlink()

def get_protein_atlas_csv(name):
    """Download a Kaggle competition .csv file name `name` and store it in
    PATH["raw"]
    """
    outputdir = PATH[name].parent
    if not outputdir.exists():
        outputdir.mkdir(parents=True,exist_ok=True)
        
    download_cmd = [ "kaggle", "competitions", "download",
                     "human-protein-atlas-image-classification", "-f", name,
                     "-p", str(outputdir) ]

    print(" ".join(download_cmd))
    subprocess.run(download_cmd,check=True)

def get_protein_atlas_file(name,clean=False):
    """Download a Kaggle competition file `name` and store the file at
    `PATH[name]`. If `name` is a .csv file, run get_protein_atlas_csv(name).  If
    `name` is a .zip file, run `get_protein_atlas_zip(name)` and store the
    contents of the archive at `PATH[name]`.

    Patemeters:

        name : string

        clean : bool, default False

            If `clean` is `True`, delete .zip files after unzipping.
    """
    suffix = Path(name).suffix
    if suffix == ".csv":
        get_protein_atlas_csv(name)
    elif suffix == ".zip":
        get_protein_atlas_zip(name,clean)
    else:
        print("skipping {}: extension {} not recognized".format(name,suffix))

def install():
    for name in KAGGLE_FILES:
        get_protein_atlas_file(name, clean=True)

if __name__ == "__main__":
    install()

