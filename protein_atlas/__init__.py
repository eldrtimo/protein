# Standard Library Imports
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

from clint.textui import progress

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import equalize_adapthist
from sklearn.preprocessing import MultiLabelBinarizer

from .install import PATH

class ProteinAtlas():
    def __init__(self):
        df          = pd.read_csv(self.csvpath).set_index("Id")
        read_target = lambda s: np.array(s.split(" "),dtype=np.int32)
        targets     = df["Target"].apply(read_target)
        mlb         = MultiLabelBinarizer()
        y           = mlb.fit_transform(targets.values)
        index       = targets.index
        columns     = pd.MultiIndex.from_product([self.classes],names=["Target"])

        self.labels = pd.DataFrame(y,index,columns)

    def any(self,class_):
        """
        Return examples belonging to any of the classes in `class_`.

        Parameters:

            class_: int or iterable of ints

                Integer index or indices of classes to get examples from.
        """
        mask = pd.DataFrame(self.labels.iloc[:,class_] == 1).any(axis = 1)
        return self.labels.loc[mask]
        
    def all(self,class_):
        """
        Return examples belonging to all of the classes in `class_`.

        Parameters:

            class_: int or iterable of ints

                Integer index or indices of classes to get examples from.
        """
        mask = pd.DataFrame(self.labels.iloc[:,class_] == 1).all(axis = 1)
        return self.labels.loc[mask]

    @property
    def csvpath(self): return PATH["train.csv"]

    @property
    def channels(self):
        """
        Channels present in the Protein Atlas dataset.
        """
        return [
            "Microtubules",
            "Antibody",
            "Nucleus",
            "Endoplasmic Reticulum"
        ]

    @property
    def n_channels(self):
        """Number of channels in each image of the Protein Atlas dataset."""
        return len(self.channels)
    
    @property
    def channel_colors(self):
        """
        Color of each channel in the Protein Atlas dataset.  These are not
        technically colors, just identifiers necessary to locate the path of
        each .png file.
        """
        return ["red", "green", "blue", "yellow"]

    @property
    def width(self):
        """Width of each image in the Protein Atlas dataset, in pixels"""
        return 512

    @property
    def height(self):
        """Height of each image in the Protein Atlas dataset, in pixels"""
        return 512

    @property
    def depth(self):
        """Number of channels in each pixel for images in the Protein Atlas dataset."""
        return len(self.channels)

    @property
    def classes(self):
        return [
            "Nucleoplasm",
            "Nuclear membrane",
            "Nucleoli",
            "Nucleoli fibrillar center",
            "Nuclear speckles",
            "Nuclear bodies",
            "Endoplasmic reticulum",
            "Golgi apparatus",
            "Peroxisomes",
            "Endosomes",
            "Lysosomes",
            "Intermediate filaments",
            "Actin filaments",
            "Focal adhesion sites",
            "Microtubules",
            "Microtubule ends",
            "Cytokinetic bridge",
            "Mitotic spindle",
            "Microtubule organizing center",
            "Centrosome",
            "Lipid droplets",
            "Plasma membrane",
            "Cell junctions",
            "Mitochondria",
            "Aggresome",
            "Cytosol",
            "Cytoplasmic bodies",
            "Rods & rings",
        ]

    @property
    def n_classes(self):
        return len(self.classes)

    def get_path(self,id_,channel_ix):
        channel_color = self.channel_colors[channel_ix]
        return PATH["train"].joinpath("{}_{}.png".format(id_,channel_color))

    def get_image(self,id_):
        img = np.zeros((self.width,self.height,self.n_channels))
        for channel_ix in range(self.n_channels):
            channel_path = self.get_path(id_,channel_ix)
            img[:,:,channel_ix] = plt.imread(str(channel_path))
        return img

    def plot_intensities(self, img):
        pass

    def imshow(self,id_, ax=None):
        if not ax:
            ax = plt.gca()
        
        img = self.get_image(id_)
        img = equalize_adapthist(img)
        ax.imshow(img)
