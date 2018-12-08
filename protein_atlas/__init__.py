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

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from keras.utils import Sequence
from PIL import Image


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
    def csvpath(self):
        """The path where the train.csv index file is located."""
        return PATH["train.csv"]

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
        return [ "Nucleoplasm", "Nuclear membrane", "Nucleoli",
                 "Nucleoli fibrillar center", "Nuclear speckles",
                 "Nuclear bodies", "Endoplasmic reticulum",
                 "Golgi apparatus", "Peroxisomes", "Endosomes", "Lysosomes",
                 "Intermediate filaments", "Actin filaments", "Focal adhesion sites",
                 "Microtubules", "Microtubule ends", "Cytokinetic bridge",
                 "Mitotic spindle", "Microtubule organizing center",
                 "Centrosome", "Lipid droplets", "Plasma membrane",
                 "Cell junctions", "Mitochondria", "Aggresome", "Cytosol",
                 "Cytoplasmic bodies", "Rods & rings" ]
    @property
    def n_classes(self):
        return len(self.classes)

    def get_path(self,id_,channel_ix):
        channel_color = self.channel_colors[channel_ix]
        return PATH["train"].joinpath("{}_{}.png".format(id_,channel_color))

    def get_image(self,id_):
        bands = [None] * self.n_channels
        for chan_ix in range(self.n_channels):
            chan_path = self.get_path(id_,chan_ix)
            bands[chan_ix] = Image.open(chan_path)

        return np.stack(bands, axis = 2) / 255
    def plot_intensities(self, img):
        pass



class ProteinAtlasGenerator(Sequence):
    def __init__(self,batch_size = 32,labels = None):
        self.batch_size = batch_size
        self.atlas = ProteinAtlas()

        if labels is not None:
            self.labels = labels
        else:
            self.labels = self.atlas.labels

        self.n_batches = int(np.ceil(len(self.labels)/float(self.batch_size)))
        self.mskf = MultilabelStratifiedKFold(n_splits = self.n_batches)

        y = self.labels.values
        self.batches = [test_ix for _, test_ix in self.mskf.split(y,y)]


    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for item in [self[i] for i in range(len(self))]:
            yield item

    def __getitem__(self, ix):
        batch_ixs = self.batches[ix]

        # Dimensions of output array X
        n_examples = len(batch_ixs)
        rows = self.atlas.height
        cols = self.atlas.width
        channels = self.atlas.n_channels

        ids = self.labels.index.values[batch_ixs]
        y_batch = self.labels.values[batch_ixs]
        x_batch = np.zeros((n_examples, rows, cols, channels))

        for example_ix, id_ in enumerate(ids):
            x_batch[example_ix,:,:,:] = self.atlas.get_image(id_)

        return x_batch, y_batch
