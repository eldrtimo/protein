# Standard Library Imports
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

from clint.textui import progress

import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import equalize_adapthist
from sklearn.preprocessing import MultiLabelBinarizer

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image


from .install import PATH


class ProteinAtlas():
    """
    Class for the ProteinAtlas
    """

    def __init__(self, train_portion = 0.9, random_state = None):
        df          = pd.read_csv(self.csvpath).set_index("Id")
        read_target = lambda s: np.array(s.split(" "),dtype=np.int32)
        targets     = df["Target"].apply(read_target)
        mlb         = MultiLabelBinarizer()
        labels      = mlb.fit_transform(targets.values)
        index       = targets.index
        columns     = self.classes
        
        self.labels = pd.DataFrame(labels,index,columns)

        # Make test and train datasets
        mskf = MultilabelStratifiedKFold(n_splits = int(1/(1-train_portion)), random_state = random_state)
        train, test = mskf.split(X = self.labels, y = self.labels).__next__()

        self.train = self.labels.iloc[train]
        self.test  = self.labels.iloc[test]

        def make_cmap(i,**kwds):
            return sns.cubehelix_palette(start = i*3.0/self.n_channels,
                                         dark = 0, light = 1,
                                         gamma = 2.0, rot = 0, hue = 1,
                                         **kwds)

        self.cmaps = [make_cmap(i, as_cmap = True) for i in range(self.n_channels)]

        
    def any(self, class_, labels=None,):
        """
        Return examples belonging to any of the classes in `class_`.

        Parameters:

            class_: int or iterable of ints

                Integer index or indices of classes to get examples from.
        """

        if labels is None:
            labels = self.labels
        
        mask = pd.DataFrame(labels.iloc[:,class_] == 1).any(axis = 1)
        return labels.loc[mask]
        
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
        return ["Microtubules", "Antibody", "Nucleus", "Endoplasmic Reticulum"]

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
        """Set of target classes for the Protein Atlas dataset.

        Each of these classes represents a possible location in a human cell
        where a protein of interest resides.
        """
        return [ "Nucleoplasm", "Nuclear membrane", "Nucleoli",
                 "Nucleoli fibrillar center", "Nuclear speckles",
                 "Nuclear bodies", "Endoplasmic reticulum",
                 "Golgi apparatus", "Peroxisomes", "Endosomes", "Lysosomes",
                 "Intermediate filaments", "Actin filaments",
                 "Focal adhesion sites", "Microtubules", "Microtubule ends",
                 "Cytokinetic bridge", "Mitotic spindle",
                 "Microtubule organizing center", "Centrosome", "Lipid droplets",
                 "Plasma membrane", "Cell junctions", "Mitochondria",
                 "Aggresome", "Cytosol", "Cytoplasmic bodies", "Rods & rings" ]

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
        return np.stack(bands, axis=2) / 255

    def get_batch(self,ids):
        y = self.labels.loc[ids].values
        X = np.zeros((len(ids),self.height,self.width,self.n_channels))
        for example_ix, id_ in enumerate(ids):
            X[example_ix,:,:,:] = self.get_image(id_)

        return X, y

    

class DataGenerator(Sequence):
    def __init__(self,batch_size = 32,labels = None, augment = False):
        self.batch_size = batch_size
        self.atlas = ProteinAtlas()

        if labels is not None:
            self.labels = labels
        else:
            self.labels = self.atlas.labels

        self.n_batches = int(np.ceil(len(self.labels)/float(self.batch_size)))
        self.mskf = MultilabelStratifiedKFold(n_splits = max(2,self.n_batches))

        y = self.labels.values
        self.batches = [test_ix for _, test_ix in self.mskf.split(y,y)]
        self.augment = augment

        self.image_gen = ImageDataGenerator(
            horizontal_flip = True,
            vertical_flip = True
        )

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for X, y in [self[i] for i in range(len(self))]:
            yield X, y

    def __getitem__(self, ix):
        batch_ixs = self.batches[ix]
        ids = self.atlas.labels.iloc[batch_ixs].index
        return self.atlas.get_batch(ids)
