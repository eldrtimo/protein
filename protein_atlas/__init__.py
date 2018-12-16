import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

from clint.textui import progress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator


from .install import PATH

class ProteinAtlas():
    def __init__(self,img_path):
        self.img_path = img_path
        # Make test and train datasets
        # n_splits = int(1/(1-train_portion))
        # mskf = MultilabelStratifiedKFold(
        #     n_splits = int(1/(1-train_portion)),
        #     random_state=random_state
        # )
        
        self.cmaps = []
        for chan_ix in range(self.n_channels):
            self.cmaps.append(self.make_cmap(chan_ix,as_cmap=True))

    def any(self, class_):
        """
        Return examples belonging to any of the classes in `class_`.

        Parameters:

            class_: int or iterable of ints

                Integer index or indices of classes to get examples from.
        """
        if isinstance(class_, pd.Index):
            mask = pd.DataFrame(self.labels.loc[:,class_] == 1).any(axis = 1)
        elif isinstance(class_, int) or (isinstance(class_, list) and isinstance(class_[0],int)):
            mask = pd.DataFrame(self.labels.iloc[:,class_] == 1).any(axis = 1)
        elif isinstance(class_, str):
            class_ = pd.Index([class_])
            mask = pd.DataFrame(self.labels.loc[:,class_] == 1).any(axis = 1)

        return self.labels.loc[mask]


    def render_batch(self,X):
        """Render a batch of images X in the HUSL colorspace for maximizing
        perceptual uniformity of each channel.

        Parameters:

            X : ndarray, (n_samples, rows, cols, 4)

        Returns:
        
            X_new : ndarray, (n_samples, rows_new, cols_new, 4)

               Batch of images in RGBa channels.
        """
        if len(X.shape) == 3:
            samples = 1
            row, cols, channels = X.shape
            X = X.reshape((1,rows,cols,channels))
        elif len(X.shape) == 4:
            samples, rows, cols, channels = X.shape
        else:
            pass

        bands = [0] * self.n_channels
        img = np.zeros((self.height,self.width,4))
        for chan in range(self.n_channels):
            bands[chan] = self.cmaps[chan](X[sample,:,:,chan])
            img = img + bands[chan]/self.n_channels
            
        img = Image.fromarray(np.uint8(img*255))
        return img

    @property
    def nrows(self): return 512

    @property
    def ncols(self): return 512

    def get_images(self,ids):
        """
        Given a pd.Index of example IDs, return an array of shape
        (samples,rows,cols,channels)
        """
        X = np.zeros((len(ids),self.nrows,self.ncols,self.n_channels))
        for sample_ix, id_ in enumerate(ids):
            X[sample_ix,:,:,:] = self.get_image(id_)

        return X

    def get_path(self,id_,channel_ix):
        """
        Get the file path for an image from the Protein Atlas Kaggle dataset.
        
        Parameters:

            id_ : str

                The ID of the sample.

            channel_ix : int, valid values in range(self.n_channels)

                The index of the image channel to retrieve. These are:
                    0. red (microtubules band)
                    1. green (antigen band)
                    2. blue (nucleus band)
                    3. yellow (endoplasmic reticulum band)
        """
        channel_color = self.channel_colors[channel_ix]
        return self.img_path.joinpath("{}_{}.png".format(id_,channel_color))

    @property
    def classes(self):
        """Set of target classes for the Protein Atlas dataset.

        Each of these classes represents a possible location in a human cell
        where a protein of interest resides.
        """
        return ["Nucleoplasm", "Nuclear membrane", "Nucleoli",
                "Nucleoli fibrillar center", "Nuclear speckles",
                "Nuclear bodies", "Endoplasmic reticulum", "Golgi apparatus",
                "Peroxisomes", "Endosomes", "Lysosomes",
                "Intermediate filaments", "Actin filaments",
                "Focal adhesion sites", "Microtubules", "Microtubule ends",
                "Cytokinetic bridge", "Mitotic spindle",
                "Microtubule organizing center", "Centrosome", "Lipid droplets",
                "Plasma membrane", "Cell junctions", "Mitochondria",
                "Aggresome", "Cytosol", "Cytoplasmic bodies", "Rods & rings"]

    @property
    def n_classes(self):
        return len(self.classes)
                       
    def make_cmap(self,chan_ix,**kwds):
        return sns.cubehelix_palette(start = chan_ix*3.0/self.n_channels,
                                     dark = 0, light = 1, gamma = 2.0, rot = 0,
                                     hue = 1, **kwds)

    @property
    def channel_colors(self):
        """
        Color of each channel in the Protein Atlas dataset.  These are not
        technically colors, just identifiers necessary to locate the path of
        each .png file.
        """
        return ["red", "green", "blue", "yellow"]


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


    def get_image(self,id_):
        """Get the 4-band image corresponding to example `id_`, returning a numpy array
        of shape (512, 512, 4)

        Parameters:
        
            id_ : str

                The ID of the sample.

        Returns:

            img : ndarray of float, shape (512, 512, 4)

                Values range between 0 and 1.
        """
        bands = [None] * self.n_channels
        for chan_ix in range(self.n_channels):
            chan_path = self.get_path(id_,chan_ix)
            bands[chan_ix] = Image.open(chan_path)
        
        return np.stack(bands, axis=2) / 255




class Test(ProteinAtlas):
    def __init__(self):
        super().__init__(img_path = PATH["test"])
        df = pd.read_csv(PATH["sample_submission.csv"]).set_index("Id")
        self.index = df.index

class Train(ProteinAtlas):
    def __init__(self):
        super().__init__(img_path = PATH["train"])
        df          = pd.read_csv(PATH["train.csv"]).set_index("Id")
        read_target = lambda s: np.array(s.split(" "),dtype=np.int32)
        targets     = df["Target"].apply(read_target)
        mlb         = MultiLabelBinarizer()
        labels      = mlb.fit_transform(targets.values)
        index       = targets.index
        columns     = self.classes
        self.labels = pd.DataFrame(labels,index,columns)
        self.index  = self.labels.index


# class ProteinAtlas():
#     """
#     Class for the ProteinAtlas
#     """

#     def __init__(self, train_portion = 0.9, random_state = None):
#         df          = pd.read_csv(self.csvpath).set_index("Id")
#         read_target = lambda s: np.array(s.split(" "),dtype=np.int32)
#         targets     = df["Target"].apply(read_target)
#         mlb         = MultiLabelBinarizer()
#         labels      = mlb.fit_transform(targets.values)
#         index       = targets.index
#         columns     = self.classes
        
#         self.labels = pd.DataFrame(labels,index,columns)

#         # Make test and train datasets
#         mskf = MultilabelStratifiedKFold(n_splits = int(1/(1-train_portion)), random_state = random_state)
#         train, test = mskf.split(X = self.labels, y = self.labels).__next__()

#         self.train = self.labels.iloc[train]
#         self.test  = self.labels.iloc[test]

#         def make_cmap(i,**kwds):
#             return sns.cubehelix_palette(start = i*3.0/self.n_channels,
#                                          dark = 0, light = 1,
#                                          gamma = 2.0, rot = 0, hue = 1,
#                                          **kwds)

#         self.cmaps = [make_cmap(i, as_cmap = True) for i in range(self.n_channels)]

        
#     def any(self, class_, labels=None,):
#         """
#         Return examples belonging to any of the classes in `class_`.

#         Parameters:

#             class_: int or iterable of ints

#                 Integer index or indices of classes to get examples from.
#         """

#         if labels is None:
#             labels = self.labels
        
#         mask = pd.DataFrame(labels.iloc[:,class_] == 1).any(axis = 1)
#         return labels.loc[mask]
        
#     def all(self,class_):
#         """
#         Return examples belonging to all of the classes in `class_`.

#         Parameters:

#             class_: int or iterable of ints

#                 Integer index or indices of classes to get examples from.
#         """
#         mask = pd.DataFrame(self.labels.iloc[:,class_] == 1).all(axis = 1)
#         return self.labels.loc[mask]

#     @property
#     def csvpath(self):
#         """The path where the train.csv index file is located."""
#         return PATH["train.csv"]

#     @property
#     def channels(self):
#         """
#         Channels present in the Protein Atlas dataset.
#         """
#         return ["Microtubules", "Antibody", "Nucleus", "Endoplasmic Reticulum"]

#     @property
#     def n_channels(self):
#         """Number of channels in each image of the Protein Atlas dataset."""
#         return len(self.channels)
    
#     @property
#     def channel_colors(self):
#         """
#         Color of each channel in the Protein Atlas dataset.  These are not
#         technically colors, just identifiers necessary to locate the path of
#         each .png file.
#         """
#         return ["red", "green", "blue", "yellow"]

#     @property
#     def width(self):
#         """Width of each image in the Protein Atlas dataset, in pixels"""
#         return 512

#     @property
#     def height(self):
#         """Height of each image in the Protein Atlas dataset, in pixels"""
#         return 512

#     @property
#     def depth(self):
#         """Number of channels in each pixel for images in the Protein Atlas dataset."""
#         return len(self.channels)

#     @property
#     def classes(self):
#         """Set of target classes for the Protein Atlas dataset.

#         Each of these classes represents a possible location in a human cell
#         where a protein of interest resides.
#         """
#         return [ "Nucleoplasm", "Nuclear membrane", "Nucleoli",
#                  "Nucleoli fibrillar center", "Nuclear speckles",
#                  "Nuclear bodies", "Endoplasmic reticulum",
#                  "Golgi apparatus", "Peroxisomes", "Endosomes", "Lysosomes",
#                  "Intermediate filaments", "Actin filaments",
#                  "Focal adhesion sites", "Microtubules", "Microtubule ends",
#                  "Cytokinetic bridge", "Mitotic spindle",
#                  "Microtubule organizing center", "Centrosome", "Lipid droplets",
#                  "Plasma membrane", "Cell junctions", "Mitochondria",
#                  "Aggresome", "Cytosol", "Cytoplasmic bodies", "Rods & rings" ]

#     @property
#     def n_classes(self):
#         return len(self.classes)

#     def get_path(self,id_,channel_ix):
#         """
#         Get the file path for an image from the Protein Atlas Kaggle dataset.
        
#         Parameters:

#             id_ : str

#                 The ID of the sample.

#             channel_ix : int, valid values in range(self.n_channels)

#                 The index of the image channel to retrieve. These are:
#                     0. red (microtubules band)
#                     1. green (antigen band)
#                     2. blue (nucleus band)
#                     3. yellow (endoplasmic reticulum band)
#         """
#         channel_color = self.channel_colors[channel_ix]
#         return PATH["train"].joinpath("{}_{}.png".format(id_,channel_color))

#     def get_image(self,id_):
#         """Get the 4-band image corresponding to example `id_`, returning a numpy array
#         of shape (512, 512, 4)

#         Parameters:
        
#             id_ : str

#                 The ID of the sample.

#         Returns:

#             img : ndarray of float, shape (512, 512, 4)

#                 Values range between 0 and 1.
#         """
#         bands = [None] * self.n_channels
#         for chan_ix in range(self.n_channels):
#             chan_path = self.get_path(id_,chan_ix)
#             bands[chan_ix] = Image.open(chan_path)
        
#         return np.stack(bands, axis=2) / 255

#     def get_batch(self,ids):
#         """
#         Given a pd.Index of example IDs, return an array of shape
#         (samples,rows,cols,channels)
#         """
#         y = self.labels.loc[ids].values
#         X = np.zeros((len(ids),self.height,self.width,self.n_channels))
#         for example_ix, id_ in enumerate(ids):
#             X[example_ix,:,:,:] = self.get_image(id_)

#         return X, y

#     def render_batch(self,X):
#         """
#         Parameters:

#             X : ndarray, (n_samples, rows, cols, 4)

#         Returns:
        
#             X_new : ndarray, (n_samples, rows_new, cols_new, 3)
#         """
#         if len(X.shape) == 3:
#             samples = 1
#             row, cols, channels = X.shape
#             X = X.reshape((1,rows,cols,channels))
#         elif len(X.shape) == 4:
#             samples, rows, cols, channels = X.shape
#         else:
#             pass

#         bands = [0] * self.n_channels
#         img = np.zeros((self.height,self.width,4))
#         for chan in range(self.n_channels):
#             bands[chan] = self.cmaps[chan](X[sample,:,:,chan])
#             img = img + bands[chan]/self.n_channels
            
#         img = Image.fromarray(np.uint8(img*255))
#         return img

# class DataGenerator(Sequence):
#     def __init__(self,batch_size = 32,labels=None, augment=False):
#         self.batch_size = batch_size
#         self.atlas = ProteinAtlas()

#         if labels is not None:
#             self.labels = labels
#         else:
#             self.labels = self.atlas.labels

#         self.n_batches = int(np.ceil(len(self.labels)/float(self.batch_size)))
#         self.mskf = MultilabelStratifiedKFold(n_splits = max(2,self.n_batches))

#         y = self.labels.values
#         self.batches = [test_ix for _, test_ix in self.mskf.split(y,y)]
#         self.augment = augment

#         self.image_gen = ImageDataGenerator(
#             horizontal_flip = True,
#             vertical_flip = True
#         )

#     def __len__(self):
#         return self.n_batches

#     def __iter__(self):
#         for X, y in [self[i] for i in range(len(self))]:
#             yield X, y

#     def __getitem__(self, ix):
#         batch_ixs = self.batches[ix]
#         ids = self.atlas.labels.iloc[batch_ixs].index
#         return self.atlas.get_batch(ids)

# if __name__ == "__main__":
#     print("Hello")
