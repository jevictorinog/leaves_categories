import os
import numpy as np


class LeafCategory:
    """docstring for LeafCategory"""

    path_data  = '/home/jorge/data/leaf_image/ImageCLEF2012/data/bin_species_part/'
    data_file  = 'species.xlsx' # input data file of species specifications

    def __init__(self, lst_species, num_samples):
        self.species = lst_species
        self.num_samples = num_samples
        self.num_leaves  = 0
        self.num_species = len(lst_species)





palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
           'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan','mediumseagreen', 'coral',
           'cornflowerblue', 'lawngreen', 'gold', 'burlywood', 'silver','chocolate',
           'slateblue','slategray','orchid','tan','rosybrown', 'darkgolgenrod', 'k']


