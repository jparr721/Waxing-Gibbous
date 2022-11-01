import os

import matplotlib.pyplot as plt
import numpy as np


def binarize_grid(grid_entry: np.ndarray):
    grid_entry[np.nonzero(grid_entry)] = 1


class Dataset(object):
    def __init__(self):
        pass

    def __repr__(self):
        pass
