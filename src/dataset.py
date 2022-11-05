from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix


def binarize_grid(grid_entry: Union[np.ndarray, csc_matrix]):
    grid_entry = grid_entry.todense()
    grid_entry[grid_entry.nonzero()] = 1
    plt.imsave("grid_entry_dbg.png", grid_entry)


class Dataset(object):
    def __init__(self):
        pass

    def __repr__(self):
        pass
