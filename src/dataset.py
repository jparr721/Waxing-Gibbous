import os

import matplotlib.pyplot as plt
import numpy as np


def binarize_grid(grid_entry: np.ndarray):
    grid_entry[np.nonzero(grid_entry)] = 1
    # if not os.path.exists("out_1.png"):
    #     plt.imsave("out_1.png", grid_entry[:, :, 0])
    #     plt.imsave("out_2.png", grid_entry[:, :, 1])
    # else:
    #     plt.imsave("out_3.png", grid_entry[:, :, 0])
    #     plt.imsave("out_4.png", grid_entry[:, :, 1])


class Dataset(object):
    def __init__(self):
        pass

    def __repr__(self):
        pass
