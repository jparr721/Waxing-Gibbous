import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm


@dataclass(frozen=True)
class InputOutputGroup(object):
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Dataset(object):
    # Input
    x: np.ndarray

    # Targets
    y: np.ndarray


def load_model_result(timestep_folder_path: str):
    files = list(os.listdir(timestep_folder_path))
    x = np.load(os.path.join(timestep_folder_path, files[files.index("igs.npy")]))
    y = np.load(os.path.join(timestep_folder_path, files[files.index("gbc.npy")]))

    xv = x[:, :, 0]
    yv = x[:, :, 1]
    mask = (xv != 0) | (yv != 0)

    x = np.stack((xv, yv, mask))

    xv = y[:, :, 0]
    yv = y[:, :, 1]
    mask = (xv != 0) | (yv != 0)

    y = np.stack((xv, yv, mask))

    return InputOutputGroup(x, y)


def load_model_results(folder_path: str) -> List[InputOutputGroup]:
    timesteps = list(os.listdir(folder_path))
    groups_at_timestep = [None] * len(timesteps)
    for timestep in tqdm(timesteps):
        fullpath = os.path.join(folder_path, timestep)
        try:
            groups_at_timestep[int(timestep)] = load_model_result(fullpath)
        except Exception as e:
            raise e

    return groups_at_timestep


def load_pickle_files(
    folders: List[str], mem_limit: int = 6
) -> Tuple[Dict[str, List[InputOutputGroup]], List[str]]:
    """This loads the saved pickle files. Any unloaded files get marked as "leftovers",
    that way the system utilizing this process can know which files are next up to be
    loaded for continued training

    Args:
        folders (List[str]): The list of folder names (full path)
        mem_limit (int, optional): The max amount of memory used (in GB). Defaults to 6.

    Returns:
        Tuple[Dict[str, List[InputOutputGroup]], List[str]]: The loaded datasets and
        leftovers
    """

    def is_memory_at_limit(arr_arrs: List[List[InputOutputGroup]]) -> Tuple[bool, float]:
        """Checks whether the loaded files reach the memory limit imposed by the system

        Args:
            arr_arrs (List[List[InputOutputGroup]]): The list of arrays of inputs

        Returns:
            Tuple[bool, float]: Whether or not the memory limit is reached. The the
            memory currently in use.
        """
        total_memory = 0
        for arrs in arr_arrs:
            for arr in arrs:
                total_memory += arr.x.nbytes + arr.y.nbytes
        total_memory *= 1e-9
        return total_memory > mem_limit, total_memory

    datasets = {}
    leftovers = []
    at_memory_limit = False
    total_memory = 0
    for folder in tqdm(folders):
        # We want just the folder name, sans the "pkl" ending.
        n, _ = os.path.basename(folder).split(".")
        at_memory_limit, total_memory = is_memory_at_limit(list(datasets.values()))

        if not at_memory_limit:
            # Dataset at this name is assigned the pickle file of data.
            datasets[n] = pickle.load(open(folder, "rb"))
        else:
            # Once we hit memory limit, finish iterating and adding the leftover names.
            leftovers.append(folder)

    logger.info(f"Using {total_memory}gb of memory.")
    return datasets, leftovers


def load_datasets(datasets_path: str) -> List[str]:
    """Loads the datasets from the given absolute path to the folder. This also loads the
    pickle files at the same time.

    Args:
        datasets_path (str): The absolute path to the "datasets" folder or whatever it's
        named.

    Returns:
        List[str]: The list of file paths in this folder to pkl files.
    """
    folders = [
        os.path.join(datasets_path, folder) for folder in os.listdir(datasets_path)
    ]

    folders = list(
        filter(lambda x: "jelly" in x or "snow" in x or "liquid" in x, folders)
    )

    files = []
    for folder in tqdm(folders):
        fn = f"{folder}.pkl"
        with open(fn, "wb+") as pf:
            files.append(fn)
            pickle.dump(load_model_results(folder), pf)

    return files
