from typing import Tuple

import numpy as np


def cube(center: Tuple[float, float], res=10) -> np.ndarray:
    return np.random.uniform(*center, (res, 2))
