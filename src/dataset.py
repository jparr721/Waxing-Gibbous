import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class Dataset(object):
    def __init__(self):
        self.input_dataset = []
        self.output_dataset = []

    def load_from_folder(self, inp: Union[str, Path], outp: Union[str, Path]):
        pass
