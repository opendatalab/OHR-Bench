import json
import os
import random
from typing import Any

from src.datasets.base import BaseDataset

class OHRBench(BaseDataset):
    def __init__(self, data, shuffle: bool = False, seed: int = 22):
        self.data = data

        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
        return self.data[key]

    def load(self) -> list[dict]:
        return self.data[:]

    def statistics(self) -> dict:
        stat = {'doc': 0, 'gen': 0, 'kno': 0, 'num': 0}
        for type_ in stat.keys():
            stat[type_] = sum([obj['type']==type_ for obj in self.data])
        return stat


def get_task_datasets(path: str, task: str, shuffle: bool = False, seed: int = 22):
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
    
    return [OHRBench(data, shuffle, seed)]
