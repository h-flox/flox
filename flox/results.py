from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Result:
    pass


class ResultLogger:
    def __init__(self):
        self._data = defaultdict(list)

    def log(self, name, value):
        self._data[name].append(value)

    def dict_log(self, dictionary):
        for name, value in dictionary.items():
            self.log(name, value)
