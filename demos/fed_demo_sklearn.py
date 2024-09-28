import os
import sys

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

sys.path.append(os.getcwd())

import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from flight.federation.topologies import Node
from flight.fit import federated_fit
from flight.learning.scikit import ScikitDataModule, ScikitModule

SEED = 42


class DataModule(ScikitDataModule):
    def __init__(self, n: int = 100, f: int = 20):
        self.n = n
        self.f = f

    def _data(self, idx):
        inputs, labels = make_classification(
            n_samples=self.n, n_features=self.f, random_state=idx
        )

        x_train, x_test, y_train, y_test = train_test_split(
            inputs, labels, test_size=0.3
        )

        return x_train, x_test, y_train, y_test

    def train_data(self, node: Node | None = None):
        assert node is not None
        x, _, y, _ = self._data(node.idx)
        return x, y

    def test_data(self, node: Node | None = None):
        assert node is not None
        _, x, _, y = self._data(node.idx)
        return x, y

    def valid_data(self, node: Node | None = None):
        assert node is not None
        _, x, _, y = self._data(node.idx)
        return x, y

    def size(self, node: Node | None = None, kind="train"):
        if kind == "train":
            return self.n
        else:
            return int(self.n * 0.3)


def main():
    trained_module, results = federated_fit(
        "demos/topo.yaml",
        ScikitModule(MLPClassifier()),
        DataModule(),
        rounds=10,
    )

    records = []
    for res in results:
        records.extend(res.records)

    df = pd.DataFrame.from_records(records)
    sns.lineplot(df, x="round", y="train/loss")
    plt.show()


if __name__ == "__main__":
    main()
