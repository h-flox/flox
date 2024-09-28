import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from flight.federation.topologies import Node
from flight.learning.scikit import ScikitDataModule, ScikitModule, ScikitTrainer

SEED = 42


class DataModule(ScikitDataModule):
    def __init__(self, n: int = 10_000, f: int = 20):
        self.n = n
        self.f = f
        inputs, labels = make_classification(
            n_samples=self.n, n_features=self.f, random_state=SEED
        )

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            inputs, labels, test_size=0.3
        )

    def train_data(self, node: Node | None = None):
        return self.x_train, self.y_train

    def test_data(self, node: Node | None = None):
        return self.x_test, self.y_test

    def valid_data(self, node: Node | None = None):
        return self.x_test, self.y_test

    def size(self, node: Node | None = None, kind="train"):
        return len(self.x_train)


def main():
    # topo = flat_topology(10)
    # federation = SyncFederation(
    #     topology=topo,
    #     strategy=FedAvg(),
    #     module=ScikitTrainable(MLPClassifier()),
    #     data=DataModule(),
    # )
    # model, results = federation.start(3)

    module = ScikitModule(MLPClassifier())
    trainer = ScikitTrainer(max_epochs=5, partial=True)
    records = trainer.fit(module, data=DataModule())

    df = pd.DataFrame.from_records(records)
    sns.lineplot(df, x="train/step", y="train/loss")
    plt.show()

    # results.to_feather("scikit-tmp.feather")


if __name__ == "__main__":
    main()
