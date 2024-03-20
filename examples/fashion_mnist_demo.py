import os
from datetime import timedelta
from time import perf_counter

import pandas as pd
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from flox.data.utils import federated_split
from flox.flock import Flock
from flox.nn import FloxModule
from flox.runtime import federated_fit
from flox.strategies.impl.fedavg import FedAvg
from flox.strategies.impl.fedprox import FedProx
from flox.strategies.impl.fedsgd import FedSGD


class MyModule(FloxModule):
    def __init__(self, lr: float = 0.01):
        super().__init__()
        self.lr = lr
        self.flatten = torch.nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_stack(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self(inputs)
        loss = torch.nn.functional.cross_entropy(preds, targets)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.lr)


def run_trial(data, flock, num_labels, labels_alpha: float, samples_alpha: float):
    p = 0.5
    strategies = {
        "fedsgd": FedSGD(participation=p),
        "fedavg": FedAvg(participation=p),
        "fedprox": FedProx(participation=p, mu=0.3),
    }
    fed_data = federated_split(data, flock, num_labels, samples_alpha, labels_alpha)
    assert len(fed_data) == len(
        list(flock.workers)
    ), f"{len(fed_data)=} != {len(list(flock.workers))=}"

    histories = []
    for strategy_name, strategy in strategies.items():
        print(f">> Running experiment for {strategy_name.upper()}.")
        module, hist = federated_fit(
            flock,
            MyModule(),
            fed_data,
            15,
            strategy=strategy,
            # launcher_cfg={"max_workers": 1},
            launcher_cfg={"max_workers": flock.number_of_workers},
            # where="local",  # "globus_compute",
        )
        hist["strategy"] = strategy_name
        histories.append(hist)

    return pd.concat(histories).reset_index()


def main():
    start_time = perf_counter()
    flock = Flock.from_yaml("examples/flocks/complex.yaml")
    mnist = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=True,
        transform=ToTensor(),
    )

    histories = []
    domain = [1.0]
    for labels_alpha in domain:
        for samples_alpha in domain:
            df = run_trial(mnist, flock, 10, labels_alpha, samples_alpha)
            df["labels_alpha"] = labels_alpha
            df["samples_alpha"] = samples_alpha
            histories.append(df)

    df = pd.concat(histories).reset_index()
    df.to_feather("out/fmnist_test.feather")
    end_time = perf_counter()
    time_taken = timedelta(seconds=end_time - start_time)
    print(f"Done! Time take: {time_taken}")


if __name__ == "__main__":
    main()
