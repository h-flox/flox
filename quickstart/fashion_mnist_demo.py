import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

try:
    sys.path.append("..")
    from flox.data.utils import federated_split
    from flox.federation.topologies import Topology
    from flox.learn import FloxModule
    from flox.run import federated_fit
    from flox.strategies_depr import FedProx
except Exception as e:
    raise ImportError("unable to import FloX libraries") from e


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


def main():
    flock = Topology.from_yaml("../examples/flocks/complex.yaml")
    # topologies = Flock.from_yaml("../examples/flocks/gce-complex-sample.yaml")
    mnist = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=True,
        transform=ToTensor(),
    )
    fed_data = federated_split(mnist, flock, 10, 1.0, 1.0)
    assert len(fed_data) == len(list(flock.workers))

    df_list = []
    strategies = {
        "fed-prox": FedProx,
        # "fed-avg": FedAvg,
        # "fed-sgd": FedSGD,
    }
    for strategy_label, strategy_cls in strategies.items():
        print(f">>> Running FLoX with strategy={strategy_label}.")
        _, df = federated_fit(
            flock,
            MyModule(),
            fed_data,
            5,
            strategy=strategy_cls(),
            # where="local",  # "globus_compute",
        )
        df["strategy"] = strategy_label
        df_list.append(df)

    train_history = pd.concat(df_list).reset_index(drop=True)
    train_history.to_feather(Path("out/demo_history.feather"))
    print(">>> Finished!")


if __name__ == "__main__":
    main()
