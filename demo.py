import os
import pandas as pd

from flox.flock import Flock
from flox.learn import federated_fit
from flox.strategies import FedSGD, FedAvg, FedProx
from flox.utils.data.beta import randomly_federate_dataset
from flox.utils.data import federated_split
from pathlib import Path
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


def main():
    flock = Flock.from_yaml("examples/flock_files/2-tier.yaml")

    mnist = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=True,
        transform=ToTensor(),
    )
    # fed_data = randomly_federate_dataset(
    #     flock,
    #     mnist,
    #     shuffle=True,
    #     random_state=None,
    # )
    fed_data = federated_split(mnist, flock, 10, 1, 1)
    assert len(fed_data) == len(list(flock.workers))

    df_list = []
    for strategy, strategy_label in zip(
        [FedProx, FedAvg, FedSGD],
        ["fed-prox", "fed-avg", "fed-sgd"],
    ):
        print(f"Running FLoX with strategy={strategy_label}.")
        df = federated_fit(
            flock,
            MyModule,
            fed_data,
            5,
            strategy=strategy(),
            where="local",
        )
        df["strategy"] = strategy_label
        df_list.append(df)

    train_history = pd.concat(df_list).reset_index()
    train_history.to_csv(Path("out/demo_history.csv"))
    print("Finished!")


if __name__ == "__main__":
    main()
