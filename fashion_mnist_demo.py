import os
from pathlib import Path

import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from flox.data.utils import federated_split
from flox.flock import Flock
from flox.nn import FloxModule
from flox.runtime import federated_fit


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
    flock = Flock.from_yaml("examples/flocks/complex.yaml")
    # flock = Flock.from_yaml("../examples/flocks/gce-complex-sample.yaml")
    mnist = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=False,
        transform=ToTensor(),
    )
    fed_data = federated_split(mnist, flock, 10, 1.0, 1.0)
    assert len(fed_data) == len(list(flock.workers))

    module, df = federated_fit(
        flock,
        MyModule(),
        fed_data,
        5,
        strategy="fedavg",
        launcher_cfg={"max_workers": 1},
        # where="local",  # "globus_compute",
    )
    df.to_feather(Path("out/fashion_mnist_demo.feather"))


if __name__ == "__main__":
    main()
