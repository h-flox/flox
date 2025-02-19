import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from v1 import flight as fl
from v1.flight import TensorLoss, TorchModule, federated_split

NUM_LABELS = 10


class MyModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 28 * 28 * 3),
            nn.ReLU(),
            nn.Linear(28 * 28 * 3, 28 * 28),
            nn.ReLU(),
            nn.Linear(28 * 28, 28),
            nn.ReLU(),
            nn.Linear(28, NUM_LABELS),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> TensorLoss:
        x, y = batch
        y_hat = self(x)
        return nn.functional.cross_entropy(y_hat, y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.02)


def main():
    data = MNIST(
        root="~/Research/Data/Torch-Data/",
        download=False,
        train=False,
        transform=ToTensor(),
    )
    data = Subset(data, indices=list(range(200)))
    topo = fl.flat_topology(10)
    module = MyModule()
    fed_data = federated_split(
        topo=topo,
        data=data,
        num_labels=NUM_LABELS,
        label_alpha=100.0,
        sample_alpha=100.0,
    )
    trained_module, records = fl.federated_fit(
        topo, module, fed_data, strategy="fedavg", rounds=10
    )

    df = pd.DataFrame.from_records(records)
    print(df.head())
    sns.lineplot(
        df,
        x="train/time",
        y="train/loss",
        hue="node/idx",
        # errorbar=None,
    ).set(yscale="linear")
    plt.show()


if __name__ == "__main__":
    main()
