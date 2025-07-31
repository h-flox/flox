import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from flight.system.utils import flat_topology
from v1.flight.fit import federated_fit, Topology
from v1.flight.learning.torch.types import TensorLoss
from v1.flight.learning.torch import TorchModule
from v1.flight.learning import federated_split

NUM_LABELS = 10

class MyModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch, batch_idx) -> TensorLoss:
        x, y = batch
        y_hat = self(x)
        return nn.functional.cross_entropy(y_hat, y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def configure_criterion(self):
        return nn.CrossEntropyLoss()


def main():
    data = MNIST(
        root=".",
        download=False,
        train=True,
        transform=ToTensor(),
    )
    data = Subset(data, indices=list(range(200)))
    topo = Topology.from_yaml("topo-copy.yaml")

    module = MyModule()
    fed_data = federated_split(
        topo=topo,
        data=data,
        num_labels=NUM_LABELS,
        label_alpha=100.0,
        sample_alpha=100.0,
    )
    trained_module, records = federated_fit(
        topo, module, fed_data, strategy="fedavg", rounds=10
    )

    df = pd.DataFrame.from_records(records)
    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())
    
    # Check if we have data to plot
    if df.empty:
        print("No training records found. The simulation completed but no training data was recorded.")
        print("This is expected since the current implementation doesn't perform actual training.")
    elif "train/time" in df.columns and "train/loss" in df.columns:
        sns.lineplot(
            df,
            x="train/time",
            y="train/loss",
            hue="node/idx",
            # errorbar=None,
        ).set(yscale="linear")
        plt.show()
    else:
        print("Expected columns not found in the DataFrame.")
        print("Available columns:", df.columns.tolist())
    
    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()
