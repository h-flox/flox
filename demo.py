import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import flight as fl
from flight.learning import federated_split
from flight.learning.torch import TorchModule
from flight.learning.torch.types import TensorLoss

NUM_LABELS = 10


class TrainingModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.Linear(10, 100),
            nn.Linear(100, NUM_LABELS),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch) -> TensorLoss:
        x, y = batch
        y_hat = self(x)
        return nn.functional.nll_loss(y_hat, y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.02)


def main():
    topo = fl.flat_topology(10)
    module = TrainingModule()
    fed_data = federated_split(
        topo=topo,
        data=TensorDataset(
            torch.randn(100, 1), torch.randint(low=0, high=NUM_LABELS, size=(100, 1))
        ),
        num_labels=NUM_LABELS,
        label_alpha=100.0,
        sample_alpha=100.0,
    )
    trained_module, records = fl.federated_fit(topo, module, fed_data, rounds=2)
    print(records)


if __name__ == "__main__":
    main()
