from flox.logger import Logger, CSVLogger, TensorBoardLogger
from flox.topos import Topology, Node

# from flox.runtime.process.process_sync_v2 import SyncProcessV2
from flox.runtime import federated_fit
from flox.learn.data import federated_split
#from flox.runtime.launcher import Launcher, LocalLauncher
#from flox.runtime.runtime import Runtime
from flox.learn import FloxModule


from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from datetime import datetime
import os
import torch
from torch import nn
from torch.utils.data import Subset

class MyModule(FloxModule):
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

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self(inputs)
        loss = nn.functional.cross_entropy(preds, targets)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=1e-3)



flock = Topology.from_yaml("./flox/tests/logger/examples/three-level.yaml")


data = MNIST(
    root=os.environ["TORCH_DATASETS"],
    download=True,
    train=True,
    transform=ToTensor(),
)

subset_data = Subset(data, indices=range(len(data) // 1000))

fs = federated_split(subset_data, flock, 10, samples_alpha=10.0, labels_alpha=10.0)

logger = CSVLogger(filename="./flox/tests/logger/logs.csv")
print(fs.dataset)
print(fs.indices[1])

module, train_history = federated_fit(
    flock,
    MyModule(),
    fs,
    1,
    strategy="fedavg",
    launcher_kind="thread",
    logger=logger,
)