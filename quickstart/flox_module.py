import os
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from flox.nn import FloxModule, Trainer


class MyModule(FloxModule):
    def __init__(self):
        super().__init__()
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
        return torch.optim.SGD(self.parameters(), lr=0.01)


if __name__ == "__main__":
    model = MyModule()
    mnist = MNIST(
        root=os.environ["TORCH_DATASETS"],
        train=True,
        download=False,
        transform=ToTensor(),
    )

    history = Trainer().fit(model, DataLoader(mnist, batch_size=32), num_epochs=10)

    print("Finished training!")
    print(history.head())
    history.to_feather("trainer_demo.feather")
