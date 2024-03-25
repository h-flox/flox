import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F

from flox.nn import FloxModule

DEFAULT_LR = 0.01


class SmallModel(FloxModule):
    def __init__(self, lr: float = DEFAULT_LR, device: str | None = None):
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

        # if device is not None:
        #     self.device = torch.device(device)
        # elif torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        # else:
        #     self.device = torch.device("cpu")

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.last_accuracy = None

    def forward(self, x):
        # x = x.to(self.device)
        # x = x.to(DEVICE)
        x = self.flatten(x)
        return self.linear_stack(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.to("cpu")
        targets = targets.to("cpu")
        preds = self(inputs)
        loss = F.cross_entropy(preds, targets)

        self.last_accuracy = self.accuracy(preds, targets)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.lr)


class SmallConvModel(FloxModule):
    def __init__(self, lr: float = DEFAULT_LR, device: str | None = None):
        super().__init__()
        self.lr = lr
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # if device is not None:
        #     self.device = torch.device(device)
        # elif torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        # else:
        #     self.device = torch.device("cpu")

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.last_accuracy = None

    def forward(self, x):
        # x = x.to(self.device))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(f"{x.shape=} (after flatten)")
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        # print(f"{inputs.shape=}")
        inputs = inputs.to("cpu")
        targets = targets.to("cpu")
        preds = self.forward(inputs)
        loss = F.cross_entropy(preds, targets)

        self.last_accuracy = self.accuracy(preds, targets)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.lr)
