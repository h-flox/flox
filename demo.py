# import flox.learn.prototype as flox_learn
import os

from flox.flock import Flock
from flox.learn import federated_fit
from flox.utils.data.federate import randomly_federate_dataset
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
    flock = Flock.from_yaml("examples/flock_files/complex.yaml")

    mnist = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=True,
        transform=ToTensor(),
    )
    fed_data = randomly_federate_dataset(
        flock,
        mnist,
        shuffle=True,
        random_state=None,
    )

    train_history = federated_fit(flock, MyModule, fed_data, 5)
    train_history.head()


if __name__ == "__main__":
    main()
