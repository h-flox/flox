import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import FashionMNIST

import flox
from flox.data.utils import federated_split
from flox.flock.factory import create_standard_flock
from flox.nn import FloxModule


class Net(FloxModule):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def configure_optimizers(self):
        pass  # NOTE (nathaniel-hudson): Not needed for debugging, just here to get around ABC requirements.

    def training_step(self, batch, batch_idx):
        pass  # NOTE (nathaniel-hudson): Not needed for debugging, just here to get around ABC requirements.


def main(args: argparse.Namespace):
    flock = create_standard_flock(num_workers=args.workers_nodes)
    root_dir = Path(args.root_dir)
    if "~" in str(root_dir):
        root_dir = root_dir.expanduser()
    data = FashionMNIST(
        root=str(root_dir),
        download=False,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        ),
    )
    fed_data = federated_split(
        data,
        flock,
        num_classes=10,
        labels_alpha=args.labels_alpha,
        samples_alpha=args.samples_alpha,
    )
    flox.federated_fit(
        flock=flock,
        module=Net(),  # nathaniel-hudson: this uses a reasonable model.
        # module=None, # nathaniel-hudson: this uses a VERY small debug model.
        datasets=fed_data,
        num_global_rounds=args.rounds,
        strategy="fedsgd",
        kind="sync",
        debug_mode=True,
        launcher_kind=args.executor,
        launcher_cfg=dict(
            label="Expanse_CPU_Multinode",
        ),
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--executor",
        "-e",
        type=str,
        choices=["process", "thread", "parsl", "globus-compute"],
        default="parsl",
    )
    args.add_argument("--max_workers", "-w", type=int, default=1)
    args.add_argument("--workers_nodes", "-n", type=int, default=32)
    args.add_argument("--samples_alpha", "-s", type=float, default=1000.0)
    args.add_argument("--labels_alpha", "-l", type=float, default=1000.0)
    args.add_argument("--rounds", "-r", type=int, default=1)
    args.add_argument("--root_dir", "-d", type=str, default=".")
    parsed_args = args.parse_args()
    assert parsed_args.samples_alpha > 0.0
    assert parsed_args.labels_alpha > 0.0
    main(parsed_args)
