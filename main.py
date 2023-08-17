import argparse
import os
import torch

from depr.run import fit
from pandas import DataFrame
from pathlib import Path
from torch import nn
from torch.utils.data import random_split
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


def main(args: argparse.Namespace):
    root = os.environ["TORCH_DATASETS"]
    train_femnist = FashionMNIST(
        root=root, train=True, download=True, transform=ToTensor()
    )
    test_femnist = FashionMNIST(
        root=root, train=False, download=True, transform=ToTensor()
    )

    flock = {}
    lengths = [1 / args.num_workers] * args.num_workers
    for worker_id, subset in enumerate(random_split(train_femnist, lengths)):
        flock[worker_id] = subset

    module, train_hist, test_hist = fit(
        flock, MyModule, 10, test_dataset=test_femnist, device=torch.device("mps")
    )

    train_df = DataFrame.from_dict(train_hist)
    test_df = DataFrame.from_dict(test_hist)

    train_df.to_csv(Path("out/data/train_history (SimpleAvg).csv"))
    test_df.to_csv(Path("out/data/test_history (SimpleAvg).csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_workers", type=int, default=10)
    main(parser.parse_args())
