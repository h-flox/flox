# This is done to silence an annoying `UserWarning` being thrown by pydantic from `funcx_common` which assumes
# you are using pydantic V1.
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
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
    flock = Flock.from_yaml("examples/flocks/2-tier.yaml")
    # flock = Flock.from_yaml("../examples/flocks/gce-complex-sample.yaml")
    mnist = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=False,
        transform=ToTensor(),
    )
    fed_data = federated_split(mnist, flock, 10, 100.0, 1.0)
    assert len(fed_data) == len(list(flock.workers))

    for kind in ["async", "sync"]:
        print(f">>> Running {kind.upper()} FLoX.")
        _, df = federated_fit(
            flock,
            MyModule(),
            fed_data,
            5,
            strategy="fedsgd",
            kind=kind,
            # where="local",  # "globus_compute",
        )
        df.to_feather(Path(f"out/{kind}_comparison.feather"))
    print(">>> Finished!")


if __name__ == "__main__":
    main()
