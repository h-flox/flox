import matplotlib.pyplot as plt
import os

from flox.flock import Flock
from flox.utils.data import fed_barplot, federated_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

if __name__ == "__main__":
    flock = Flock.from_yaml("examples/flock_files/2-tier.yaml")
    mnist = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=True,
        transform=ToTensor(),
    )

    alphas = [0.1, 1, 10, 1_000]
    # alphas = [1]
    fig, axes = plt.subplots(nrows=len(alphas), ncols=len(alphas))
    for i, samples_alpha in enumerate(alphas):
        for j, labels_alpha in enumerate(alphas):
            print(f">>> {samples_alpha=}, {labels_alpha=}")
            try:
                ax = axes[i, j]
            except TypeError:
                ax = axes

            fed_data = federated_split(
                mnist, flock, 10, samples_alpha=samples_alpha, labels_alpha=labels_alpha
            )
            assert len(fed_data) == len(list(flock.workers))

            fed_barplot(fed_data, 10, ax=ax)
            ax.set_title(
                "$\\alpha_{s} ="
                f"{samples_alpha}$, "
                "$\\alpha_{\\ell} ="
                f"{labels_alpha}$"
            )

    plt.tight_layout()
    plt.show()
