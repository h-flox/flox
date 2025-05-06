import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import typing as t

from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from flight.utils.fed_data import fed_barplot, federated_split
from flight.system.utils import flat_topology


DATA_DIR: t.Final[Path] = Path("~/Research/Data/Torch-Data/").expanduser()


def job(topo, data, label_alpha, sample_alpha, ax):
    fed_data = federated_split(
        topo,
        data,
        num_labels=10,
        label_alpha=label_alpha,
        sample_alpha=sample_alpha,
        rng=1,
    )

    fed_barplot(fed_data, num_labels=10, ax=ax)


if __name__ == "__main__":
    ALPHAS: t.Final[list[float]] = [0.1, 1.0, 10.0, 100.0, 1000.0]
    N: t.Final[int] = len(ALPHAS)

    fig, axes = plt.subplots(nrows=N, ncols=N)
    exc = ThreadPoolExecutor(max_workers=N * N)
    topo = flat_topology(10)
    data = MNIST(
        root=DATA_DIR,
        train=False,
        transform=ToTensor(),
        download=False,
    )

    futures = []
    pbar = tqdm.tqdm(total=N*N)

    for i, sample_alpha in enumerate(ALPHAS):
        for j, label_alpha in enumerate(ALPHAS):
            # fut = exc.submit(
            #     job,
            #     topo,
            #     data,
            #     label_alpha,
            #     sample_alpha,
            #     axes[i, j],
            # )
            # futures.append(fut)

            fed_data = federated_split(
                topo,
                data,
                num_labels=10,
                label_alpha=label_alpha,
                sample_alpha=sample_alpha,
                rng=1,
            )

            fed_barplot(fed_data, num_labels=10, ax=axes[i, j])
            pbar.update()
            axes[i, j].set_xlabel("")
            axes[i, j].set_ylabel("")
            axes[i, j].set_title(
                "$\\alpha_s = {}, \\alpha_l = {}$".format(
                    sample_alpha, label_alpha
                )
            )

    for fut in as_completed(futures):
        pbar.update()

    wait(futures)
    exc.shutdown()

    # plt.tight_layout()
    plt.show()
