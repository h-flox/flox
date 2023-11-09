import matplotlib.pyplot as plt
import numpy as np
import warnings

from matplotlib.axes import Axes
from collections import defaultdict
from scipy import stats
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Optional

from flox.flock import Flock
from flox.data import FederatedSubsets


# TODO: Implement something similar for regression-based data.
def federated_split(
    data: Dataset,
    flock: Flock,
    num_classes: int,
    samples_alpha: float = 1.0,
    labels_alpha: float = 1.0,
) -> FederatedSubsets:
    r"""
    Splits up Datasets across worker nodes in a Flock using Dirichlet distributions for IID and non-IID settings.

    It is recommended to use an alpha value of 1.0 for either `samples_alpha` want non-IID number of samples across
    workers. Setting this alpha value to be < 1 will result in extreme cases where many workers will have 0 data
    samples.

    Notes:
        Currently, this function only works with data for classification tasks with a discrete number
        of labels/classes. Do *not* use this function for regression-based data.

    Args:
        data (Dataset): The original centralized data object that needs to be split into subsets.
        flock (Flock): The network to split data across.
        num_classes (int): Number of classes available in ``data``.
        samples_alpha (float): The $\alpha>0$ parameter under the Dirichlet distribution for *the number of
            data samples* each worker node in ``flock`` will have. The number of data samples across all worker
            nodes become increasingly heterogeneous as $\alpha$ gets larger.
        labels_alpha (float): The $\alpha>0$ parameter under the Dirichlet distribution for the class distributions
            across worker nodes in ``flock``. The number of data samples across all worker nodes become increasingly
            heterogeneous as $\alpha$ gets larger.

    Examples:
        >>> from torchvision.datasets import MNIST
        >>> flock = Flock.from_yaml("my_flock.yml")
        >>> data = MNIST()
        >>> fed_data = federated_split(data, flock, num_classes=10, samples_alpha=1., labels_alpha=1.)
        >>> next(iter(fed_data.items()))
        >>> # (FlockNodeID(1), Subset(...)) # TODO: Run a real example and paste it here.

    Returns:
        A federated version of the dataset that is split up statistically based on the arguments alpha arguments.
    """
    assert samples_alpha > 0
    assert labels_alpha > 0

    num_workers = len(list(flock.workers))
    sample_distr = stats.dirichlet(np.full(num_workers, samples_alpha))
    label_distr = stats.dirichlet(np.full(num_classes, labels_alpha))

    num_samples_for_workers = (sample_distr.rvs()[0] * len(data)).astype(int)
    num_samples_for_workers = {
        worker.idx: num_samples
        for worker, num_samples in zip(flock.workers, num_samples_for_workers)
    }
    label_probs = {w.idx: label_distr.rvs()[0] for w in flock.workers}

    indices: dict[int, list[int]] = defaultdict(list)
    loader = DataLoader(data, batch_size=1)
    worker_samples = defaultdict(int)
    for idx, batch in enumerate(loader):
        _, y = batch
        label = y.item()

        probs = []
        temp_workers = []
        for w in flock.workers:
            if worker_samples[w.idx] < num_samples_for_workers[w.idx]:
                try:
                    probs.append(label_probs[w.idx][label])
                    temp_workers.append(w.idx)
                except IndexError as err:
                    if isinstance(label, float):
                        warnings.warn(
                            "Label cannot be of type `float` (must be an `int`). Perhaps, use "
                            "`y.to(torch.int32)` in your `Dataset` object definition to resolve this issue.",
                            category=RuntimeWarning,
                        )
                    raise err

        probs = np.array(probs)
        probs = probs / probs.sum()

        if len(temp_workers) > 0:
            chosen_worker = np.random.choice(temp_workers, p=probs)
            indices[chosen_worker].append(idx)
            worker_samples[chosen_worker] += 1

    mapping = {w.idx: Subset(data, indices[w.idx]) for w in flock.workers}
    return FederatedSubsets(mapping)


def fed_barplot(
    fed_data: FederatedSubsets,
    num_labels: int,
    width: float = 0.5,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plots the label/sample distributions across worker nodes of a ``FederatedSubsets`` as a stacked barplot.

    Args:
        fed_data (FederatedSubsets): The federated data cross a ``Flock``.
        num_labels (int): The total number of unique labels in ``fed_data``.
        width (float): The width of the bars.
        ax (Axes): The ``axes`` to draw onto, if provided. If one is not provided, a new one is created.

    Returns:
        The ``Axes`` object that was drawn onto.
    """
    if not isinstance(fed_data, dict):  # TODO: Make `FederatedSubsets` compatible here.
        raise ValueError(
            f"This function (`fed_barplot`) does not support data of type "
            f"``{type(fed_data).__name__}``. `fed_data` argument MUST be of "
            f"type ``FederatedSubsets``."
        )

    label_counts_per_worker = {
        label: np.zeros(len(fed_data), dtype=np.int32) for label in range(num_labels)
    }

    for idx, (worker, subset) in enumerate(fed_data.items()):
        loader = DataLoader(subset, batch_size=1)
        for batch in loader:
            _, y = batch
            label = y.item()
            label_counts_per_worker[label][idx] += 1

    if ax is None:
        fig, ax = plt.subplots()

    bottom = np.zeros(len(fed_data))
    workers = list(range(len(fed_data)))
    for label, worker_count in label_counts_per_worker.items():
        ax.bar(workers, worker_count, width, label=label, bottom=bottom)
        bottom += worker_count

    return ax
