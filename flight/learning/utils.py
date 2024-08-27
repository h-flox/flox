from __future__ import annotations

import typing as t
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from torch.utils.data import DataLoader, Subset

from ..commons import proportion_split, random_generator
from ..federation import Topology
from ..federation.topologies.node import Node, NodeID, NodeKind
from .base import AbstractDataModule, Data

if t.TYPE_CHECKING:
    FloatDouble: t.TypeAlias = tuple[float, float]
    FloatTriple: t.TypeAlias = tuple[float, float, float]


class FederatedDataModule(AbstractDataModule):
    """

    Notes:
        Currently, only supports PyTorch `Dataset` objects.

    """

    def __init__(
        self,
        dataset: Data,
        train_indices: t.Mapping[NodeID, t.Sequence[int]],
        test_indices: t.Mapping[NodeID, t.Sequence[int]] | None,
        valid_indices: t.Mapping[NodeID, t.Sequence[int]] | None,
        *,
        batch_size: int = 32,
    ) -> None:
        """
        Args:
            dataset:
            train_indices:
            test_indices:
            valid_indices:

        Throws:
            - `NotImplementedError`: If the dataset is not mappable.

        """
        self.dataset = dataset
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.valid_indices = valid_indices

        self.batch_size = batch_size

        try:
            self.dataset[0]
        except IndexError:
            pass
        except NotImplementedError as err:
            err.add_note(
                "Your PyTorch `Dataset` must be mappable "
                "(i.e., implements `__getitem__`)."
            )
            raise err

    def train_data(self, node: Node | NodeID | None = None) -> Data:
        if isinstance(node, Node):
            node_id = node.idx
        elif isinstance(node, NodeID):
            node_id = node
        else:
            try:
                node_id = int(node)
            except Exception as e:
                e.add_note(
                    f"`node` must be an instance of `Node` or `NodeID`, "
                    f"got {type(node)}."
                )

        indices = self.train_indices[node_id]
        subset = Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=self.batch_size)

    def __iter__(self) -> t.Iterator[tuple[NodeID, Data]]:
        """
        Returns an iterator over the workers and their respective datasets.

        Returns:
            An iterator over the workers and their respective datasets.

        """
        for worker_idx in self.train_indices:
            yield worker_idx, self.train_data(worker_idx)

    def __len__(self) -> int:
        """
        Returns the number of workers that have shards of the original dataset.

        Returns:
            Number of workers.
        """
        return len(self.train_indices)


def federated_split(
    topo: Topology,
    data: Data,
    num_labels: int,
    label_alpha: float,
    sample_alpha: float,
    train_test_valid_split: FloatTriple | FloatDouble | None = None,  # TODO
    ensure_at_least_one_sample: bool = True,
    rng: np.random.Generator | int | None = None,
) -> FederatedDataModule:
    """
    Splits a dataset across a federation of workers.

    The splitting of a dataset can be tuned via the `label_alpha` and `sample_alpha`
    arguments to simulate iid and non-iid data distributions across workers in
    a federation.

    Args:
        topo (Topology): The topology of the federation.
        data (Data): The dataset to split across workers.
        num_labels (int): The number of labels/classes in the dataset. This, of course,
            must be at least 1.
        label_alpha (float): The concentration parameter across labels for the
            Dirichlet distribution.
        sample_alpha (float): The concentration parameter across samples for the
            Dirichlet distribution.
        train_test_valid_split (FloatTriple | FloatDouble | None): The split ratio
            for the training, testing, and validation datasets.
        ensure_at_least_one_sample (bool): If `True`, this ensures that each worker
            has at least 1 data sample; `False` if you want no such guarantee. It is
            generally encouraged to ensure at least 1 sample per worker.
        rng (numpy.random.Generator | int | None): Random number generator.

    Returns:
        A federated data module that where the originally provided dataset is now split
        across the workers following a Dirichlet distribution along classes and samples.

    Examples:
        >>> ...

    Notes:
        This function assumes that the data is for classification-based tasks
        (i.e., the data is trying to predict for a discrete set of classes/labels).

    Throws:
        - `ValueError`: Is thrown in the case either of the `label_alpha` or
            `sample_alpha` args are $\\leq 0$. This can also be thrown in the case that
            `num_labels` is $< 1$.
        - `ValueError`: This can also be thrown in the case that NaN values occur. This
            can occur depending on the values of `label_alpha` and `sample_alpha`.
        - `NotImplementedError`: Is thrown if the provided dataset does not have a
            length given by `__len__()`.
    """
    if label_alpha <= 0 or sample_alpha <= 0:
        raise ValueError(
            "Both `label_alpha` and `sample_alpha` must be greater than 0."
        )
    if num_labels < 1:
        raise ValueError("The number of labels must be at least 1.")

    try:
        train_data_len = len(data)
    except NotImplementedError as err:
        err.add_note("The provided dataset must have `__len__()` implemented.")
        raise err

    rng: np.random.Generator = random_generator(rng)

    num_workers = topo.number_of_nodes(NodeKind.WORKER)
    sample_distr = rng.dirichlet(np.full(num_workers, sample_alpha))
    label_distr = rng.dirichlet(np.full(num_labels, label_alpha), size=num_workers)
    num_samples = (sample_distr * train_data_len).astype(int)

    label_probs_per_worker = {}
    samples_per_worker = {}
    for worker, label_prob, samples in zip(topo.workers, label_distr, num_samples):
        label_probs_per_worker[worker.idx] = label_prob
        samples_per_worker[worker.idx] = samples

    indices: dict[NodeID, list[int]] = defaultdict(list)  # indices on each worker
    worker_samples: dict[NodeID, int] = defaultdict(int)  # num. samples on each worker

    for idx, batch in enumerate(DataLoader(data, batch_size=1)):
        _, label = batch
        label = label.item()

        probs, temp_workers = [], []
        for w in topo.workers:
            if worker_samples[w.idx] < samples_per_worker[w.idx]:
                try:
                    probs.append(label_probs_per_worker[w.idx][label])
                    temp_workers.append(w.idx)
                except IndexError as err:
                    if isinstance(label, float):
                        err.add_note(
                            "Label cannot be of type `float` (must be an `int`). "
                            "Perhaps, use `y.to(torch.int32)` in your `Dataset` object "
                            "definition to resolve this issue."
                        )
                    raise err

        probs_norm = np.array(probs)
        probs_norm = probs_norm / probs_norm.sum()

        if len(temp_workers) > 0:
            chosen_worker = rng.choice(temp_workers, p=probs_norm)
            indices[chosen_worker].append(idx)
            worker_samples[chosen_worker] += 1

    if train_test_valid_split is None:
        train_indices = indices
        test_indices = None
        valid_indices = None

    elif len(train_test_valid_split) == 2:
        train_indices, test_indices = ({} for _ in range(2))
        valid_indices = None
        for w_idx, w_indices in indices.items():
            train_split, test_split = proportion_split(
                w_indices, train_test_valid_split
            )
            train_indices[w_idx] = train_split
            test_indices[w_idx] = test_split

    elif len(train_test_valid_split) == 3:
        train_indices, test_indices, valid_indices = ({} for _ in range(3))
        for w_idx, w_indices in indices.items():
            train_split, test_split, valid_split = proportion_split(
                w_indices, train_test_valid_split
            )
            train_indices[w_idx] = train_split
            test_indices[w_idx] = test_split
            valid_indices[w_idx] = valid_split

    else:
        raise ValueError("Invalid number of elements in `train_test_valid_split`.")

    if ensure_at_least_one_sample:
        for worker in topo.workers:
            worker_with_most_samples = max(worker_samples, key=worker_samples.get)
            if worker_samples[worker.idx] == 0:
                index = indices[worker_with_most_samples].pop()
                worker_samples[worker_with_most_samples] -= 1

                indices[worker.idx].append(index)
                worker_samples[worker.idx] += 1

    return FederatedDataModule(
        data,
        train_indices=train_indices,
        test_indices=test_indices,
        valid_indices=valid_indices,
    )


def fed_barplot(
    fed_data: FederatedDataModule,
    num_labels: int,
    width: float = 0.5,
    *,
    ax: Axes | None = None,
    edgecolor: str | None = "black",
):
    """

    Args:
        fed_data:
        num_labels:
        width:
        ax:
        edgecolor:

    Returns:

    """

    if not isinstance(fed_data, FederatedDataModule):
        raise TypeError(
            f"`fed_data` must be an instance of `FederatedDataModule` "
            f"(got `{type(fed_data)}`)."
        )

    label_counts_per_worker = {
        label: np.zeros(len(fed_data), dtype=np.int32) for label in range(num_labels)
    }

    for i, (_, worker_loader) in enumerate(fed_data):
        for batch in worker_loader:
            _, labels = batch
            for label in labels:
                label_counts_per_worker[label.item()][i] += 1

    if ax is None:
        fig, ax = plt.subplots()

    bottom = np.zeros(len(fed_data))
    workers = list(range(len(fed_data)))
    for label, worker_count in label_counts_per_worker.items():
        ax.bar(
            workers,
            worker_count,
            width,
            bottom=bottom,
            label=label,
            edgecolor=edgecolor,
        )
        bottom += worker_count

    ax.set_xlabel("workers")
    ax.set_ylabel("label counts")

    return ax
