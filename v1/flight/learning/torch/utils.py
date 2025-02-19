from __future__ import annotations

import typing as t
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from torch.utils.data import DataLoader, Subset

from v1.flight.commons import proportion_split, random_generator
from v1.flight.topologies.node import Node, NodeID, NodeKind

from .data import TorchDataModule

if t.TYPE_CHECKING:
    from torch.utils.data import Dataset

    from v1.flight.federation import Topology
    from v1.flight.learning.types import Data, FloatDouble, FloatTriple


IID: t.Final[float] = 1e5
NON_IID: t.Final[float] = 1e-5


class FederatedDataModule(TorchDataModule):
    """
    This class defines a DataModule that is split across worker nodes in a federation's
    topology.

    This is especially helpful for simulation-based federations that are run with
    Flight. Rather than needing to manually define the logic to load data that are
    sharded across workers in a federation, this class simply requires the original
    dataset and the indices for training, testing, and validation data for each
    worker.

    A good analogy for this class is to think of it as the federated version of
    PyTorch's [`Subset`](https://pytorch.org/docs/stable/data.html#
    torch.utils.data.Subset) class.

    Notes:
        ==Currently, only supports PyTorch `Dataset` objects.==

    """

    def __init__(
        self,
        dataset: Dataset,
        train_indices: t.Mapping[NodeID, t.Sequence[int]],
        test_indices: t.Mapping[NodeID, t.Sequence[int]] | None,
        valid_indices: t.Mapping[NodeID, t.Sequence[int]] | None,
        **dataloader_kwargs,
    ) -> None:
        """
        Args:
            dataset (Dataset): The dataset to split across workers.
            train_indices (t.Mapping[NodeID, t.Sequence[int]]): The indices for the
                training data for each worker.
            test_indices (t.Mapping[NodeID, t.Sequence[int]] | None): The indices
                for the test data for each worker.
            valid_indices (t.Mapping[NodeID, t.Sequence[int]] | None): The indices
                for the validation data for each worker.
            **dataloader_kwargs: Keyword arguments to pass to the `DataLoader` class
                when calling `train_data()`, `valid_data()`, and `test_data()`.

        Throws:
            - `NotImplementedError`: If the dataset is not mappable.

        """
        self.dataset = dataset
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.valid_indices = valid_indices
        self.dataloader_kwargs = dataloader_kwargs

        try:
            self.dataset[0]
        except IndexError:
            pass
        except NotImplementedError as err:
            err.add_note(
                "Your PyTorch `Dataset` must be mappable (i.e., implements "
                "`__getitem__`)."
            )
            raise err

    def __iter__(self) -> t.Iterator[tuple[NodeID, Dataset]]:
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

    def __contains__(self, node_or_idx: Node | NodeID) -> bool:
        """
        Checks if a (worker) node exists in the federated data module.

        Args:
            node_or_idx (Node | NodeID): The node or node index to check exists in the
                federated data module.

        Returns:
            `True` if the node exists in the federated data module; `False` otherwise.
        """
        if isinstance(node_or_idx, Node):
            node_idx = node_or_idx.idx
        elif isinstance(node_or_idx, NodeID):
            node_idx = node_or_idx
        else:
            raise ValueError(
                f"FedDataModule.__contains__ only accepts arguments of type "
                f"`Node` or `NodeID`; got `{type(node_or_idx)}`."
            )

        return node_idx in self.train_indices

    def train_data(self, node: Node | NodeID | None = None) -> DataLoader:
        return self._get_data(node, self.train_indices)

    def test_data(self, node: Node | NodeID | None = None) -> DataLoader | None:
        if self.test_indices is not None:
            return self._get_data(node, self.test_indices)
        else:
            return None

    # @t.overload
    # def valid_data(self, node: Node) -> DataLoader:
    #     pass
    #
    # @t.overload
    # def valid_data(self, node: None) -> None:
    #     pass

    def valid_data(self, node: Node | None = None) -> DataLoader | None:
        if self.valid_indices is None:
            return None
        return self._get_data(node, self.test_indices)

    def _get_data(
        self, node: Node | NodeID | None, indices: t.Mapping[NodeID, t.Sequence[int]]
    ) -> DataLoader:
        node_id = self._resolve_node(node)
        subset = Subset(self.dataset, indices[node_id])
        return DataLoader(subset, **self.dataloader_kwargs)

    def _resolve_node(self, node_or_idx: Node | NodeID | None = None) -> NodeID:
        if node_or_idx is None:
            raise ValueError(
                f"`node` argument for {self.__class__.__name__} cannot be `None`."
            )
        elif isinstance(node_or_idx, Node):
            node_id = node_or_idx.idx
        elif isinstance(node_or_idx, NodeID):
            node_id = node_or_idx
        else:
            try:
                node_id = int(node_or_idx)
            except Exception as e:
                e.add_note(
                    f"`node` must be an instance of `Node` or `NodeID`, "
                    f"got {type(node_or_idx)}."
                )
                raise e
        return node_id


def federated_split(
    topo: Topology,
    data: Data,
    num_labels: int,
    label_alpha: float = IID,
    sample_alpha: float = IID,
    train_test_valid_split: FloatTriple | FloatDouble | None = None,
    ensure_at_least_one_sample: bool = True,
    rng: np.random.Generator | int | None = None,
    allow_overlapping_samples: bool = False,  # TODO.
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
        allow_overlapping_samples (bool): If `True`, this allows for samples that can
            be shared across workers; `False` if you want no such sharing. Note: this
            is currently not implemented.

    Returns:
        A federated data module that where the originally provided dataset is now split
            across the workers following a Dirichlet distribution along classes and
            samples.

    Examples:
        >>> import torch
        >>> from torch.utils.data import TensorDataset
        >>> from v1.flight import flat_topology
        >>> from v1.flight import federated_split
        >>>
        >>> topo = flat_topology(2)  # flat topology with 2 workers
        >>> data = TensorDataset(
        >>>     torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]),
        >>>     torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        >>> )
        >>> fed_data = federated_split(
        >>>     topo, data, num_labels=2, label_alpha=1.0, sample_alpha=1.0, generator=1
        >>> )
        >>>
        >>> node_id = next(iter(topo.workers))
        >>> for mini_batch in fed_data.train_data(node_id):
        >>>     print(mini_batch)
        [tensor([1., 3., 5., 6., 7., 8., 9.]), tensor([0, 0, 0, 1, 0, 1, 0])]

    Notes:
        - This function assumes that the data is for classification-based tasks
            (i.e., the data is trying to predict for a discrete set of classes/labels).
        - The `sample_alpha` and `label_alpha` parameters must be greater than 0. The
            smaller these values are, the less uniform the distribution of samples or
            labels across workers; the larger these values are, the more uniform. For
            instance, if `sample_alpha=0.1`, then there will be high variance in the
            number of samples across each worker; whereas if `sample_alpha=1000.0`,
            then the number of samples across workers will look very uniform. The
            distribution of labels behaves similarly.
        - It is worth experimenting with this function using the
            [`fed_barplot`][flight.learning.utils.fed_barplot] function
            to visualize the distribution of the federated data module.
        - The `allow_overlapping_samples` feature is ==**not yet implemented**==.

    Throws:
        - `ValueError`: Is thrown in the case either of the `label_alpha` or
            `sample_alpha` args are $\\leq 0$. This can also be thrown in the case that
            `num_labels` is $< 1$.
        - `ValueError`: This can also be thrown in the case that NaN values occur. This
            can occur depending on the values of `label_alpha` and `sample_alpha`.
        - `NotImplementedError`: Is thrown if the provided dataset does not have a
            length given by `__len__()`.
        - `IndexError`: Is thrown if the label is of type `float`.
    """
    if label_alpha <= 0 or sample_alpha <= 0:
        raise ValueError(
            "Both `label_alpha` and `sample_alpha` must be greater than 0."
        )
    if num_labels < 1:
        raise ValueError("The number of labels must be at least 1.")

    try:
        train_data_len = len(data)  # type: ignore
    except NotImplementedError as err:
        err.add_note("The provided dataset must have `__len__()` implemented.")
        raise err

    generator = random_generator(rng)
    num_workers = topo.number_of_nodes(NodeKind.WORKER)
    sample_distr = generator.dirichlet(np.full(num_workers, sample_alpha))
    label_distr = generator.dirichlet(
        np.full(num_labels, label_alpha), size=num_workers
    )
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
            chosen_worker = generator.choice(temp_workers, p=probs_norm)
            indices[chosen_worker].append(idx)
            worker_samples[chosen_worker] += 1

    if ensure_at_least_one_sample:
        for worker in topo.workers:
            worker_with_most_samples = max(worker_samples, key=worker_samples.get)
            if worker_samples[worker.idx] == 0:
                index = indices[worker_with_most_samples].pop()
                worker_samples[worker_with_most_samples] -= 1

                indices[worker.idx].append(index)
                worker_samples[worker.idx] += 1

    if train_test_valid_split is None:
        train_indices = indices
        test_indices = None
        valid_indices = None

    elif len(train_test_valid_split) == 2:
        train_indices, test_indices = dict(), dict()
        valid_indices = None
        for w_idx, w_indices in indices.items():
            train_split, test_split = proportion_split(
                w_indices, train_test_valid_split
            )
            train_indices[w_idx] = train_split
            test_indices[w_idx] = test_split

    elif len(train_test_valid_split) == 3:
        train_indices, test_indices, valid_indices = dict(), dict(), dict()
        for w_idx, w_indices in indices.items():
            train_split, test_split, valid_split = proportion_split(
                w_indices, train_test_valid_split
            )
            train_indices[w_idx] = train_split
            test_indices[w_idx] = test_split
            valid_indices[w_idx] = valid_split

    else:
        raise ValueError("Invalid number of elements in `train_test_valid_split`.")

    return FederatedDataModule(
        data,
        train_indices=train_indices,
        test_indices=test_indices,
        valid_indices=valid_indices,
    )


def fed_barplot(
    data: FederatedDataModule,
    num_labels: int,
    width: float = 0.5,
    ax: Axes | None = None,
    **kwargs,
):
    """
    Plots the distribution of a `FederatedDataModule` instance as a stacked bar plot.

    Args:
        data (FederatedDataModule): Federated data module.
        num_labels (int): Number of labels in the dataset.
        width (float): Width of the bars.
        ax (Axes | None): Axes object to draw onto. If `None` is provided, then one
            will be created.
        **kwargs: Keyword arguments to pass to the `ax.bar()` method.

    Returns:
        Axes object that has distribution of federated data module plotted
            as a stacked bar plot.

    Throws:
        - `ValueError`: If the number of labels is less than 1.
        - `TypeError`: If the `fed_data` argument is not an instance of
            `FederatedDataModule`.
    """

    if num_labels < 1:
        raise ValueError("The number of labels must be at least 1.")

    if not isinstance(data, FederatedDataModule):
        raise TypeError(
            f"`fed_data` must be an instance of `FederatedDataModule` "
            f"(got `{type(data)}`)."
        )

    label_counts_per_worker = {
        label: np.zeros(len(data), dtype=np.int32) for label in range(num_labels)
    }

    i = 0  # worker index -- we use this instead of the actual worker ID to start at 0
    for node, _ in data:
        for batch in data.train_data(node):
            _, labels = batch
            for label in labels:
                label_counts_per_worker[label.item()][i] += 1

        i += 1  # increment worker index

    if ax is None:
        fig, ax = plt.subplots()

    bottom = np.zeros(len(data))
    workers = list(range(len(data)))
    for label, worker_count in label_counts_per_worker.items():
        ax.bar(
            workers,
            worker_count,
            width,
            bottom=bottom,
            label=label,
            **kwargs,
        )
        bottom += worker_count

    ax.set_xlabel("workers")
    ax.set_ylabel("label counts")

    return ax
