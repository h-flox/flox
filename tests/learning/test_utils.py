import itertools
import typing as t

import pytest
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from flight.federation.topologies.utils import flat_topology
from flight.learning.utils import federated_split

NumberOfLabels: t.TypeAlias = int


@pytest.fixture
def pair_topo():
    return flat_topology(2)


@pytest.fixture
def simple_data() -> tuple[Dataset, NumberOfLabels]:
    inputs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return TensorDataset(inputs, targets), 2


class TestFederatedSplit:
    def test_valid_train_only_uses(self, pair_topo, simple_data):
        data, num_labels = simple_data
        fed_data = federated_split(
            pair_topo,
            data,
            num_labels=num_labels,
            label_alpha=1.0,
            sample_alpha=1.0,
            rng=1,
        )

        for worker in pair_topo.workers:
            assert worker in fed_data
            assert worker.idx in fed_data
            assert isinstance(fed_data.train_data(worker), DataLoader)
            assert fed_data.test_data(worker) is None
            assert fed_data.valid_data(worker) is None

    def test_valid_train_test_valid_uses(self, pair_topo, simple_data):
        data, num_labels = simple_data
        fed_data = federated_split(
            pair_topo,
            data,
            num_labels=num_labels,
            label_alpha=1.0,
            sample_alpha=1.0,
            train_test_valid_split=(0.9, 0.1),
            rng=1,
        )

        for worker in pair_topo.workers:
            assert worker in fed_data
            assert worker.idx in fed_data
            assert isinstance(fed_data.train_data(worker), DataLoader)
            assert isinstance(fed_data.test_data(worker), DataLoader)
            assert fed_data.valid_data(worker) is None

        fed_data = federated_split(
            pair_topo,
            data,
            num_labels=num_labels,
            label_alpha=1.0,
            sample_alpha=1.0,
            train_test_valid_split=(0.8, 0.1, 0.1),
            rng=1,
        )

        for worker in pair_topo.workers:
            assert worker in fed_data
            assert worker.idx in fed_data
            assert isinstance(fed_data.train_data(worker), DataLoader)
            assert isinstance(fed_data.test_data(worker), DataLoader)
            assert isinstance(fed_data.valid_data(worker), DataLoader)

    def test_invalid_splits(self, pair_topo, simple_data):
        data, num_labels = simple_data
        with pytest.raises(ValueError):
            federated_split(
                pair_topo,
                data,
                num_labels=num_labels,
                label_alpha=1.0,
                sample_alpha=1.0,
                train_test_valid_split=(0.9, 0.1, 0.1),
                rng=1,
            )

        with pytest.raises(ValueError):
            federated_split(
                pair_topo,
                data,
                num_labels=num_labels,
                label_alpha=1.0,
                sample_alpha=1.0,
                train_test_valid_split=(1.1, 0.0),
                rng=1,
            )

    def test_invalid_alphas(self, pair_topo, simple_data):
        data, num_labels = simple_data

        sample_alphas = [-1.0, -0.1, 0.0]
        label_alphas = [-1.0, -0.1, 0.0]

        for s_alpha, l_alpha in itertools.product(sample_alphas, label_alphas):
            with pytest.raises(ValueError):
                federated_split(
                    pair_topo,
                    data,
                    num_labels=num_labels,
                    label_alpha=s_alpha,
                    sample_alpha=l_alpha,
                    rng=1,
                )
