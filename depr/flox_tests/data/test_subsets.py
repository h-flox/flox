import torch
from torch.utils.data import Dataset, TensorDataset

from flox.federation.topologies import Node, NodeKind
from flox.learn.data import FederatedSubsets


def test_1():
    n_samples = 100
    n_features = 10
    generator = torch.manual_seed(0)

    dataset = TensorDataset(
        torch.rand((n_samples, n_features), generator=generator),
        torch.randint(0, 1 + 1, (n_samples,), generator=generator),
    )

    n_workers = 10
    data_per_worker = n_samples // n_workers
    worker_indices = {
        Node(worker, NodeKind.WORKER): list(
            range(worker * data_per_worker, (worker + 1) * data_per_worker)
        )
        for worker in range(n_workers)
    }

    nodes = list(worker_indices)

    subsets = FederatedSubsets(dataset, worker_indices)
    subsetted_data = subsets.load(nodes[0])
    assert isinstance(subsetted_data, Dataset)
