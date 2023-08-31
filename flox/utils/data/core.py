import numpy as np

from collections import defaultdict
from scipy import stats
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Mapping

from flox.flock import FlockNodeID, FlockNode, Flock


def federated_split(
    data: Dataset,
    flock: Flock,
    num_labels: int,
    samples_alpha: float = 1.0,
    labels_alpha: float = 1.0,
) -> Mapping[FlockNodeID, Subset]:
    """

    Args:
        data ():
        workers ():
        num_labels ():
        samples_alpha ():
        labels_alpha ():

    Returns:

    """
    assert samples_alpha > 0
    assert labels_alpha > 0

    num_workers = len(list(flock.workers))
    sample_distr = stats.dirichlet(np.full(num_workers, samples_alpha))
    label_distr = stats.dirichlet(np.full(num_labels, labels_alpha))

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
                probs.append(label_probs[w.idx][label])
                temp_workers.append(w.idx)
        probs = np.array(probs)
        probs = probs / probs.sum()

        if len(temp_workers) > 0:
            chosen_worker = np.random.choice(temp_workers, p=probs)
            indices[chosen_worker].append(idx)
            worker_samples[chosen_worker] += 1

    subsets = {w.idx: Subset(data, indices[w.idx]) for w in flock.workers}
    return subsets
