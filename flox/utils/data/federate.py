import numpy as np

from flox.flock import Flock, FlockNodeID
from numpy.random import RandomState
from numpy.typing import ArrayLike
from scipy.special import softmax
from scipy.stats import rv_continuous
from torch.utils.data import Dataset, Subset
from typing import Mapping, Optional


def randomly_federate_dataset(
    flock: Flock,
    data: Dataset,
    shuffle: bool = True,
    random_state: Optional[RandomState] = None,
) -> Mapping[FlockNodeID, Subset]:
    if random_state is None:
        random_state = RandomState()
    n = flock.number_of_workers
    lengths = [len(data) // n] * n
    sum_lengths, len_data = sum(lengths), len(data)
    if sum_lengths < len_data:
        lengths[0] += len_data - sum_lengths

    universe_indices = list(range(len_data))
    if shuffle:
        random_state.shuffle(universe_indices)
    fed_data: dict[FlockNodeID, Subset] = {}
    length_iter = iter(lengths)
    start, end = 0, next(length_iter)

    for worker in flock.workers:
        indices = universe_indices[start:end]
        fed_data[worker.idx] = Subset(data, indices)
        try:
            start = end
            end = start + next(length_iter)
        except StopIteration:
            break

    return fed_data


def federate_dataset(
    flock: Flock,
    data: Dataset,
    lengths: Optional[ArrayLike] = None,
    overlap: bool = False,
    stats_gen: Optional[rv_continuous] = None,
    random_state: Optional[RandomState] = None,
) -> Mapping[FlockNodeID, Subset]:
    """

    Args:
        flock (Flock):
        data (Dataset):
        lengths (Optional[ArrayLike]):
        overlap (bool): If False, then data samples cannot be shared across worker nodes in the Flock; otherwise True.
        stats_gen (Optional[rv_continuous]):
        random_state (Optional[RandomState]):

    Returns:
        Mapping[FlockNodeID, Subset]: Subsets of the original Dataset, `data`,
    """
    fed_data: dict[FlockNodeID, Subset] = {}
    for worker in flock.workers:
        indices = [1, 2, 3]  # _generate_indices(...)
        subset = Subset(data, indices)
        fed_data[worker.idx] = subset
    return fed_data


def _generate_indices(
    n_workers: int, total_samples: int, lengths: ArrayLike, overlap: bool, random_state
):
    sample_spread = 0.5
    worker_samples = softmax(
        [np.random.normal(1, sample_spread) for _ in range(n_workers)]
    )
    worker_samples *= total_samples
    worker_alpha = total_samples
    worker_samples = worker_samples.astype(np.int32)
    return [1, 2, 3]
