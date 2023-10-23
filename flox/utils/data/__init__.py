"""
This module implements several functions that can be used to create `FederatedDataset` objects. These are used
to do local simulations (or remote simulations) with benchmark Machine Learning datasets (e.g., MNIST, CIFAR-10).
More specifically, this module aims to make it easy to launch FL experiments with different statistical data
distributions.
"""
from flox.utils.data.base import FederatedDataset
from flox.utils.data.dirs import FederatedDataDir
from flox.utils.data.subsets import FederatedSubsets
from flox.utils.data.core import fed_barplot, federated_split

__all__ = [
    "FederatedDataset",
    "FederatedDataDir",
    "FederatedSubsets",
    "fed_barplot",
    "federated_split",
]
