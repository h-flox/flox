"""
This module implements several functions that can be used to create `FederatedDataset` objects. These are used
to do local simulations (or remote simulations) with benchmark Machine Learning datasets (e.g., MNIST, CIFAR-10).
More specifically, this module aims to make it easy to launch FL experiments with different statistical data
distributions.
"""
from typing import NewType, Mapping, Union

from torch.utils import data as torch_data

from flox.flock import FlockNodeID
from flox.utils.data.core import FederatedDataset, fed_barplot, federated_split


__all__ = ["FederatedDataset", "fed_barplot", "federated_split"]
