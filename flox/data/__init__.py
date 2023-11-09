"""
This module implements several functions that can be used to create `FloxDataset` objects. These are used
to do local simulations (or remote simulations) with benchmark Machine Learning data (e.g., `MNIST`, `CIFAR-10`).
More specifically, this module aims to make it easy to launch FL experiments with different statistical data
distributions.

## Kinds of Datasets in FLoX
Datasets in FLoX can grossly be thought of from a simple mental model with two types of data:

2. **Real-World:** naturally decentralized data for real-world FL experimentation/deployment
1. **Simulated:** federated subsets of a single, centralized dataset** for simulated FL experimentation

The former is aptly referred to as **Real-World** data and the latter is referred to as **Simulated** data.

```mermaid
flowchart LR
    base([FLoX-compatible Datasets])

    torch([PyTorch])
    flox([FLoX])

    mp[Map-style Dataset]
    it[Iter-style Dataset]
    subs[Federated Subsets]

    subgraph Real-World
        direction TB
        torch-->it
        torch-->mp
    end

    subgraph Simulated
        direction TB
        flox-->subs
    end

    base-->Real-World
    base-->Simulated

    Real-World-->|FLoX utility funcs|Simulated
```

### Real-World Datasets
Real-world data refer to the data that are already located on decentralized devices. In this case, the data are
naturally already decentralized. This means the only tasks necessary are to load them into memory and preprocess the
data before using them to train the model. The standard PyTorch ``Dataset`` object included in the ``torch.utils.data``
module already does this well. So these types of use cases for FLoX users will only require a standard PyTorch
``Dataset``.

Standard PyTorch data can be written using an iter-style or a map-style implementation (read more on this
[here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)).


### Simulated Datasets
Simulated data are needed for FL research and experimentation for discovering better algorithms for aggregation,
parameter communication, worker/client/endpoint selection, etc. In this way, a simulated dataset takes a dataset
that is naturally *centralized* and converts it into a *decentralized* dataset. Unlike real-world data that are
already decentralized (e.g., data on decentralized sensors), simulated data will take a benchmark dataset (e.g.,
ImageNet) and split across a ``Flock`` network.

FLoX includes utility functions to simplify the conversion from a standard, centralized PyTorch dataset to a
simulated, decentralized dataset.
"""
from flox.data.core import FloxDataset, FederatedSubsets
from flox.data.utils import fed_barplot, federated_split

__all__ = ["FloxDataset", "fed_barplot", "federated_split"]
