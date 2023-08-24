import torch

from typing import Literal, Mapping, TypeAlias

from flox.flock import Flock, FlockNodeID
from flox.learn._sync import sync_federated_fit
from flox.learn.types import Kind, Where
from flox.strategies import Strategy
from flox.utils.data import FederatedDataset


def federated_fit(
    flock: Flock,
    module_cls: type[torch.nn.Module],
    datasets: FederatedDataset,
    num_global_rounds: int,
    strategy: Strategy,
    kind: Kind = "sync",
    where: Where = "local",
):
    """

    Args:
        flock ():
        module_cls ():
        datasets ():
        num_global_rounds ():
        strategy (Strategy):
        kind ():
        where ():

    Returns:

    """
    if kind == "sync":
        return sync_federated_fit(
            flock, module_cls, datasets, num_global_rounds, strategy
        )
    elif kind == "async":
        raise NotImplementedError("Asynchronous FL is not yet implemented.")
    else:
        raise ValueError(
            "Illegal value for argument `kind`. Must be either 'sync' or 'async'."
        )
