from flox.fit_sync import sync_federated_fit
from flox.flock import Flock
from torch import nn


def federated_fit(
    module: nn.Module, flock: Flock, deploy: str = "local", mode: str = "sync"
):
    if mode == "sync":
        return sync_federated_fit(module, flock, deploy)
    else:
        raise NotImplementedError()
