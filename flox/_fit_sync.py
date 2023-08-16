from concurrent.futures import Future
from functools import partial
from torch import nn

from flox.flock import Flock


def sync_federated_fit(module: nn.Module, flock: Flock, where: str = "local"):
    """A synchronous implementation of federated learning/fitting.

    Args:
        module (torch.nn.Module): The deep learning model to train.
        flock (Flock): The topology to run the FL process on.
        where (str, default="local"): Where to deploy the FL process.

    Returns:

    """
    worker = True
    if worker:
        local_training()
    else:
        children_futures = get_child_futures()
        my_future = Future()
        aggr_callback = partial(aggregator_round, children_futures, my_future)
        for fut in children_futures:
            fut.add_done_callback(lambda x: x)
        return my_future


def aggregator_round(*args, **kwargs):
    pass


def local_training():
    pass


def get_child_futures() -> list[Future]:
    pass
