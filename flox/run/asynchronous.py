from __future__ import annotations

import pandas as pd
import torch

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
)

from flox.flock import Flock
from flox.learn.nn.model import FloxModule
from flox.strategies import Strategy
from flox.utils.data import FederatedDataset


def async_federated_fit(
    flock: Flock,
    module_cls: type[FloxModule],
    datasets: FederatedDataset,
    num_global_rounds: int,
    strategy: Strategy | str = "fedavg",
    executor: str = "thread",
    max_workers: int = 1,
) -> pd.DataFrame:
    """
    Asynchronous Federated Learning.

    Args:
        flock (Flock):
        module_cls (type[FloxModule]):
        datasets (FederatedDataset):
        num_global_rounds (int):
        strategy (Strategy | str):
        executor (str):
        max_workers (int):

    Returns:

    """
    # assert that the flock is a 2-tier system with no intermediary aggregators.
    futures = [worker for worker in flock.children(flock.leader)]
    executor = ThreadPoolExecutor(max_workers=max_workers)
    while True:
        done, futures = wait(futures, return_when=FIRST_COMPLETED)

        # TODO: Finish implementing asynchronous logic.

        fut = executor.submit(...)
        futures.append(fut)

        if not futures:
            break

    return pd.DataFrame.from_dict({})
