from __future__ import annotations

import pandas as pd

from collections import defaultdict
from concurrent.futures import (
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
)

from flox.flock import Flock
from flox.nn import FloxModule
from flox.run.jobs import local_training_job
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
    executor = ThreadPoolExecutor(max_workers=max_workers)
    global_module = module_cls()
    hyper_params = {}

    futures = [
        executor.submit(
            local_training_job,
            node,
            parent=flock.leader,
            strategy=strategy,
            module_cls=module_cls,
            module_state_dict=global_module.state_dict(),
            dataset=datasets[node.idx],
            **hyper_params,
        )
        for node in flock.workers
    ]
    num_local_fitins = defaultdict(int)

    while futures:
        done, futures = wait(futures, return_when=FIRST_COMPLETED)

        # TODO: Finish implementing asynchronous logic.

        if len(done) == 1:
            results = [done.pop().result()]
        else:
            results = [d.result() for d in done]

        for res in results:
            futures = list(futures)
            num_local_fitins[res.idx] += 1
            node = flock[res.idx]

            if num_local_fitins[res.idx] < num_global_rounds:
                fut = executor.submit(
                    local_training_job,
                    node,
                    parent=flock.leader,
                    strategy=strategy,
                    module_cls=module_cls,
                    module_state_dict=global_module.state_dict(),
                    dataset=datasets[node.idx],
                    **hyper_params,
                )
                futures.append(fut)

    return pd.DataFrame.from_dict({})
