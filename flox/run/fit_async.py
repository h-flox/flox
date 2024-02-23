from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pandas as pd

from flox.backends.transfer.base import BaseTransfer
from flox.data import FloxDataset
from flox.flock import Flock
from flox.flock.node import FlockNodeID
from flox.nn import FloxModule
from flox.run.jobs import local_training_job
from flox.strategies import Strategy


def async_federated_fit(
    flock: Flock,
    module_cls: type[FloxModule],
    datasets: FloxDataset,
    num_global_rounds: int,
    strategy: Strategy | str = "fedavg",
    max_workers: int = 1,
) -> pd.DataFrame:
    """
    Asynchronous Federated Learning.

    Args:
        flock (Flock):
        module_cls (type[FloxModule]):
        datasets (FloxDataset):
        num_global_rounds (int):
        strategy (Strategy | str):
        executor (str):
        max_workers (int):

    Returns:

    """
    # assert that the flock is a 2-tier system with no intermediary aggregators.
    executor = ThreadPoolExecutor(max_workers=max_workers)
    global_module = module_cls()

    if isinstance(strategy, str):
        strategy = Strategy.get_strategy(strategy)()

    futures = {
        executor.submit(
            local_training_job,
            node,
            BaseTransfer(),
            parent=flock.leader,
            strategy=strategy,
            module_cls=module_cls,
            module_state_dict=global_module.state_dict(),
            dataset=datasets[node.idx],
        )
        for node in flock.workers
    }
    num_local_fitins: Counter[FlockNodeID] = Counter()

    while futures:
        done, futures = wait(futures, return_when=FIRST_COMPLETED)

        # TODO: Finish implementing asynchronous logic.

        if len(done) == 1:
            results = [done.pop().result()]
        else:
            results = [d.result() for d in done]

        for res in results:
            num_local_fitins[res.node_idx] += 1
            node = flock[res.node_idx]

            if num_local_fitins[res.node_idx] < num_global_rounds:
                fut = executor.submit(
                    local_training_job,
                    node,
                    BaseTransfer(),
                    parent=flock.leader,
                    strategy=strategy,
                    module_cls=module_cls,
                    module_state_dict=global_module.state_dict(),
                    dataset=datasets[node.idx],
                )
                futures.add(fut)

    return pd.DataFrame.from_dict({})
