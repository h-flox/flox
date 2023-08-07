import lightning as L

from depr.aggregator import SynchAggregatorLogicInterface
from depr.aggregator.asynch.base import AsynchAggregatorLogicInterface
from depr.fit._async import _async_federated_fit
from depr.fit._sync import _sync_federated_fit
from depr.worker import WorkerLogicInterface
from numpy.random import RandomState
from typing import Optional, Union


def federated_fit(
    global_module: L.LightningModule,
    aggr: Union[AsynchAggregatorLogicInterface, SynchAggregatorLogicInterface],
    workers: dict[str, WorkerLogicInterface],
    global_rounds: int,
    test: bool = False,
    n_threads: int = 4,
    fit_kind: str = "synch",
    random_state: Optional[RandomState] = None,
    **kwargs
):
    if random_state is None:
        random_state = RandomState()

    if fit_kind == "asynch":
        # TODO: Finish implementing the asynch FL fitting function.
        train_results, test_results = _async_federated_fit(
            global_module,
            aggr,
            workers,
            global_rounds,
            test,
            n_threads,
            random_state,
            **kwargs
        )
    elif fit_kind == "synch":
        train_results, test_results = _sync_federated_fit(
            global_module,
            aggr,
            workers,
            global_rounds,
            test,
            n_threads,
            random_state,
            **kwargs
        )
    else:
        raise ValueError("Illegal value for argument `fit_kind`.")

    return {
        "module": global_module,
        "train_results": train_results,
        "test_results": test_results,
    }
