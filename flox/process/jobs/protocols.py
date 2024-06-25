"""
This file provides protocols for the different types of jobs that are used in FLoX. More specifically, we
define protocols for:

1. aggregation jobs (``AggregableJob``)
2. local training jobs (``TrainableJob``)

These protocols can be used to define custom impl of aggregation jobs for highly-customized FLoX processes.
However, this is not necessary for the vast majority of imaginable cases.
Should users choose to do this, it is up to the user's discretion to do so safely and correctly.

All protocols presented here rely on the ``__call__`` method to define callable classes.
So, each protocol implementation is a separate class with a matching signature.

Notes:
    It is worth noting that the body of the ``__call__`` method for any protocol implementation here
    must act as a "pure function". This means any necessary Python dependencies (i.e., `import` statements)
    needed for the method to run must be included within the body of the ``__call__`` method. This is
    required for execution on *Globus Compute*.
"""

from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flox.learn.data import FloxDataset
    from flox.topos import Node
    from flox.learn import FloxModule
    from flox.learn.typing import Params
    from flox.runtime import Result
    from flox.runtime.transfer import TransferProtocol
    from flox.strategies import AggregatorStrategy, TrainerStrategy, WorkerStrategy


class LauncherFunction(t.Protocol):
    """
    Utility protocol that simply identifies any of callable that takes a ``FlockNode`` as its first argument.
    """

    def __call__(self, node: Node, *args, **kwargs) -> t.Any:
        pass


@t.runtime_checkable
class AggregableJob(t.Protocol):
    """
    A protocol that defines functions that are valid impl to be used for model aggregation in
    launching FLoX processes.

    Notes:
        FLoX provides default impl of this protocol via
        [AggregateJob][flox.jobs.aggregation.AggregateJob] and
        [DebugAggregateJob][flox.jobs.aggregation.DebugAggregateJob].
    """

    @staticmethod
    def __call__(
        node: Node,
        children: t.Iterable[Node],
        transfer: TransferProtocol,
        aggr_strategy: AggregatorStrategy,
        results: list[Result],
    ) -> Result:
        """
        AggrCallable

        Args:
            node (Node):
            transfer (TransferProtocol):
            aggr_strategy (AggregatorStrategy):
            results (list[Result]):

        Returns:
            ...
        """


@t.runtime_checkable
class TrainableJob(t.Protocol):
    """
    A protocol that defines functions that are valid impl to be used for local training in
    launching FLoX processes.

    Notes:
        FLoX provides default impl of this protocol via
        [LocalTrainJob][flox.jobs.local_training.LocalTrainJob] and
        [DebugLocalTrainJob][flox.jobs.local_training.DebugLocalTrainJob].
    """

    @staticmethod
    def __call__(
        node: Node,
        parent: Node,
        global_model: FloxModule,
        module_state_dict: Params,
        dataset: FloxDataset,
        transfer: TransferProtocol,
        worker_strategy: WorkerStrategy,
        trainer_strategy: TrainerStrategy,
        **train_hyper_params,
    ) -> Result:
        """

        Args:
            node ():
            parent ():
            module ():
            module_state_dict ():
            dataset ():
            transfer ():
            worker_strategy ():
            trainer_strategy ():
            **train_hyper_params ():

        Returns:

        """
