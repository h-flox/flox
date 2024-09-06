from __future__ import annotations

import abc
import typing as t
from copy import deepcopy

from flox.federation.jobs import DebugLocalTrainJob, LocalTrainJob
from flox.logger import NullLogger

if t.TYPE_CHECKING:
    from pandas import DataFrame

    from flox.federation.topologies import Node, Topology
    from flox.learn.data import FloxDataset
    from flox.learn.model import FloxModule
    from flox.learn.types import Params
    from flox.logger import Logger
    from flox.runtime import ResultFuture, Runtime
    from flox.strategies import (
        AggregatorStrategy,
        ClientStrategy,
        Strategy,
        TrainerStrategy,
        WorkerStrategy,
    )


class Federation(abc.ABC):
    process_name: str
    """Name of the federation type."""

    runtime: Runtime

    params: Params | None

    dataset: FloxDataset

    strategy: Strategy

    global_model: FloxModule

    debug_mode: bool

    logger: Logger | None

    def __init__(
        self,
        runtime: Runtime,
        topo: Topology,
        num_global_rounds: int,
        module: FloxModule,
        dataset: FloxDataset,
        strategy: Strategy,
        logger: Logger | None,
        debug_mode: bool,
    ):
        self.runtime = runtime
        self.topo = topo
        self.num_global_rounds = num_global_rounds
        self.global_model = module
        self.dataset = dataset
        self.strategy = strategy
        if logger is None:
            logger = NullLogger()
        self.logger = logger
        self.debug_mode = debug_mode

    def start_worker_tasks(self, node: Node, parent: Node) -> ResultFuture:
        """
        Submits local training jobs to the designated child node (`node`) with the result returning to the `parent`.

        Args:
            node: The node to submit the local training job to.
            parent: The node's parent that receives the returned result.

        Returns:
            A future reference to the result.
        """
        if self.debug_mode:
            job = DebugLocalTrainJob()
            dataset = None
            module_state_dict = None
        else:
            job = LocalTrainJob()
            dataset = self.runtime.transfer(self.dataset)
            module_state_dict = self.runtime.transfer(self.params)

        return self.runtime.submit(
            job,
            node=node,
            parent=parent,
            # ...
            global_model=self.runtime.transfer(deepcopy(self.global_model)),
            module_state_dict=module_state_dict,
            worker_strategy=self.worker_strategy,
            trainer_strategy=self.trainer_strategy,
            dataset=dataset,
        )

    ####################
    # ABSTRACT METHODS #
    ####################
    @abc.abstractmethod
    def start(self, debug_mode: bool = False) -> tuple[FloxModule, DataFrame]:
        """Starts the FL federation.

        Returns:
            A tuple that contains the following items
            (i) the trained global module hosted on the leader of `topologies` and
            (ii) the history metrics from training.
        """

    @abc.abstractmethod
    def start_aggregator_tasks(
        self, node: Node, children: t.Iterable[Node] | None = None
    ):
        pass

    ##############
    # PROPERTIES #
    ##############

    @property
    def client_strategy(self) -> ClientStrategy:
        return self.strategy.client_strategy

    @property
    def aggr_strategy(self) -> AggregatorStrategy:
        return self.strategy.aggr_strategy

    @property
    def worker_strategy(self) -> WorkerStrategy:
        return self.strategy.worker_strategy

    @property
    def trainer_strategy(self) -> TrainerStrategy:
        return self.strategy.trainer_strategy
