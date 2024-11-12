from __future__ import annotations

import typing as t
from concurrent.futures import FIRST_COMPLETED, wait
from copy import deepcopy

import pandas as pd
from flox.federation.fed import Federation
from flox.logger import Logger
from tqdm import tqdm

if t.TYPE_CHECKING:
    from pandas import DataFrame

    from flox.federation.topologies import (
        AggrState,
        Node,
        NodeID,
        NodeState,
        Topology,
        WorkerState,
    )
    from flox.learn import FloxModule
    from flox.learn.data import FloxDataset
    from flox.learn.types import Params
    from flox.runtime import ResultFuture, Runtime
    from flox.strategies import Strategy


class AsyncFederation(Federation):
    """
    Asynchronous Federated Learning federation. This code is very much still in 'beta' and not as robust as the
    synchronous implementation, [SyncFederation][flox.runtime.federation.process_sync.SyncProcess].

    Notes:
        Currently, this federation is only compatible with two-tier ``Flock`` topologies.
    """

    def __init__(
        self,
        runtime: Runtime,
        topo: Topology,
        strategy: Strategy,
        module: FloxModule,
        dataset: FloxDataset,
        num_global_rounds: int,
        debug_mode: bool = False,
        logger: Logger | None = None,
        *args,
    ):
        # assert that the topologies is 2-tier
        if not topo.is_two_tier:
            raise ValueError(
                "Currently, FLoX only supports two-tier topologies for ``AsyncProcess`` execution."
            )

        super().__init__(
            runtime=runtime,
            topo=topo,
            num_global_rounds=num_global_rounds,
            module=module,
            dataset=dataset,
            strategy=strategy,
            logger=logger,
            debug_mode=debug_mode,
        )

        # self.runtime = runtime
        # self.topo = flock
        # self.num_global_rounds = num_global_rounds
        # self.global_model = module
        # self.dataset = dataset
        # self.strategy = strategy

        self.state_dict = None
        self.debug_mode = False
        self.params = self.global_model.state_dict()

        assert self.topo.coordinator is not None
        self.state = AggrState(
            self.topo.coordinator.idx,
            topo.children(topo.coordinator),
            self.global_model,
        )

    def start(self, debug_mode: bool = False) -> tuple[FloxModule, DataFrame]:
        if debug_mode:
            self.debug_mode = debug_mode
            if self.global_model is None:
                from flox.learn.model import DebugModule

                self.global_model = DebugModule()
                self.params = self.global_model.state_dict()
                self.state.pre_module = self.global_model

        if not self.topo.two_tier:
            raise ValueError(
                "Currently, FLoX only supports asynchronous FL for *two-tier* topologies."
            )

        histories: list[DataFrame] = []
        worker_rounds: dict[NodeID, int] = {}
        worker_states: dict[NodeID, NodeState] = {}
        worker_state_dicts: dict[NodeID, Params] = {}

        for worker in self.topo.workers:
            worker_rounds[worker.idx] = 0
            worker_states[worker.idx] = WorkerState(worker.idx)
            worker_state_dicts[worker.idx] = deepcopy(self.global_model.state_dict())

        progress_bar = tqdm(
            total=self.num_global_rounds * self.topo.number_of_workers,
            desc="federated_fit::async",
        )

        futures: t.Set[ResultFuture] = set()
        for worker in self.topo.workers:
            assert worker is not None  # mypy
            fut = self.start_worker_tasks(worker, self.topo.coordinator)
            futures.add(fut)

        while futures:
            dones, futures = wait(futures, return_when=FIRST_COMPLETED)
            if dones.intersection(futures):
                raise ValueError(
                    "Overlap between 'done' futures and 'to-be-done' Futures."
                )

            if len(dones) == 1:
                results = [dones.pop().result()]
            else:
                results = [done.result() for done in dones]

            for result in results:
                if worker_rounds[result.node_idx] >= self.num_global_rounds:
                    continue

                worker = self.topo[result.node_idx]
                assert worker is not None  # mypy
                worker_states[worker.idx] = result.node_state
                worker_state_dicts[worker.idx] = result.params

                result.history["round"] = worker_rounds[result.node_idx]
                histories.append(result.history)
                avg_params = self.strategy.aggr_strategy.aggregate_params(
                    self.state,
                    worker_states,
                    worker_state_dicts,
                    last_updated_node=worker.idx,
                )

                self.global_model.load_state_dict(avg_params)
                self.params = avg_params

                # if not self.debug_mode:
                #     test_acc, test_loss = test_model(self.global_model)
                #     result.history["test/acc"] = test_acc
                #     result.history["test/loss"] = test_loss

                fut = self.start_worker_tasks(worker, self.topo.coordinator)
                futures.add(fut)
                worker_rounds[result.node_idx] += 1
                progress_bar.update()

        return self.global_model, pd.concat(histories)

    def start_aggregator_tasks(
        self, node: Node, children: t.Iterable[Node] | None = None
    ):
        raise NotImplementedError(
            "Asynchronous FL processes do not support hierarchical topologies. "
            "So there is no implementation for launching aggregator tasks."
        )
