from __future__ import annotations

import typing
from concurrent.futures import FIRST_COMPLETED, wait

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from flox.data import FloxDataset
from flox.flock import Flock, NodeID
from flox.flock.states import AggrState, WorkerState, NodeState
from flox.jobs import LocalTrainJob
from flox.nn import FloxModule
from flox.runtime.process.proc import BaseProcess
from flox.runtime.runtime import Runtime
from flox.strategies import Strategy

if typing.TYPE_CHECKING:
    from flox.nn.typing import Params


class AsyncProcess(BaseProcess):
    """
    Asynchronous Federated Learning process.

    Notes:
        Currently, this process is only compatible with two-tier ``Flock`` topologies.
    """

    def __init__(
        self,
        runtime: Runtime,
        flock: Flock,
        num_global_rounds: int,
        module: FloxModule,
        dataset: FloxDataset,
        strategy: Strategy,
        *args,
    ):
        # assert that the flock is 2-tier
        if not flock.is_two_tier:
            raise ValueError(
                "Currently, FLoX only supports two-tier topologies for ``AsyncProcess`` execution."
            )

        self.runtime = runtime
        self.flock = flock
        self.num_global_rounds = num_global_rounds
        self.global_module = module
        self.dataset = dataset
        if isinstance(strategy, str):
            self.strategy = Strategy.get_strategy(strategy)()
        else:
            self.strategy = strategy

        self.state_dict = None
        self.debug_mode = False

        assert self.flock.leader is not None
        self.state = AggrState(self.flock.leader.idx)

    def start(self, debug_mode: bool = False) -> tuple[FloxModule, DataFrame]:
        if not self.flock.two_tier:
            raise ValueError

        histories: list[DataFrame] = []
        worker_rounds: dict[NodeID, int] = {}
        worker_states: dict[NodeID, NodeState] = {}
        worker_state_dicts: dict[NodeID, Params] = {}
        for worker in self.flock.workers:
            worker_rounds[worker.idx] = 0
            worker_states[worker.idx] = WorkerState(worker.idx)
            worker_state_dicts[worker.idx] = self.global_module.state_dict()

        futures = set()
        progress_bar = tqdm(total=self.num_global_rounds * self.flock.number_of_workers)
        for worker in self.flock.workers:
            # data = self.dataset[worker.idx]
            job = LocalTrainJob()
            data = self.fetch_worker_data(worker)
            fut = self.runtime.submit(
                job,
                worker,
                parent=self.flock.leader,
                dataset=self.runtime.proxy(data),
                module=self.global_module,
                module_state_dict=self.runtime.proxy(self.global_module.state_dict()),
                strategy=self.strategy,
            )
            futures.add(fut)

        while futures:
            dones, futures = wait(futures, return_when=FIRST_COMPLETED)
            if dones.intersection(futures):
                raise ValueError(
                    "Overlap between 'done' futures and 'to-be-done' futures."
                )

            if len(dones) == 1:
                results = [dones.pop().result()]
            else:
                results = [done.result() for done in dones]

            for result in results:
                if worker_rounds[result.node_idx] >= self.num_global_rounds:
                    continue

                worker = self.flock[result.node_idx]
                worker_states[worker.idx] = result.node_state
                worker_state_dicts[worker.idx] = result.params
                result.history["round"] = worker_rounds[result.node_idx]
                histories.append(result.history)
                avg_state_dict = self.strategy.agg_param_aggregation(
                    self.state, worker_states, worker_state_dicts
                )
                self.global_module.load_state_dict(avg_state_dict)
                # data = self.dataset[worker.idx]
                job = LocalTrainJob()
                data = self.dataset.load(worker)
                fut = self.runtime.submit(
                    job,
                    worker,
                    parent=self.flock.leader,
                    dataset=self.runtime.proxy(data),
                    module=self.global_module,
                    module_state_dict=self.runtime.proxy(
                        self.global_module.state_dict()
                    ),
                    strategy=self.strategy,
                )
                # futures.append(fut)
                futures.add(fut)
                worker_rounds[result.node_idx] += 1
                progress_bar.update()

        # TODO: Obviously fix this.
        return self.global_module, pd.concat(histories)

    def step(self):
        ...
