from concurrent.futures import FIRST_COMPLETED, wait

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from flox.data import FloxDataset
from flox.flock import Flock, FlockNodeID
from flox.flock.states import FloxAggregatorState, FloxWorkerState
from flox.nn import FloxModule
from flox.runtime.jobs import local_training_job
from flox.runtime.launcher import Launcher
from flox.runtime.process.proc import BaseProcess
from flox.runtime.transfer import BaseTransfer
from flox.strategies import Strategy
from flox.typing import StateDict


class AsyncProcess(BaseProcess):
    """
    Asynchronous Federated Learning process.

    Notes:
        Currently, this process is only compatible with two-tier ``Flock`` topologies.
    """

    def __init__(
        self,
        flock: Flock,
        num_global_rounds: int,
        launcher: Launcher,
        module: FloxModule,
        dataset: FloxDataset,
        transfer: BaseTransfer,
        strategy: Strategy | str,
        *args,
    ):
        # assert that the flock is 2-tier
        if not flock.is_two_tier:
            raise ValueError(
                "Currently, FLoX only supports two-tier topologies for ``AsyncProcess`` execution."
            )

        self.flock = flock
        self.launcher = launcher
        self.num_global_rounds = num_global_rounds
        self.global_module = module
        self.transfer = transfer
        self.dataset = dataset
        if isinstance(strategy, str):
            self.strategy = Strategy.get_strategy(strategy)()
        else:
            self.strategy = strategy

        self.state_dict = None
        self.debug_mode = False
        self.state = FloxAggregatorState(self.flock.leader.idx)

    def start(self, debug_mode: bool = False) -> tuple[FloxModule, DataFrame]:
        if not self.flock.two_tier:
            raise ValueError

        histories: list[DataFrame] = []
        worker_rounds: dict[FlockNodeID, int] = {}
        worker_states: dict[FlockNodeID, FloxWorkerState] = {}
        worker_state_dicts: dict[FlockNodeID, StateDict] = {}
        for worker in self.flock.workers:
            worker_rounds[worker.idx] = 0
            worker_states[worker.idx] = FloxWorkerState(worker.idx)
            worker_state_dicts[worker.idx] = self.global_module.state_dict()

        futures = []
        progress_bar = tqdm(total=self.num_global_rounds * self.flock.number_of_workers)
        for worker in self.flock.workers:
            # data = self.dataset[worker.idx]
            data = self.fetch_worker_data(worker)
            fut = self.launcher.submit(
                local_training_job,
                worker,
                parent=self.flock.leader,
                dataset=self.transfer.proxy(data),
                module=self.global_module,
                module_state_dict=self.transfer.proxy(self.global_module.state_dict()),
                transfer=self.transfer,
                strategy=self.strategy,
            )
            futures.append(fut)

        while futures:
            dones, futures = wait(futures, return_when=FIRST_COMPLETED)
            if dones.intersection(futures):
                raise ValueError(
                    "Overlap between 'done' futures and 'to-be-done' futures."
                )

            dones, futures = list(dones), list(futures)

            if len(dones) == 1:
                results = [dones.pop().result()]
            else:
                results = [done.result() for done in dones]

            for result in results:
                if worker_rounds[result.node_idx] >= self.num_global_rounds:
                    continue

                worker = self.flock[result.node_idx]
                worker_states[worker.idx] = result.node_state
                worker_state_dicts[worker.idx] = result.state_dict
                result.history["round"] = worker_rounds[result.node_idx]
                histories.append(result.history)
                avg_state_dict = self.strategy.agg_param_aggregation(
                    self.state, worker_states, worker_state_dicts
                )
                self.global_module.load_state_dict(avg_state_dict)
                # data = self.dataset[worker.idx]
                data = self.dataset.load(worker)
                fut = self.launcher.submit(
                    local_training_job,
                    worker,
                    parent=self.flock.leader,
                    dataset=self.transfer.proxy(data),
                    module=self.global_module,
                    module_state_dict=self.transfer.proxy(
                        self.global_module.state_dict()
                    ),
                    transfer=self.transfer,
                    strategy=self.strategy,
                )
                futures.append(fut)
                worker_rounds[result.node_idx] += 1
                progress_bar.update()

        # TODO: Obviously fix this.
        return self.global_module, pd.concat(histories)

    def step(self):
        ...
