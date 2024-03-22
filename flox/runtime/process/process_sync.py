from __future__ import annotations

import functools
import typing
from concurrent.futures import Future
from copy import deepcopy

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from flox.data import FloxDataset
from flox.flock import Flock, FlockNode, NodeKind
from flox.flock.states import AggrState
from flox.jobs import AggregateJob, DebugLocalTrainJob, LocalTrainJob
from flox.nn import FloxModule
from flox.runtime.process.future_callbacks import all_child_futures_finished_cbk
from flox.runtime.process.process import Process
from flox.runtime.process.testing import test_model
from flox.runtime.result import Result
from flox.runtime.runtime import Runtime
from flox.strategies import Strategy

if typing.TYPE_CHECKING:
    from flox.nn.typing import Params


class SyncProcess(Process):
    """
    Synchronous Federated Learning process.
    """

    flock: Flock
    runtime: Runtime
    global_module: FloxModule
    strategy: Strategy
    dataset: FloxDataset
    aggr_callback: typing.Any  # TODO: Fix
    params: Params | None
    debug_mode: bool
    pbar_desc: str

    def __init__(
        self,
        runtime: Runtime,
        flock: Flock,
        num_global_rounds: int,
        module: FloxModule,
        dataset: FloxDataset,
        strategy: Strategy,
    ):
        self.flock = flock
        self.runtime = runtime
        self.num_global_rounds = num_global_rounds
        self.global_module = module
        self.strategy = strategy
        self.dataset = dataset

        self.aggr_callback = all_child_futures_finished_cbk
        self.params = None
        self.debug_mode = False
        self.pbar_desc = "federated_fit::sync"
        self.seed = 0

        # TODO: Add description option for the progress bar when it's training.
        #  Also, add a configurable stop condition

    def start(self, testing_mode: bool = False) -> tuple[FloxModule, DataFrame]:
        if testing_mode:
            from flox.runtime.process.debug_utils import DebugModule

            self.debug_mode = True
            self.global_module = DebugModule()

        histories = []
        progress_bar = tqdm(total=self.num_global_rounds, desc=self.pbar_desc)
        for round_num in range(self.num_global_rounds):
            self.params = self.global_module.state_dict()
            step_result = self.step().result()
            step_result.history["round"] = round_num

            if not testing_mode:
                test_acc, test_loss = test_model(self.global_module)
                step_result.history["test/acc"] = test_acc
                step_result.history["test/loss"] = test_loss

            histories.append(step_result.history)
            self.global_module.load_state_dict(step_result.params)
            progress_bar.update()

        history = pd.concat(histories)
        return self.global_module, history

    def step(
        self,
        node: FlockNode | None = None,
        parent: FlockNode | None = None,
    ) -> Future:
        flock = self.flock
        value_err_template = "Illegal kind ({}) of `FlockNode` (ID=`{}`)."

        if node is None:
            assert flock.leader is not None
            node = flock.leader
        elif isinstance(node, FlockNode):
            node = node
        else:
            raise ValueError

        match flock.get_kind(node):
            case NodeKind.LEADER | NodeKind.AGGREGATOR:
                return self.submit_aggr_job(node)
                # if self.debug_mode:
                #     return self.submit_aggr_debug_job(node) # FIXME
                # else:
                #     return self.submit_aggr_job(node)

            case NodeKind.WORKER:
                assert parent is not None
                # (^^^) avoids mypy issue which won't naturally occur with valid Flock topo
                if self.debug_mode:
                    return self.submit_worker_debug_job(node, parent)
                else:
                    return self.submit_worker_job(node, parent)

            case _:
                kind = flock.get_kind(node)
                idx = node.idx
                raise ValueError(value_err_template.format(kind, idx))

    ########################################################################################################
    ########################################################################################################

    def submit_aggr_job(self, node: FlockNode) -> Future[Result]:
        children = list(self.flock.children(node.idx))
        aggr_state = AggrState(node.idx, children, deepcopy(self.global_module))
        selected_workers = self.strategy.client_strategy.select_worker_nodes(
            aggr_state, children, self.seed
        )
        self.seed += 1

        selected_worker_futures = [
            self.step(node=worker, parent=node) for worker in selected_workers
        ]

        # This partial function (`subtree_done_cbk`) will perform the aggregation only
        # when all futures in `children_futures` has completed. This partial function
        # will be added as a callback which is run after the completion of each child
        # future. But, it will only perform aggregation once since only the last future
        # to be completed will activate the conditional.
        future: Future[Result] = Future()
        job = AggregateJob()
        subtree_done_cbk = functools.partial(
            self.aggr_callback,
            job,
            future,
            children,
            selected_worker_futures,
            deepcopy(self.global_module),
            node,
            self.runtime,
            self.strategy.aggr_strategy,
        )
        for child_fut in selected_worker_futures:
            child_fut.add_done_callback(subtree_done_cbk)

        return future

    def submit_aggr_debug_job(self, node: FlockNode) -> Future[Result]:
        raise NotImplementedError

    def submit_worker_job(self, node: FlockNode, parent: FlockNode) -> Future[Result]:
        job = LocalTrainJob()
        data = self.dataset
        return self.runtime.submit(
            job,
            node=node,
            parent=parent,
            global_model=deepcopy(self.global_module),
            worker_strategy=self.strategy.worker_strategy,
            trainer_strategy=self.strategy.trainer_strategy,
            dataset=self.runtime.proxy(data),
            module_state_dict=self.runtime.proxy(self.params),
        )

    def submit_worker_debug_job(
        self, node: FlockNode, parent: FlockNode
    ) -> Future[Result]:
        job = DebugLocalTrainJob()
        data = self.dataset
        return self.runtime.submit(
            job,
            node=node,
            parent=parent,
            global_model=deepcopy(self.global_module),
            worker_strategy=self.strategy.worker_strategy,
            trainer_strategy=self.strategy.trainer_strategy,
            dataset=self.runtime.proxy(data),
            module_state_dict=self.runtime.proxy(self.params),
        )
