from __future__ import annotations

import functools
import typing
from concurrent.futures import Future

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from flox.data import FloxDataset
from flox.flock import Flock, FlockNode, FlockNodeKind
from flox.flock.states import AggrState
from flox.nn import FloxModule
from flox.runtime.jobs import local_training_job, debug_training_job
from flox.runtime.process.future_callbacks import all_child_futures_finished_cbk
from flox.runtime.process.proc import BaseProcess
from flox.runtime.result import Result
from flox.runtime.runtime import Runtime
from flox.strategies import Strategy

if typing.TYPE_CHECKING:
    from flox.nn.typing import StateDict


class SyncProcess(BaseProcess):
    """
    Synchronous Federated Learning process.
    """

    flock: Flock
    runtime: Runtime
    global_module: FloxModule
    strategy: Strategy
    dataset: FloxDataset
    aggr_callback: typing.Any  # TODO: Fix
    state_dict: StateDict | None
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
        self.state_dict = None
        self.debug_mode = False
        self.pbar_desc = "federated_fit::sync"

        # TODO: Add description option for the progress bar when it's training.
        #  Also, add a configurable stop condition

    def start(
        self, testing_mode: bool = False
    ) -> tuple[FloxModule, DataFrame]:  # , global_module: FloxModule):
        if testing_mode:
            from flox.runtime.process.debug_utils import DebugModule

            self.debug_mode = True
            self.global_module = DebugModule()

        dataframes = []
        progress_bar = tqdm(total=self.num_global_rounds, desc=self.pbar_desc)
        for round_num in range(self.num_global_rounds):
            self.state_dict = self.global_module.state_dict()
            future = self.step()
            update = future.result()
            history = update.history
            history["round"] = round_num
            dataframes.append(history)
            self.global_module.load_state_dict(update.state_dict)
            progress_bar.update()

        history = pd.concat(dataframes)
        return self.global_module, history

    def step(
        self,
        node: FlockNode | None = None,
        parent: FlockNode | None = None,
    ) -> Future:
        flock = self.flock
        value_err_template = "Illegal kind ({}) of `FlockNode` (ID=`{}`)."

        if node is None:
            node = flock.leader
        elif isinstance(node, FlockNode):
            node = node
        else:
            raise ValueError

        match flock.get_kind(node):
            case FlockNodeKind.LEADER | FlockNodeKind.AGGREGATOR:
                if self.debug_mode:
                    # return self._debug_aggr_job(node) # FIXME
                    return self._aggr_job(node)
                else:
                    return self._aggr_job(node)

            case FlockNodeKind.WORKER:
                assert parent is not None
                # (^^^) avoids mypy issue which won't naturally occur with valid Flock topo
                if self.debug_mode:
                    return self._debug_worker_job(node, parent)
                else:
                    return self._worker_job(node, parent)

            case _:
                kind = flock.get_kind(node)
                idx = node.idx
                raise ValueError(value_err_template.format(kind, idx))

    def _aggr_job(self, node: FlockNode) -> Future[Result]:
        aggr_state = AggrState(node.idx)
        self.strategy.cli_worker_selection(aggr_state, list(self.flock.children(node)))
        # FIXME: This (^^^) shouldn't be run on the aggregator
        children_futures = [
            self.step(node=child, parent=node) for child in self.flock.children(node)
        ]

        # This partial function (`subtree_done_cbk`) will perform the aggregation only
        # when all futures in `children_futures` has completed. This partial function
        # will be added as a callback which is run after the completion of each child
        # future. But, it will only perform aggregation once since only the last future
        # to be completed will activate the conditional.
        future: Future[Result] = Future()
        subtree_done_cbk = functools.partial(
            self.aggr_callback,
            future,
            children_futures,
            node,
            self.runtime,
            self.strategy,
        )
        for child_fut in children_futures:
            child_fut.add_done_callback(subtree_done_cbk)

        return future

    def _debug_aggr_job(self, node: FlockNode) -> Future[Result]:
        raise NotImplementedError

    def _worker_job(self, node: FlockNode, parent: FlockNode) -> Future[Result]:
        data = self.fetch_worker_data(node)
        return self.runtime.submit(
            local_training_job,
            node,
            parent=parent,
            module=self.global_module,
            strategy=self.strategy,
            dataset=self.runtime.proxy(data),
            module_state_dict=self.runtime.proxy(self.state_dict),
        )

    def _debug_worker_job(self, node: FlockNode, parent: FlockNode) -> Future[Result]:
        return self.runtime.submit(
            debug_training_job,
            node,
            parent=parent,
            module=self.global_module,
            strategy=self.strategy,
        )
