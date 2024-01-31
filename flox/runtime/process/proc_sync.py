import functools
from concurrent.futures import Future

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from flox.data import FloxDataset
from flox.flock import Flock, FlockNode, FlockNodeKind
from flox.flock.states import FloxAggregatorState
from flox.nn import FloxModule
from flox.runtime.jobs import aggregation_job, local_training_job, debug_training_job
from flox.runtime.launcher import Launcher
from flox.runtime.process.proc import BaseProcess
from flox.runtime.result import Result
from flox.runtime.transfer.base import BaseTransfer
from flox.runtime.utils import set_parent_future
from flox.strategies import Strategy


# def dict_to_node_state(d: dict[str, Any]) -> NodeState:
#


def aggr_cbk(
    parent_future: Future,
    children_futures: list[Future],
    node: FlockNode,
    launcher: Launcher,
    transfer: BaseTransfer,
    strategy: Strategy,
    _: Future,
):
    if all([child_future.done() for child_future in children_futures]):
        # TODO: We need to add error-handling for cases when the
        #       `TaskExecutionFailed` error from Globus-Compute is thrown.
        children_results = [child_future.result() for child_future in children_futures]
        future = launcher.submit(
            aggregation_job,
            node,
            transfer=transfer,
            strategy=strategy,
            results=children_results,
        )
        aggr_done_callback = functools.partial(set_parent_future, parent_future)
        future.add_done_callback(aggr_done_callback)


class SyncProcess(BaseProcess):
    def __init__(
        self,
        flock: Flock,
        num_global_rounds: int,
        launcher: Launcher,
        module: FloxModule,
        dataset: FloxDataset,
        transfer: BaseTransfer,
        strategy: Strategy | str,
    ):
        self.flock = flock
        self.launcher = launcher
        self.num_global_rounds = num_global_rounds
        self.global_module = module
        self.transfer = transfer
        self.strategy = strategy
        self.dataset = dataset

        self.aggr_callback = aggr_cbk
        self.state_dict = None
        self.debug_mode = False

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
        progress_bar = tqdm(total=self.num_global_rounds, desc="federated_fit::sync")
        for round_num in range(self.num_global_rounds):
            self.state_dict = self.global_module.state_dict()
            future = self.traverse()
            update = future.result()
            history = update.history
            history["round"] = round_num
            dataframes.append(history)
            self.global_module.load_state_dict(update.state_dict)
            progress_bar.update()

        history = pd.concat(dataframes)
        return self.global_module, history

    def traverse(
        self,
        node: FlockNode | None = None,
        parent: FlockNode | None = None,
    ) -> Future:
        flock = self.flock
        node = flock.leader if node is None else node

        match flock.get_kind(node):
            case FlockNodeKind.LEADER | FlockNodeKind.AGGREGATOR if not self.debug_mode:
                return self._aggr_job(node)
            case FlockNodeKind.LEADER | FlockNodeKind.AGGREGATOR if self.debug_mode:
                return self._debug_aggr_job(node)
            case FlockNodeKind.WORKER if not self.debug_mode:
                return self._worker_job(node, parent)
            case FlockNodeKind.WORKER if self.debug_mode:
                return self._debug_worker_job(node, parent)
            case _:
                raise ValueError(
                    f"Illegal kind ({flock.get_kind(node)=}) of `FlockNode` (ID=`{node.idx}`)."
                )

    def _aggr_job(self, node: FlockNode) -> Future[Result]:
        aggr_state = FloxAggregatorState(node.idx)
        self.strategy.agg_worker_selection(aggr_state, list(self.flock.children(node)))
        children_futures = [
            self.traverse(node=child, parent=node)
            for child in self.flock.children(node)
        ]

        future: Future[Result] = Future()
        # This partial function (`subtree_done_cbk`) will perform the aggregation only
        # when all futures in `children_futures` has completed. This partial function
        # will be added as a callback which is run after the completion of each child
        # future. But, it will only perform aggregation once since only the last future
        # to be completed will activate the conditional.
        subtree_done_cbk = functools.partial(
            self.aggr_callback,
            future,
            children_futures,
            node,
            self.launcher,
            self.transfer,
            self.strategy,
        )
        for child_fut in children_futures:
            child_fut.add_done_callback(subtree_done_cbk)

        return future

    def _debug_aggr_job(self, node: FlockNode, parent: FlockNode) -> Future:
        # TODO: Implement this.
        pass

    def _worker_job(self, node: FlockNode, parent: FlockNode) -> Future:
        data = self.fetch_worker_data(node)
        # TODO: Check if we can just do `load` here (^^^).
        return self.launcher.submit(
            local_training_job,
            node,
            parent=parent,
            dataset=self.transfer.proxy(data),
            module=self.global_module,
            module_state_dict=self.transfer.proxy(self.state_dict),
            transfer=self.transfer,
            strategy=self.strategy,
        )

    def _debug_worker_job(
        self, node: FlockNode, parent: FlockNode
    ) -> Future:  # Future[Result]
        return self.launcher.submit(
            debug_training_job,
            node,
            parent=parent,
            transfer=self.transfer,
            module=self.global_module,
            strategy=self.strategy,
        )
