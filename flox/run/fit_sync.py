from __future__ import annotations

import functools
import pandas as pd
import torch

from concurrent.futures import Executor, Future
from tqdm import tqdm
from typing import Optional

from flox.flock import Flock, FlockNode, FlockNodeKind
from flox.flock.states import FloxAggregatorState
from flox.backends.launcher.base import FloxExecutor
from flox.backends.launcher import GlobusComputeExecutor
from flox.backends.launcher import LocalExecutor
from flox.run.jobs import local_training_job, aggregation_job, JobResult
from flox.run.utils import set_parent_future
from flox.strategies import Strategy
from flox.data import FloxDataset
from flox.typing import StateDict


def sync_federated_fit(
    flock: Flock,
    module_cls: type[torch.nn.Module],
    datasets: FloxDataset,
    num_global_rounds: int,
    strategy: Strategy | str = "fedsgd",
    executor: str = "thread",
    num_workers: int = 1,
) -> pd.DataFrame:
    """Synchronous federated learning implementation.

    This implementation traverses the provide Flock network using DFS. During traversal,
    it will launch appropriate tasks on the aggregator and worker nodes.

    Args:
        flock (Flock): The topology of nodes for the FL process.
        module_cls (type[torch.nn.Module]): The class for the PyTorch Module to train.
        datasets (FloxDataset): Datasets for workers to train.
        num_global_rounds (int): Total number of global (aggregation) rounds are performed
            during the entire FL process.
        strategy (Strategy | str): The strategy logic to use during the FL process.
            It is recommended that you pass a direct instance of a class that extends
            ``Strategy``.  If the provided argument is of type ``str``, then the ``Strategy``
            base class will check its registry for a registered ``Strategy`` of that name
            (using the default parameters).
        executor (str): Which executor to launch tasks with, defaults to "thread" (i.e.,
            ``ThreadPoolExecutor``).
        num_workers (int): Number of workers to execute tasks.

    Returns:
        Results from the FL process.
    """
    if executor == "thread" or executor == "process":
        executor = LocalExecutor(executor, num_workers)
    elif executor == "globus_compute":
        executor = GlobusComputeExecutor()

    # executor = ThreadPoolExecutor(num_workers)
    if isinstance(strategy, str):
        strategy = Strategy.get_strategy(strategy)()

    df_list = []
    global_module = module_cls()
    prog_bar = tqdm(total=num_global_rounds, desc="federated_fit::sync")

    for round_no in range(num_global_rounds):
        # Launch the tasks recursively starting with the aggregation task on the
        # leader of the Flock.
        rnd_future = sync_flock_traverse(
            executor,
            flock=flock,
            node=flock.leader,
            module_cls=module_cls,
            module_state_dict=global_module.state_dict(),
            datasets=datasets,
            strategy=strategy,
            parent=None,
        )

        # Collect results from the aggregated future.
        round_update: JobResult = rnd_future.result()
        round_df = round_update.history  # pd.DataFrame.from_dict(round_update.history)
        round_df["round"] = round_no
        df_list.append(round_df)

        # Apply the aggregated weights to the leader's global module for the next round.
        global_module.load_state_dict(round_update.state_dict)
        prog_bar.update()

    return pd.concat(df_list)


def sync_flock_traverse(
    executor: FloxExecutor,
    flock: Flock,
    node: FlockNode,
    module_cls: type[torch.nn.Module],
    module_state_dict: StateDict,
    datasets: FloxDataset,
    strategy: Strategy,
    parent: Optional[FlockNode] = None,
) -> Future[JobResult]:
    """
    Launches an aggregation task on the provided ``FlockNode`` and the appropriate tasks
    for its child nodes.

    Returns:

    """
    # If the current node is a worker node, then Launch the LOCAL FITTING job.
    if flock.get_kind(node) is FlockNodeKind.WORKER:
        hyper_params = {}
        return executor.submit(
            local_training_job,
            node,
            parent=parent,
            strategy=strategy,
            module_cls=module_cls,
            module_state_dict=module_state_dict,
            dataset=datasets[node.idx],
            **hyper_params,
        )

    # Otherwise, launch the recursive AGGREGATION job.
    state = FloxAggregatorState()
    children_nodes = list(flock.children(node))
    strategy.agg_worker_selection(state, children_nodes)
    children_futures = []

    # Launch the appropriate jobs on the children nodes.
    for child in children_nodes:
        future = sync_flock_traverse(
            executor,
            flock=flock,
            module_cls=module_cls,
            module_state_dict=module_state_dict,
            datasets=datasets,
            strategy=strategy,
            node=child,
            parent=node,
        )
        children_futures.append(future)

    # Initialize the Future that this aggregator will return to its parent.
    future = Future()
    subtree_done_callback = functools.partial(
        aggregation_callback,
        executor,
        children_futures,
        strategy,
        node,
        future,
    )

    for child_fut in children_futures:
        child_fut.add_done_callback(subtree_done_callback)

    return future


def aggregation_callback(
    executor: Executor,
    children_futures: list[Future],
    strategy: Strategy,
    node: FlockNode,
    parent_future: Future,
    child_future_to_resolve: Future,
) -> None:
    """
    Callback that is used to set up when the children futures are completed before aggregation.

    Args:
        executor (FloxExecutor):
        children_futures (list[Future]):
        strategy (Strategy):
        node (FlockNode):
        parent_future (Future):
        child_future_to_resolve (Future):
    """
    if all([future.done() for future in children_futures]):
        children_results = [future.result() for future in children_futures]
        future = executor.submit(
            aggregation_job, node, strategy=strategy, results=children_results
        )
        aggregation_done_callback = functools.partial(set_parent_future, parent_future)
        future.add_done_callback(aggregation_done_callback)
