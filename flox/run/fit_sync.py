from __future__ import annotations

import functools
from concurrent.futures import Future
from typing import Literal, TypeAlias

import pandas as pd
from tqdm import tqdm

from flox.backends.launcher import GlobusComputeLauncher, LocalLauncher
from flox.backends.launcher.impl_base import Launcher
from flox.backends.transfer.base import BaseTransfer
from flox.backends.transfer.proxystore import ProxyStoreTransfer
from flox.data import FloxDataset
from flox.flock import Flock, FlockNode, FlockNodeKind
from flox.flock.states import FloxAggregatorState
from flox.nn import FloxModule
from flox.run.jobs import aggregation_job, local_training_job
from flox.run.utils import set_parent_future
from flox.strategies import Strategy
from flox.typing import StateDict

Where: TypeAlias = Literal["local", "globus_compute"]
Transfer: TypeAlias = BaseTransfer | ProxyStoreTransfer


def sync_federated_fit(
    flock: Flock,
    module_cls: type[FloxModule],
    datasets: FloxDataset,
    num_global_rounds: int,
    strategy: Strategy | str = "fedsgd",
    launcher: str = "thread",
    max_workers: int = 1,
    where: Where = "local",
) -> tuple[FloxModule, pd.DataFrame]:
    """Synchronous federated learning implementation.

    This implementation traverses the provide Flock network using DFS. During traversal,
    it will launch appropriate tasks on the aggregator and worker nodes.

    Args:
        flock (Flock): The topology of nodes for the FL process.
        module_cls (type[FloxModule]): The class for the PyTorch Module to train.
        datasets (FloxDataset): Datasets for workers to train.
        num_global_rounds (int): Total number of global (aggregation) rounds are performed
            during the entire FL process.
        strategy (Strategy | str): The strategy logic to use during the FL process.
            It is recommended that you pass a direct instance of a class that extends
            ``Strategy``.  If the provided argument is of type ``str``, then the ``Strategy``
            base class will check its registry for a registered ``Strategy`` of that name
            (using the default parameters).
        launcher (str): Which launcher to launch tasks with, defaults to "thread" (i.e.,
            ``ThreadPoolExecutor``).
        max_workers (int): Number of workers to execute tasks.
        where (Where): Where to launch jobs, defaults to "local".

    Returns:
        Results from the FL process.
    """
    transfer: Transfer
    launcher_instance: Launcher

    if launcher == "thread" or launcher == "process":
        launcher_instance = LocalLauncher(launcher, max_workers)
    elif launcher == "globus_compute":
        launcher_instance = GlobusComputeLauncher()

    if where == "local":
        transfer = BaseTransfer()
    else:
        transfer = ProxyStoreTransfer(flock=flock)

    if isinstance(strategy, str):
        strategy = Strategy.get_strategy(strategy)()

    df_list = []
    global_module = module_cls()
    prog_bar = tqdm(total=num_global_rounds, desc="federated_fit::sync")

    for round_no in range(num_global_rounds):
        # Launch the tasks recursively starting with the aggregation task on the
        # leader of the Flock.
        rnd_future = sync_flock_traverse(
            launcher_instance,
            transfer=transfer,
            flock=flock,
            node=flock.leader,
            module_cls=module_cls,
            module_state_dict=global_module.state_dict(),
            datasets=datasets,
            strategy=strategy,
            parent=None,
        )

        # Collect results from the aggregated future.
        # TODO: FIX: removed type definition JobResult for a test
        round_update = rnd_future.result()
        round_df = round_update.history  # pd.DataFrame.from_dict(round_update.history)
        round_df["round"] = round_no
        df_list.append(round_df)

        # Apply the aggregated weights to the leader's global module for the next round.
        global_module.load_state_dict(round_update.state_dict)
        prog_bar.update()

    return global_module, pd.concat(df_list)


def sync_flock_traverse(
    launcher: Launcher,
    transfer: Transfer,
    flock: Flock,
    node: FlockNode,
    module_cls: type[FloxModule],
    module_state_dict: StateDict,
    datasets: FloxDataset,
    strategy: Strategy,
    parent: FlockNode | None = None,
) -> Future:  # TODO: Fix
    """
    Launches an aggregation task on the provided ``FlockNode`` and the appropriate tasks
    for its child nodes.

    Args:
        launcher (Launcher): ...
        transfer (Transfer): ...
        flock (Flock): ...
        node (FlockNode): ...
        module_cls (type[FloxModule]): ...
        module_state_dict (StateDict): ...
        datasets (FloxDataset): ...
        strategy (Strategy): ...
        parent (Optional[FlockNode]): ...

    Returns:
        The ``Future[JobResult]`` of the current ``node`` (either a worker or an aggregator).
    """

    # If the current node is a worker node, then Launch the LOCAL FITTING job.
    if flock.get_kind(node) is FlockNodeKind.WORKER:
        if isinstance(transfer, ProxyStoreTransfer):
            dataset = transfer.proxy(datasets[node.idx])
        else:
            dataset = datasets[node.idx]

        return launcher.submit(
            local_training_job,
            node,
            transfer=transfer,
            parent=parent,
            strategy=strategy,
            module_cls=module_cls,
            module_state_dict=module_state_dict,
            dataset=dataset,
        )

    if isinstance(transfer, ProxyStoreTransfer):
        datasets = transfer.proxy(datasets)

    # Otherwise, launch the recursive AGGREGATION job.
    state = FloxAggregatorState(node.idx)
    children_nodes = list(flock.children(node))
    strategy.agg_worker_selection(state, children_nodes)
    children_futures = []

    # Launch the appropriate jobs on the children nodes.
    for child in children_nodes:
        future = sync_flock_traverse(
            launcher,
            transfer=transfer,
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
        launcher,
        children_futures,
        strategy,
        node,
        future,
        transfer=transfer,
    )

    for child_fut in children_futures:
        child_fut.add_done_callback(subtree_done_callback)

    return future


# TODO: We need to look into how to generalize this logic. Requiring all child futures complete before aggregating
#       is a good default. But there might be some cases where we want to aggregate after some time has elapsed.
def aggregation_callback(
    launcher: Launcher,
    children_futures: list[Future],
    strategy: Strategy,
    node: FlockNode,
    parent_future: Future,
    child_future_to_resolve: Future,
    transfer: Transfer,
) -> None:
    """
    Callback that requires all child futures to complete before the aggregation job is launched.

    Args:
        launcher (Launcher):
        children_futures (list[Future]):
        strategy (Strategy):
        node (FlockNode):
        parent_future (Future):
        child_future_to_resolve (Future):
        transfer (Transfer): ...
    """
    if all([future.done() for future in children_futures]):
        children_results = [future.result() for future in children_futures]
        future = launcher.submit(
            aggregation_job,
            node,
            transfer=transfer,
            strategy=strategy,
            results=children_results,
        )
        aggregation_done_callback = functools.partial(set_parent_future, parent_future)
        future.add_done_callback(aggregation_done_callback)
