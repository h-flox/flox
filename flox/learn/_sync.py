from __future__ import annotations

import datetime
import functools
import pandas as pd
import torch
import torch.utils.data as torch_data

from collections import defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, Future
from tqdm import tqdm
from typing import Mapping, NewType, Optional, Union, Any

from flox.flock import Flock, FlockNode, FlockNodeID, FlockNodeKind
from flox.flock.states import FloxAggregatorState, FloxWorkerState
from flox.learn.backends.base import FloxExecutor
from flox.learn.backends.globus import GlobusComputeExecutor
from flox.learn.backends.local import LocalExecutor
from flox.learn.update import TaskUpdate
from flox.strategies import Strategy
from flox.utils.data import FederatedDataset
from flox.typing import StateDict
from flox.utils.misc import extend_dicts


def sync_federated_fit(
    flock: Flock,
    module_cls: type[torch.nn.Module],
    datasets: FederatedDataset,
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
        datasets (FederatedDataset): Datasets for workers to train.
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
    if executor == "thread":
        executor = LocalExecutor("thread", num_workers)
    elif executor == "process":
        executor = LocalExecutor("pool", num_workers)
    elif executor == "globus_compute":
        executor = GlobusComputeExecutor()

    # executor = ThreadPoolExecutor(num_workers)
    if isinstance(strategy, str):
        strategy = Strategy.get_strategy(strategy)()

    df_list = []
    global_module = module_cls()
    prog_bar = tqdm(total=num_global_rounds, desc="federated_fit::sync")

    for rnd in range(num_global_rounds):
        # Launch the tasks recursively starting with the aggregation task on the
        # leader of the Flock.
        rnd_future = flock_traverse(
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
        round_update: TaskUpdate = rnd_future.result()
        round_df = pd.DataFrame.from_dict(round_update.history)
        round_df["round"] = rnd
        df_list.append(round_df)

        # Apply the aggregated weights to the leader's global module for the next round.
        global_module.load_state_dict(round_update.state_dict)
        prog_bar.update()

    return pd.concat(df_list).reset_index()


###############################################################################################


def flock_traverse(
    executor: FloxExecutor,
    flock: Flock,
    node: FlockNode,
    module_cls: type[torch.nn.Module],
    module_state_dict: StateDict,
    datasets: FederatedDataset,
    strategy: Strategy,
    parent: Optional[FlockNode] = None,
) -> Future:
    """
    Launches an aggregation task on the provided ``FlockNode`` and the appropriate tasks
    for its child nodes.

    Returns:

    """
    # Launch the local fitting function on the worker node.
    if flock.get_kind(node) is FlockNodeKind.WORKER:
        hyper_params = {}
        return executor.submit(
            _worker_task,
            node,
            parent=parent,
            strategy=strategy,
            module_cls=module_cls,
            module_state_dict=module_state_dict,
            dataset=datasets[node.idx],
            **hyper_params,
        )

    # Launch the recursive aggregation task to the children nodes.
    state = FloxAggregatorState()
    children_nodes = list(flock.children(node))
    strategy.agg_on_worker_selection(state, children_nodes)
    children_futures = []

    for child in children_nodes:
        future = flock_traverse(
            executor,
            flock=flock,
            module_cls=module_cls,
            module_state_dict=module_state_dict,
            datasets=datasets,
            strategy=strategy,
            node=child,
            parent=node,  # Note: Set the parent to this call's `node`.
        )
        children_futures.append(future)

    # Initialize the Future that this aggregator will return to its parent.
    future = Future()
    subtree_done_callback = functools.partial(
        _aggregation_callback,
        executor,
        children_futures,
        strategy,
        node,
        future,
    )

    for child_fut in children_futures:
        child_fut.add_done_callback(subtree_done_callback)

    return future


###############################################################################################


def _worker_task(
    node: FlockNode,
    parent: FlockNode,
    strategy: Strategy,
    module_cls: type[torch.nn.Module],
    module_state_dict: StateDict,
    dataset: torch_data.Dataset | torch_data.Subset,
    **train_hyper_params,
) -> TaskUpdate:
    # TODO: Generalize this so that users can provide a "Trainer" object with the training logic in a Lightning-like
    #       format.
    node_state, state_dict, history = _local_fitting(
        node,
        parent,
        strategy,
        module_cls,
        module_state_dict,
        dataset,
        **train_hyper_params,
    )
    return TaskUpdate(node_state, node.idx, node.kind, state_dict, history)


def _local_fitting(
    node: FlockNode,
    parent: FlockNode,
    strategy: Strategy,
    module_cls: type[torch.nn.Module],
    module_state_dict: StateDict,
    dataset: Optional[torch_data.Dataset | torch_data.Subset] = None,
    **train_hyper_params,
):
    """Perform local training on a worker node.

    Args:
        node ():
        parent ():
        module_cls ():
        module_state_dict ():
        dataset ():
        **train_hyper_params ():

    Returns:

    """

    global_module = module_cls()
    global_module.load_state_dict(module_state_dict)
    node_state = FloxWorkerState(pre_local_train_model=global_module)

    num_epochs = train_hyper_params.get("num_epochs", 2)
    batch_size = train_hyper_params.get("batch_size", 32)
    lr = train_hyper_params.get("lr", 1e-3)
    shuffle = train_hyper_params.get("shuffle", True)

    local_module = module_cls()
    local_module.load_state_dict(module_state_dict)
    optimizer = torch.optim.SGD(local_module.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    node_state.post_local_train_model = local_module

    strategy.wrk_on_before_train_step(node_state, dataset=dataset)

    history = defaultdict(list)
    train_loader = torch_data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    for epoch in range(num_epochs):
        running_loss, last_loss = 0, 0
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            preds = local_module(inputs)
            loss = criterion(preds, targets)

            try:
                strategy.wrk_on_after_train_step(node_state, loss)
            except NotImplementedError:
                pass

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        history["node/idx"].append(node.idx)
        history["node/kind"].append(node.kind)
        history["parent/idx"].append(parent.idx)
        history["parent/kind"].append(parent.kind)
        history["train/loss"].append(running_loss / len(train_loader))
        history["epoch"].append(epoch)
        history["time"].append(str(datetime.datetime.now()))

    return node_state, local_module.state_dict(), history


###############################################################################################


def _aggregation_callback(
    executor: Executor,
    children_futures: list[Future],
    strategy: Strategy,
    node: FlockNode,
    parent_future: Future,
    child_future_to_resolve: Future,
) -> None:
    """

    Args:
        executor (FloxExecutor):
        children_futures (list[Future]):
        strategy (Strategy):
        node (FlockNode):
        node (FlockNode):
        parent_future (Future):
        child_future_to_resolve (Future):
    """
    if all([future.done() for future in children_futures]):
        child_updates = [future.result() for future in children_futures]
        future = executor.submit(
            _aggregation_task, node, strategy=strategy, updates=child_updates
        )
        aggregation_done_callback = functools.partial(_set_parent_future, parent_future)
        future.add_done_callback(aggregation_done_callback)


def _aggregation_task(
    node: FlockNode, strategy: Strategy, updates: list[TaskUpdate]
) -> TaskUpdate:
    """Aggregate the state dicts from each of the results.

    Args:
        node (FlockNode): The aggregator node.
        strategy (Strategy): ...
        updates (list[dicts[str, Any]]): Results from children of ``node``.

    Returns:
        dict[str, Any]: Aggregation results.
    """
    child_states, child_state_dicts = {}, {}
    for update in updates:
        idx = update.node_idx
        child_states[idx] = update.node_state
        child_state_dicts[idx] = update.state_dict

    node_state = FloxAggregatorState()
    avg_state_dict = strategy.agg_on_param_aggregation(
        node_state, child_states, child_state_dicts
    )

    histories = (update.history for update in updates)
    return TaskUpdate(
        node_state, node.idx, node.kind, avg_state_dict, extend_dicts(*histories)
    )


def _set_parent_future(parent_future: Future, child_future: Future) -> Any:
    """Sets the result of the `parent_future` to the result of its `child_future` and returns it.

    Args:
        parent_future (Future): The parent Future.
        child_future (Future): The child Future.

    Returns:
        The result of `child_future` which is now set to be the result of `parent_future`.
    """
    assert child_future.done()
    if child_future.exception():
        parent_future.set_exception(child_future.exception())
    else:
        result = child_future.result()
        try:
            parent_future.set_result(result)
        except Exception as ex:
            print(ex)  # TODO: Log this better.
        return result
