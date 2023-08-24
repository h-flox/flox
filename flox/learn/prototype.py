from __future__ import annotations

import datetime
import functools

import pandas as pd
import torch
import torch.utils.data as torch_data

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from collections import defaultdict
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm
from typing import Any, Literal, Mapping, Optional, TypeAlias

from flox.aggregator.base import SimpleAvg
from flox.flock import Flock, FlockNodeID, FlockNode, FlockNodeKind
from flox.utils.misc import extend_dicts

Kind: TypeAlias = Literal["async", "sync"]
Where: TypeAlias = Literal["local", "globus_compute"]


def federated_fit(
    flock: Flock,
    module_cls: type[nn.Module],
    datasets: Mapping[FlockNodeID, Subset],
    num_global_rounds: int,
    kind: Kind = "sync",
    where: Where = "local",
) -> pd.DataFrame:
    """

    Args:
        flock ():
        module_cls ():
        datasets ():
        num_global_rounds ():
        kind ():
        where ():

    Returns:

    """
    if kind == "sync":
        return _sync_federated_fit(flock, module_cls, datasets, num_global_rounds)
    elif kind == "async":
        raise NotImplementedError()
    else:
        raise ValueError(
            "Illegal value for argument `kind`. Must be either 'sync' or 'async'."
        )


def set_parent_future(parent_future, child_future):
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


def _sync_federated_fit(
    flock: Flock,
    module_cls: type[nn.Module],
    datasets: Mapping[FlockNodeID, torch_data.Dataset | torch_data.Subset],
    num_global_rounds: int,
):
    executor = ThreadPoolExecutor(max_workers=1)
    module = module_cls()
    dataframes = []

    for gr in tqdm(range(num_global_rounds), desc="federated_fit::sync"):
        round_fut = _sync_traverse(
            executor=executor,
            flock=flock,
            module_cls=module_cls,
            module_state_dict=module.state_dict(),
            datasets=datasets,
            node=flock.leader,
            parent=None,
        )
        round_results = round_fut.result()
        round_history = round_results["history"]
        round_df = pd.DataFrame.from_dict(round_history)
        round_df["round"] = gr
        dataframes.append(round_df)

        # Apply the aggregated weights to the leader's global module for the next round.
        module.load_state_dict(round_results["state_dict"])

    results = pd.concat(dataframes)
    return results


def _sync_traverse(
    executor: Executor,
    flock: Flock,
    module_cls: type[nn.Module],
    module_state_dict: dict[str, torch.Tensor],
    datasets: Mapping[FlockNodeID, torch_data.Dataset | torch_data.Subset],
    node: FlockNode,
    parent: Optional[FlockNode] = None,
):
    if flock.topo.nodes[node.idx]["kind"] is FlockNodeKind.WORKER:
        future = executor.submit(
            _worker_node_fn,
            node,
            parent,
            module_cls,
            module_state_dict,
            datasets[node.idx],
        )
        return future
    else:
        children_futures = [
            _sync_traverse(
                executor, flock, module_cls, module_state_dict, datasets, child, node
            )
            for child in flock.children(node)
        ]
        aggregator_fut = Future()
        subtree_done_cbk = functools.partial(
            _aggr_node_cbk,
            executor,
            children_futures,
            node,
            aggregator_fut,
        )
        for child_fut in children_futures:
            child_fut.add_done_callback(subtree_done_cbk)

        # print(f"Aggregator({node.idx}):")
        # for i, child_fut in enumerate(children_futures):
        #     print(f"{i} :: child_fut.done() -> {child_fut.done()}")
        return aggregator_fut


def _aggr_node_cbk(
    executor: Executor,
    children_futures: list[Future],
    node: FlockNode,
    aggregator_fut: Future,
    child_fut_to_resolve: Future,
):
    if all([fut.done() for fut in children_futures]):
        child_results = [fut.result() for fut in children_futures]

        future = executor.submit(_aggr_node_fn, node, child_results)
        custom_callback = functools.partial(set_parent_future, aggregator_fut)
        result = future.add_done_callback(
            custom_callback
        )  # NOTE: Output here is a result.
        # print(result)


def _aggr_node_fn(node: FlockNode, results: list[dict[str, Any]]):
    # print(f"Running aggregation on {node=}")

    local_module_weights = {res["node/idx"]: res["state_dict"] for res in results}
    global_module = None  # NOTE: For now, this is fine because `SimpleAvg` doesn't do anything with module.
    avg_state_dict = SimpleAvg()(global_module, local_module_weights)
    # NOTE: The key-value scheme returned by aggregators has to match with workers.
    histories = (res["history"] for res in results)
    return {
        "node/idx": node.idx,
        "node/kind": node.kind,
        "state_dict": avg_state_dict,
        "history": extend_dicts(*histories),
    }


def _worker_node_fn(
    node: FlockNode,
    parent: FlockNode,
    module_cls: type[nn.Module],
    module_state_dict: dict[str, torch.Tensor],
    dataset: torch_data.Dataset | torch_data.Subset,
    num_epochs: int = 3,
    batch_size: int = 32,
    shuffle: bool = True,
    lr: float = 1e-3,
):
    state_dict, history = _worker_local_fit(
        node, parent, module_cls, module_state_dict, dataset
    )
    # NOTE: The key-value scheme returned by workers has to match with aggregators.
    # print(f">>> Finished `_worker_local_fit()` for {node=}")
    return {
        "node/idx": node.idx,
        "node/kind": node.kind,
        "state_dict": state_dict,
        "history": history,
    }


def _worker_local_fit(
    node: FlockNode,
    parent: FlockNode,
    module_cls: type[nn.Module],
    module_state_dict: dict[str, torch.Tensor],
    dataset: torch_data.Dataset | torch_data.Subset,
    num_epochs: int = 3,
    batch_size: int = 32,
    shuffle: bool = True,
    lr: float = 1e-3,
):
    # print(f">>> Starting `_worker_local_fit()` for {node=}")

    module = module_cls()
    module.load_state_dict(module_state_dict)
    optimizer = torch.optim.SGD(module.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = defaultdict(list)
    train_loader = torch_data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    for epoch in range(num_epochs):
        running_loss, last_loss = 0, 0
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            preds = module(inputs)
            loss = criterion(preds, targets)
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

    return module.state_dict(), history
