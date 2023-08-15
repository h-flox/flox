import datetime
import torch

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from pandas import DataFrame
from torch.utils.data import DataLoader, Subset
from torch import nn
from typing import Mapping, Any
from tqdm import tqdm

from flox.aggregator.base import SimpleAvg
from flox.flock import Flock, FlockNodeID, FlockNode


def flock_fit(
    flock: Flock,
    module_cls: type[nn.Module],
    datasets: Mapping[FlockNodeID, Subset],
    num_global_rounds: int,
) -> tuple[nn.Module, DataFrame]:
    # TODO: We need to change this function to operate more as a tree
    #       traversal where aggregation occurs at each non-worker node.
    train_history = defaultdict(list)
    # test_history = defaultdict(list)
    global_module = module_cls()

    for global_round in tqdm(range(num_global_rounds), desc="fed_fit()"):
        selected_workers = list(flock.workers)
        futures = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            for worker in selected_workers:
                worker_dataset = datasets[worker.idx]
                fut = pool.submit(
                    _local_fit,
                    worker=worker,
                    worker_dataset=worker_dataset,
                    module_cls=module_cls,
                    module_state_dict=global_module.state_dict(),
                )
                futures.append(fut)

        _aggr_weights(flock, global_module, futures, train_history, global_round)

    train_history = DataFrame.from_dict(train_history)
    # test_history = DataFrame.from_dict(test_history)
    return global_module, train_history  # , test_history


def _aggr_weights(
    flock: Flock,
    global_module: nn.Module,
    futures: list[Future],
    train_history: dict[str, Any],
    curr_round: int,
):
    # Collect the results from the endpoints (for now, we assume no failure).
    results = [fut.result() for fut in futures]

    # Do the aggregation across all the results.
    for res in results:
        # First, add the round number to the `train_history` separately to avoid duplicates.
        name, value = next(iter(res["local_history"].items()))
        if isinstance(value, list):
            train_history["round"].extend([curr_round] * len(value))
        else:
            train_history["round"].append(curr_round)

        # Then, add the key-value pairs returned from local fitting.
        for name, value in res["local_history"].items():
            if isinstance(value, list):
                train_history[name].extend(value)
            else:
                train_history[name].append(value)

    # Grab the module weights in order to perform aggregation.
    local_module_weights = {res["worker"]: res["module_state_dict"] for res in results}
    avg_weights = SimpleAvg()(global_module, local_module_weights)
    global_module.load_state_dict(avg_weights)


def _local_fit(
    worker: FlockNode,
    worker_dataset: Subset,
    module_cls: type[nn.Module],
    module_state_dict: dict[str, torch.Tensor],
):
    history = defaultdict(list)

    local_module = module_cls()
    local_module.load_state_dict(module_state_dict)
    optimizer = torch.optim.SGD(local_module.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(worker_dataset, batch_size=32)

    for epoch in range(3):
        running_loss, last_loss = 0, 0
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            preds = local_module(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        history["worker"].append(worker.idx)
        history["train/loss"].append(running_loss / len(train_loader))
        history["epoch"].append(epoch)
        history["time"].append(str(datetime.datetime.now()))

    return {
        "worker": worker,
        "module_state_dict": local_module.state_dict(),
        "local_history": history,
    }
