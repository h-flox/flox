import datetime
import torch

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from typing import Optional


def fit(
    flock,
    module_cls: type[nn.Module],
    num_rounds: int,
    test_dataset: Optional[Dataset] = None,
    device: Optional[torch.device] = None,
    prog_bar: bool = True,
):
    runner = "sync"
    if runner == "sync":
        module, train_history, test_history = _sync_federated_fit(
            flock, module_cls, num_rounds, test_dataset, device
        )
    elif runner == "async":
        module, train_history, test_history = 0, 0, 0
    else:
        raise ValueError("Illegal value for arg `runner`.")

    return module, train_history, test_history


def _sync_federated_fit(
    flock: dict[int, Subset],
    module_cls,
    num_rounds,
    test_dataset: Optional[Dataset] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    max_epochs: int = 2,
):
    if device is None:
        device = torch.device("cpu")

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    curr_round = 0
    prog_bar = tqdm(total=num_rounds, desc="federated_fit")
    module = module_cls()

    while curr_round < num_rounds:
        # Do client selection.
        # selected_workers = flock.workers()
        selected_workers = list(flock.keys())

        # Launch the local fitting tasks/jobs on the endpoints.
        futures = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            for worker in selected_workers:
                worker_dataset = flock[worker]
                fut = pool.submit(
                    _local_fitting_task,
                    worker=worker,
                    worker_dataset=worker_dataset,
                    module_cls=module_cls,
                    module_state_dict=module.state_dict(),
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    device=device,
                )
                futures.append(fut)

        # Collect the results from the endpoints (for now, we assume no failure).
        results = [fut.result() for fut in futures]
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
        local_module_weights = [res["module_state_dict"] for res in results]
        avg_weights = simple_avg(module, flock, local_module_weights)
        module.load_state_dict(avg_weights)

        curr_round += 1
        prog_bar.update()

    # Evaluate the global module performance.
    if test_dataset:
        test_dataloader = DataLoader(test_dataset, batch_size=4)
        test_metrics = evaluate(module, test_dataloader)
        for metric, value in test_metrics.items():
            test_history[metric].append(value)

    return module, train_history, test_history


def _local_fitting_task(
    worker,
    worker_dataset,
    module_cls,
    module_state_dict,
    batch_size: int = 4,
    max_epochs: int = 2,
    device: Optional[torch.device] = None,
):
    module = module_cls()
    module.to(device)
    module.load_state_dict(module_state_dict)

    history = defaultdict(list)
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(worker_dataset, batch_size=batch_size)

    for epoch in range(max_epochs):
        running_loss, last_loss = 0.0, 0.0
        for batch_nb, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = module(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        history["worker"].append(worker)
        history["train/loss"].append(running_loss / len(train_loader))
        history["epoch"].append(epoch)
        history["time"].append(str(datetime.datetime.now()))

    return {
        "worker": worker,
        "module_state_dict": module.state_dict(),
        "local_history": history,
    }


def simple_avg(module, workers, local_module_weights):
    with torch.no_grad():
        avg_module_weights = {}
        nk = 1 / len(workers)
        for state_dict in local_module_weights:
            for name, value in state_dict.items():
                if name not in avg_module_weights:
                    avg_module_weights[name] = nk * torch.clone(value)
                else:
                    avg_module_weights[name] += nk * torch.clone(value)
    return avg_module_weights


def evaluate(module, test_dataloader, device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cpu")

    loss_fn = nn.CrossEntropyLoss()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    module.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch in test_dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = module(inputs)
            test_loss += loss_fn(outputs, targets).item()
            correct += (outputs.argmax(1) == targets).sum().item()

    test_loss /= num_batches
    correct /= size

    return {"test_loss": test_loss, "test_acc": correct}
