import torch

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from typing import Any, Optional

from flox.aggregator.base import SimpleAvg
from flox.flock.flock import Flock
from flox.worker.trainer.torch import evaluate, local_fitting_task


def fit(
    flock,
    module_cls: type[nn.Module],
    num_rounds: int,
    test_dataset: Optional[Dataset] = None,
    device: Optional[torch.device] = None,
    prog_bar: bool = True,
) -> tuple[nn.Module, dict, dict]:
    """DEPRECATED.

    Args:
        flock (Flock):
        module_cls (type[nn.Module]): Class of model to train.
        num_rounds (int): Number of global aggregation rounds.
        test_dataset (Optional[Dataset]): Test dataset by the aggregator.
        device (Optional[torch.Device]):  The device to test on (e.g., GPU, CPU, TPU, MPS).
        prog_bar (bool): Display a progress bar if True.

    Returns:
        tuple[nn.Module, dict, dict]: The trained model and train/test results.
    """
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


# TODO: Most of this code needs to be converted into the aggregation launch script
#       It will be submitted to each aggregation endpoint with a list of their children.
#       The endpoint that launches each aggregation point will be where the submitted task
#       returns. For this, let's refer to Yadu's implementation.
def _sync_federated_fit(
    flock: Flock,
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
                    local_fitting_task,
                    worker=worker,
                    worker_dataset=worker_dataset,
                    module_cls=module_cls,
                    module_state_dict=module.state_dict(),
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    device=device,
                )
                futures.append(fut)

        # Launch the aggregation task on the aggregation endpoint.
        aggregation_task(flock, module, futures, train_history, curr_round)
        curr_round += 1
        prog_bar.update()

    # Evaluate the global module performance.
    if test_dataset:
        test_dataloader = DataLoader(test_dataset, batch_size=4)
        test_metrics = evaluate(module, test_dataloader)
        for metric, value in test_metrics.items():
            test_history[metric].append(value)

    return module, train_history, test_history


def aggregation_task(
    flock: dict[int, Subset],
    module: nn.Module,
    futures: list[Future],
    train_history: dict[str, Any],
    curr_round: int,
):
    # Here, we collect the results from the endpoints (for now, we assume no failure).
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
    avg_weights = SimpleAvg()(module, local_module_weights)
    module.load_state_dict(avg_weights)
