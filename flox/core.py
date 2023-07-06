import copy
import os
import torch
import lightning as L
import random

from concurrent.futures import ThreadPoolExecutor
from numpy.random import RandomState
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Any, Optional, Union

from flox._aggr import AggrLogic
from flox._worker import WorkerLogic
from main import MnistWorkerLogic

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


def local_fit(
        logic: WorkerLogic,
        module: L.LightningModule
) -> tuple[Any, L.LightningModule]:
    data_loader = DataLoader(logic.on_data_fetch(), batch_size=32, shuffle=True)
    module = logic.on_module_fit(module, data_loader)
    return logic.idx, module


def create_workers(num: int) -> dict[str, WorkerLogic]:
    workers = {}
    for i in range(num):
        n_samples = random.randint(100, 250)
        indices = random.sample(range(10_000), k=n_samples)
        workers[f"Worker-{i}"] = MnistWorkerLogic(indices=list(indices))
    return workers


def federated_fit(
        global_module: L.LightningModule,
        aggr: AggrLogic,
        workers: dict[str, WorkerLogic],
        global_rounds: int,
        test: bool = False,
        random_state: Optional[RandomState] = None,
        **kwargs
):
    if random_state is None:
        random_state = RandomState()

    results = {}
    # Below is the execution of the Global Aggregation Rounds. Each round consists of the following steps:
    #   (1) clients are selected to do local training
    #   (2) selected clients do local training and send back their locally-trained model udpates
    #   (3) the aggregator then aggregates the model updates using FedAvg
    #   (4) the aggregator tests/evaluates the new global model
    #   (5) the loop repeats until all global rounds have been done.
    for gr in range(global_rounds):
        print(f">> Starting global round ({gr + 1}/{global_rounds}).")

        # Perform random client selection and submit "local" fitting tasks.
        # size = max(1, int(args.participation_frac * len(workers)))
        size = len(workers)
        selected_workers = random_state.choice(list(workers), size=size, replace=False)
        futures = []
        with ThreadPoolExecutor(max_workers=size) as exc:
            for w in selected_workers:
                fut = exc.submit(local_fit, workers[w], copy.deepcopy(global_module))
                print(f"Job submitted to worker node {workers[w]}.")
                futures.append(fut)

        # Retrieve the "locally" updated the models and do aggregation.
        updates = [fut.result() for fut in futures]
        updates = {endp: module for (endp, module) in updates}
        # avg_weights = fedavg(global_module, updates, workers)
        aggr_weights = aggr.on_model_aggr(global_module, workers, updates)
        global_module.load_state_dict(aggr_weights)

        # Evaluate the global model performance.
        if test:
            results["metrics"] = aggr.on_module_eval(global_module)

    results["module"] = global_module
    return results
