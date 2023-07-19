import datetime
import lightning as L
import pandas as pd
import random
import torch

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from numpy.random import RandomState
from typing import Optional, Type

from flox.aggregator import AggregatorLogicInterface
from flox.tasks import launch_local_fitting_task
from flox.worker import WorkerLogicInterface


def fork_module(
        module: L.LightningModule
) -> L.LightningModule:
    cls = module.__class__
    forked_module = cls()
    forked_module.load_state_dict(module.state_dict())
    return forked_module


def create_workers(
        num: int,
        worker_logic: Type[WorkerLogicInterface]
) -> dict[str, WorkerLogicInterface]:
    workers = {}
    for idx in range(num):
        n_samples = random.randint(50, 250)
        indices = random.sample(range(60_000), k=n_samples)
        workers[f"Worker-{idx}"] = worker_logic(idx=idx, indices=list(indices))
    return workers


def federated_fit(
        global_module: L.LightningModule,
        aggr: AggregatorLogicInterface,
        workers: dict[str, WorkerLogicInterface],
        global_rounds: int,
        test: bool = False,
        n_threads: int = 4,
        random_state: Optional[RandomState] = None,
        **kwargs
):
    if random_state is None:
        random_state = RandomState()

    train_results = defaultdict(list)
    test_results = defaultdict(list)
    state = {"curr_round": 0, "total_rounds": global_rounds}

    while not aggr.stop_condition(state):
        state["curr_round"] += 1
        print(">> Starting global round ({}/{}).".format(
            state["curr_round"],
            state["total_rounds"]
        ))

        # Perform random client selection and submit "local" fitting tasks.
        futures = []
        with ThreadPoolExecutor(max_workers=n_threads) as exc:
            for w in aggr.on_worker_select(workers):
                fut = exc.submit(launch_local_fitting_task, workers[w], fork_module(global_module))
                # fut = exc.submit(launch_local_fitting_task, workers[w], copy.deepcopy(global_module))
                print(f"Job submitted to worker node {workers[w]}.")
                futures.append(fut)

        # Retrieve the "locally" updated the models and record training results from workers.
        results = [fut.result() for fut in futures]
        now = datetime.datetime.now()
        for res in results:
            train_results["time"].append(now)
            train_results["round"].append(state["curr_round"])
            for key, val in res[1].items():
                if key == "module":
                    continue
                if isinstance(val, dict):
                    for k, v in val.items():
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        train_results[k].append(v)
                else:
                    train_results[key].append(val)

        # Perform model aggregation.
        module_updates = {worker: payload["module"] for (worker, payload) in results}
        aggr_weights = aggr.on_module_aggregate(global_module, workers, module_updates)
        global_module.load_state_dict(aggr_weights)

        # Evaluate the global model performance.
        if test:
            now = datetime.datetime.now()
            test_metrics = aggr.on_module_evaluate(global_module)
            for metric, value in test_metrics:
                test_results["time"].append(now)
                test_metrics["metric"].append(metric)
                test_metrics["value"].append(value)

    return {
        "module": global_module,
        "train_results": pd.DataFrame.from_dict(train_results),
        "test_results": pd.DataFrame.from_dict(test_results)
    }
