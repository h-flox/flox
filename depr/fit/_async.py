import datetime
import lightning as L
import torch

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from numpy.random import RandomState
from pandas import DataFrame
from tqdm import tqdm
from typing import Optional

from depr.aggregator.asynch.base import AsynchAggregatorLogic
from depr.core import fork_module
from depr.fit.tasks import launch_local_fitting_task
from depr.worker import WorkerLogicInterface


def _async_federated_fit(
    global_module: L.LightningModule,
    aggr: AsynchAggregatorLogic,
    workers: dict[str, WorkerLogicInterface],
    global_rounds: int,
    test: bool = False,
    n_threads: int = 4,
    random_state: Optional[RandomState] = None,
    **kwargs
) -> tuple[DataFrame, DataFrame]:
    train_results = defaultdict(list)
    test_results = defaultdict(list)
    state = {"curr_round": 0, "total_rounds": global_rounds}
    pbar = tqdm(total=state["total_rounds"], desc="federated_fit")

    executor = ThreadPoolExecutor(max_workers=n_threads)
    fitting_rounds_for_worker = defaultdict(int)
    futures = [
        executor.submit(launch_local_fitting_task, worker, fork_module(global_module))
        for worker in workers
    ]

    while futures:
        done, futures = wait(futures, return_when=FIRST_COMPLETED)
        futures = list(futures)
        worker_id, worker_result = done.pop().result()

        # Store results of training.
        now = str(datetime.datetime.now())
        train_results["worker_id"].append(worker_id)
        train_results["time"].append(now)
        train_results["round"].append(state["curr_round"])
        for k, v in worker_result.items():
            if k == "module":
                continue
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(v, torch.Tensor):
                        vv = vv.item()
                    train_results[kk].append(vv)
            else:
                train_results[k].append(v)

        # Submit new training job to the endpoint that just completed (if appropriate).
        fitting_rounds_for_worker[worker_id] += 1
        if fitting_rounds_for_worker[worker_id] < global_rounds:
            fut = executor.submit(
                launch_local_fitting_task, worker_id, fork_module(global_module)
            )
            futures.append(fut)

        # Perform model aggregation.
        worker_module = worker_result["module"]
        aggr_weights = aggr.on_module_aggregate(global_module, worker_id, worker_module)
        global_module.load_state_dict(aggr_weights)

        # Evaluate the global model performance.
        if test:
            now = datetime.datetime.now()
            test_metrics = aggr.on_module_eval(global_module)
            for metric, value in test_metrics:
                test_results["time"].append(now)
                test_metrics["metric"].append(metric)
                test_metrics["value"].append(value)

        pbar.update()

    train_results = DataFrame.from_dict(train_results)
    test_results = DataFrame.from_dict(test_results)
    return train_results, test_results
