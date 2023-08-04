import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import lightning as L
import torch
from numpy.random import RandomState
from pandas import DataFrame
from tqdm import tqdm

from flox.aggregator import SynchAggregatorLogicInterface
from flox.core import fork_module
from flox.fit.tasks import launch_local_fitting_task
from flox.worker import WorkerLogicInterface


def _sync_federated_fit(
        global_module: L.LightningModule,
        aggr: SynchAggregatorLogicInterface,
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

    while not aggr.stop_condition(state):
        state["curr_round"] += 1

        # Perform random client selection and submit "local" fitting tasks.
        futures = []
        with ThreadPoolExecutor(max_workers=n_threads) as exc:
            for w in aggr.on_worker_select(workers):
                # TODO: Investigate the training/testing accuracy w.r.t. `fork_module`. Make sure the modules
                #       aren't having some implicit issue with intermediary tensor references.
                fut = exc.submit(launch_local_fitting_task, workers[w], fork_module(global_module))
                futures.append(fut)

        # Retrieve the "locally" updated the models and record training results from workers.
        results = [fut.result() for fut in futures]
        now = datetime.datetime.now()
        for (worker_id, worker_results) in results:
            train_results["time"].append(now)
            train_results["round"].append(state["curr_round"])
            for key, val in worker_results.items():
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

        pbar.update()

    train_results = DataFrame.from_dict(train_results)
    test_results = DataFrame.from_dict(test_results)
    return train_results, test_results
