from __future__ import annotations

import typing as t
from concurrent.futures import FIRST_COMPLETED, Future, wait
from copy import deepcopy
from dataclasses import dataclass, field

from torch.utils.data import Subset, TensorDataset

from flight.events import (
    FlightEventEnum,
    add_event_handler_to_obj,
    fire_event_handler_by_type,
)

from flight.jobs.protocols import Result
from flight.jobs.worker import worker_job, WorkerJobArgs
from flight.learning.module import TorchModule
from flight.learning.parameters import Params
from flight.runtime import Runtime

if t.TYPE_CHECKING:
    from flight.learning.parameters import Params
    from flight.strategies.strategy import Strategy
    from flight.system.topology import Topology  
    from flight.system.node import Node

class AsyncWorkflowEvents(FlightEventEnum):
    STARTED = "started"
    """
    ...
    """

    COMPLETED = "completed"
    AGGREGATION_COMPLETED = "aggregation_completed"
    WORKER_JOB_STARTED = "worker_job_started"
    WORKER_JOB_COMPLETED = "worker_job_completed"


@dataclass
class AsyncWorkflowState:
    global_params: Params
    worker_params: dict = field(default_factory=dict)
    worker_rounds: dict = field(default_factory=dict)
    completed_worker_jobs: int = 0  


class AsyncWorkflow:
    """
    Handles the asynchronous strategy for federated learning, managing the dispatching and
    aggregation of worker jobs.
    """
    def __init__(
        self,
        runtime: Runtime,
        topology: 'Topology',
        num_global_rounds: int,
        module: TorchModule,
        dataset: TensorDataset,
        strategy: 'Strategy',
        aggregation_policy: t.Optional[t.Callable[[t.Any, t.Optional[int]], None]] = None,
        worker_time_tracker=None,
    ):
        """
        DESCRIPTION

        Args:
            runtime (Runtime): ...
            ...
        """
        self.runtime = runtime
        self.topology = topology
        self.num_global_rounds = num_global_rounds
        self.module = module
        self.dataset = dataset
        self.strategy = strategy
        self.aggregation_policy = aggregation_policy
        self.worker_time_tracker = worker_time_tracker

        initial_params = self.module.get_params()
        self.state = AsyncWorkflowState(global_params=initial_params)

        for worker in self.topology.workers:
            self.state.worker_rounds[worker.idx] = 0
            self.state.worker_params[worker.idx] = deepcopy(initial_params)  

    def start(self) -> tuple[TorchModule, t.Any]:
        """Starts the asynchronous federated learning strategy by dispatching worker jobs"""
        self.fire_event_handler(AsyncWorkflowEvents.STARTED)

        num_workers = len(self.topology.workers)
        max_jobs = num_workers * self.num_global_rounds
        jobs_completed = 0
        worker_ids = [worker.idx for worker in self.topology.workers]
        worker_idx_map = {worker.idx: worker for worker in self.topology.workers}

        # Assign one job to each worker at the start
        futures = set()
        for worker in self.topology.workers:
            if self.state.worker_rounds[worker.idx] < self.num_global_rounds:
                futures.add(self._dispatch_worker_job(worker))
                self.state.worker_rounds[worker.idx] += 1

        while futures and jobs_completed < max_jobs:
            dones, futures = wait(futures, return_when=FIRST_COMPLETED)

            for future in dones:
                try:
                    result: Result = future.result()
                except Exception as exc:
                    print(f"Worker job failed with exception: {exc}")
                    continue
                worker_node_id = result.node.idx

                if result.params is not None:
                    self.state.worker_params[worker_node_id] = result.params
                
                if self.worker_time_tracker is not None:
                    self.worker_time_tracker.record_job_end(worker_node_id)

                self.fire_event_handler(
                    AsyncWorkflowEvents.WORKER_JOB_COMPLETED, {"result": result}
                )

                if self.aggregation_policy:
                    self.aggregation_policy(self, worker_node_id)
                else:
                    self.partial_aggregation_policy(last_updated_node=worker_node_id)
                self.state.completed_worker_jobs += 1
                jobs_completed += 1

                # Assign a new job to this worker if they haven't reached their max rounds
                if self.state.worker_rounds[worker_node_id] < self.num_global_rounds:
                    new_future = self._dispatch_worker_job(worker_idx_map[worker_node_id])
                    futures.add(new_future)
                    self.state.worker_rounds[worker_node_id] += 1

        self.fire_event_handler(AsyncWorkflowEvents.COMPLETED)
        self.module.set_params(self.state.global_params)
        return self.module, None 

    def _dispatch_worker_job(self, worker_node: 'Node') -> Future:
        """"Dispatches a worker job to the specified worker node."""

        worker_dataset = self._get_dataset_for_worker(worker_node.idx)
        args = WorkerJobArgs(
            strategy=self.strategy,
            model=self.module,
            data=worker_dataset,
            params=self.state.global_params,
            node=worker_node,
        )
        # Record job start
        if self.worker_time_tracker is not None:
            self.worker_time_tracker.record_job_start(worker_node.idx)
        future = self.runtime.submit(worker_job, args)
        self.fire_event_handler(AsyncWorkflowEvents.WORKER_JOB_STARTED, {"worker_id": worker_node.idx})
        return future

    def partial_aggregation_policy(self, last_updated_node: t.Optional[int] = None):
        """
        Implements partial aggregation policy using the FedAvg algorithm.
        Aggregates model parameters from all workers using weighted averaging,
        where each worker's weight is proportional to the number of data samples it holds.
        """
        if not self.state.worker_params:
            return

        valid_params = {
            node_id: params
            for node_id, params in self.state.worker_params.items()
            if params is not None
        }
        if not valid_params:
            return

        
        n_k = {}
        for node_id in valid_params:
            worker_dataset = self._get_dataset_for_worker(node_id)
            n_k[node_id] = len(worker_dataset)
        n = sum(n_k.values())
        if n == 0:
            return  # Avoid division by zero
        weights = {node_id: n_k[node_id] / n for node_id in valid_params}

        first_params = next(iter(valid_params.values()))
        aggregated_params = deepcopy(first_params) 
        for key in aggregated_params:
            aggregated_params[key] = aggregated_params[key] * 0.0  

        for node_id, params in valid_params.items():
            for key in aggregated_params:
                if key in params:
                    aggregated_params[key] += params[key] * weights[node_id]

        self.state.global_params = aggregated_params

        self.fire_event_handler(
            AsyncWorkflowEvents.AGGREGATION_COMPLETED,
            {
                "completed_worker_jobs": self.state.completed_worker_jobs,
                "num_workers_aggregated": len(valid_params),
                "last_updated_node": last_updated_node,
                "worker_ids": list(valid_params.keys()),
                "weights": weights,
                "n_k": n_k,
                "n": n,
            },
        )

    def _get_dataset_for_worker(self, worker_id: int):
        """Returns a subset of the dataset for the specified worker ID."""
        
        all_workers = list(self.topology.workers)
        worker_indices = list(range(len(self.dataset)))
        try:
            worker_idx = [n.idx for n in all_workers].index(worker_id)
        except ValueError:
            raise ValueError(f"Worker ID {worker_id} not found in topology workers.")
        num_workers = len(all_workers)
        indices_per_worker = len(worker_indices) // num_workers
        start = worker_idx * indices_per_worker
        end = start + indices_per_worker
        if worker_idx == num_workers - 1:
            end = len(worker_indices)
        
        return Subset(self.dataset, worker_indices[start:end])

    def add_event_handler(self, event_type, handler):
        add_event_handler_to_obj(self, event_type, handler)

    def fire_event_handler(self, event_type, context=None):
        fire_event_handler_by_type(self, event_type, context)

